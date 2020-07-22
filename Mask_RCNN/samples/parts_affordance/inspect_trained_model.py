"""
Mask R-CNN for Object_RPE
------------------------------------------------------------
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random

import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
print("ROOT_DIR: ", ROOT_DIR)

# Path to trained weights file
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')
parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/real/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='real',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')
parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/ or Logs and checkpoints directory (default=logs/)")
parser.add_argument('--num_frames', type=int, default=1, help='number of images')
parser.add_argument('--show_plots', required=False, default=False,
                    type=bool,
                    metavar='show plots from matplotlib')
parser.add_argument('--save_output', required=False, default=False,
                    type=bool,
                    metavar='save terminal output to text file')
args = parser.parse_args()

############################################################
#  REAL OR SYN
############################################################
assert args.dataset_type == 'real' or args.dataset_type == 'syn' or args.dataset_type == 'syn1' or args.dataset_type == 'hammer'
if args.dataset_type == 'real':
    import dataset_real as Affordance
    save_to_folder = '/images/test_images_real/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn':
    import dataset_syn as Affordance
    save_to_folder = '/images/test_images_syn/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn1':
    import dataset_syn1 as Affordance
    save_to_folder = '/images/test_images_syn1/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'hammer':
    import dataset_syn_hammer as Affordance
    save_to_folder = '/images/test_images_syn_hammer/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 1280
    IMAGE_MAX_DIM_ = 1280

if not (os.path.exists(os.getcwd()+save_to_folder)):
    os.makedirs(os.getcwd()+save_to_folder)

from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf

###########################################################
# Test
###########################################################

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)

        # Run object detection
        results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        APs.append(ap)
        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis, ...])
            print("{:3} {}   AP: {:.2f}".format(
                meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs

def detect_and_get_masks(model, config, args):
    np.random.seed(1)

    if args.save_output:
        sys.stdout = open(os.getcwd() + save_to_folder + 'output.txt', "w")
    else:
        pass

    ########################
    #  Load test images
    ########################

    print("args.dataset_split", args.dataset_split)
    dataset = Affordance.AffordanceDataset()
    dataset.load_Affordance(args.dataset, args.dataset_split)
    dataset.prepare()

    #### print KERAS model
    #### model.keras_model.summary()
    config.display()

    captions = np.array(dataset.class_names)

    print("Num of Test Images: {}".format(len(dataset.image_ids)))

    ########################
    #  batch mAP
    ########################
    print('\n --------------- mAP ---------------')

    # limit = len(dataset.image_ids)
    # APs = compute_batch_ap(dataset, dataset.image_ids[:limit])
    # print("Mean AP over {} test images: {:.4f}".format(len(APs), np.mean(APs)))

    # for idx in range(len(dataset.image_ids)):
        # print("\n", dataset.image_ids)
        # image_id = random.choice(dataset.image_ids)
        # image_id = 9
        # image_id = idx

    ########################
    #  Detect
    ########################
    print('\n --------------- Detection ---------------')

    image_ids = np.random.choice(len(dataset.image_ids), size=16)

    # Load the image multiple times to show augmentations
    limit = 16
    ax = get_ax(rows=int(np.sqrt(limit)), cols=int(np.sqrt(limit)))
    for i in range(limit):
        # load images
        image_id =image_ids[i]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        # run detect
        results = model.detect([image], verbose=1)
        r = results[0]
        class_ids = r['class_ids'] - 1

        # plot
        visualize.display_instances(image, r['rois'], r['masks'], class_ids, dataset.class_names, r['scores'],
                                    ax=ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))],
                                    title="Predictions", show_bbox=True, show_mask=True)

        # visualize.display_differences(image, gt_bbox, gt_class_id, gt_mask,
        #                                 r['rois'], r['class_ids'], r['scores'], r['masks'],
        #                                 dataset.class_names,
        #                                 ax=ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))],
        #                                 show_box=False, show_mask=False,
        #                                 iou_threshold=0.5, score_threshold=0.5)

    plt.savefig(os.getcwd() + save_to_folder + "gt_affordance_labels.png", bbox_inches='tight')

    #################
    # Activations
    #################
    print('\n --------------- Activations ---------------')


    # Get activations of a few sample layers
    activations = model.run_graph([image], [
        ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
        ("res2c_out", model.keras_model.get_layer("res2c_out").output),
        ("res3c_out", model.keras_model.get_layer("res3c_out").output),
        ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
        ("roi", model.keras_model.get_layer("ROI").output),
    ])

    # Backbone feature map
    display_images(np.transpose(activations["res2c_out"][0, :, :, :4], [2, 0, 1]), cols=4)
    plt.savefig(os.getcwd() + save_to_folder + "activations_res2c_out.png", bbox_inches='tight')
    display_images(np.transpose(activations["res3c_out"][0, :, :, :4], [2, 0, 1]), cols=4)
    plt.savefig(os.getcwd() + save_to_folder + "activations_res3c_out.png", bbox_inches='tight')

    ########################
    #  RPN
    ########################
    print('\n --------------- RPNs ---------------')

    limit = 10

    # Get anchors and convert to pixel coordinates
    anchors = model.get_anchors(image.shape)
    anchors = utils.denorm_boxes(anchors, image.shape[:2])
    log("anchors", anchors)

    # Generate RPN trainig targets
    # target_rpn_match is 1 for positive anchors, -1 for negative anchors
    # and 0 for neutral anchors.
    target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        image.shape, anchors, gt_class_id, gt_bbox, model.config)
    log("target_rpn_match", target_rpn_match)
    log("target_rpn_bbox", target_rpn_bbox)

    positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
    negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
    neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
    positive_anchors = anchors[positive_anchor_ix]
    negative_anchors = anchors[negative_anchor_ix]
    neutral_anchors = anchors[neutral_anchor_ix]
    log("positive_anchors", positive_anchors)
    log("negative_anchors", negative_anchors)
    log("neutral anchors", neutral_anchors)

    # Apply refinement deltas to positive anchors
    refined_anchors = utils.apply_box_deltas(
        positive_anchors,
        target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
    log("refined_anchors", refined_anchors, )

    # Display positive anchors before refinement (dotted) and
    # after refinement (solid).
    visualize.draw_boxes(
        image, ax=get_ax(),
        boxes=positive_anchors,
        refined_boxes=refined_anchors)
    plt.savefig(os.getcwd() + save_to_folder + "anchors_positive.png", bbox_inches='tight')

    # Run RPN sub-graph
    pillar = model.keras_model.get_layer("ROI").output  # node to start searching from

    # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
    nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
    if nms_node is None:
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
    if nms_node is None:  # TF 1.9-1.10
        nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")

    rpn = model.run_graph(image[np.newaxis], [
        ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        ("post_nms_anchor_ix", nms_node),
        ("proposals", model.keras_model.get_layer("ROI").output),
    ], image_metas=image_meta[np.newaxis])

    # Show top anchors by score (before refinement)
    sorted_anchor_ids = np.argsort(rpn['rpn_class'][:, :, 1].flatten())[::-1]
    visualize.draw_boxes(image, boxes=anchors[sorted_anchor_ids[:limit]], ax=get_ax())
    plt.savefig(os.getcwd() + save_to_folder + "anchors_top.png", bbox_inches='tight')

    # Show top anchors with refinement. Then with clipping to image boundaries
    ax = get_ax(1, 2)
    pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
    refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
    refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
    visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
                         refined_boxes=refined_anchors[:limit], ax=ax[0])
    visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])
    plt.savefig(os.getcwd() + save_to_folder + "anchors_refinement.png", bbox_inches='tight')

    # Show final proposals
    # These are the same as the previous step (refined anchors
    # after NMS) but with coordinates normalized to [0, 1] range.
    # Convert back to image coordinates for display
    # h, w = config.IMAGE_SHAPE[:2]
    # proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
    visualize.draw_boxes(
        image, ax=get_ax(),
        refined_boxes=utils.denorm_boxes(rpn['proposals'][0, :limit], image.shape[:2]))
    plt.savefig(os.getcwd() + save_to_folder + "final_proposals.png", bbox_inches='tight')

    #############################
    #  Proposal Classification
    #############################
    print('\n --------------- Proposal Classification ---------------')

    # Get input and output to classifier and mask heads.
    mrcnn = model.run_graph([image], [
        ("proposals", model.keras_model.get_layer("ROI").output),
        ("probs", model.keras_model.get_layer("mrcnn_class").output),
        ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]
    detections = mrcnn['detections'][0, :det_count]

    print("{} detections: {}".format(
        det_count, np.array(dataset.class_names)[det_class_ids]))

    captions = ["{} {:.3f}".format(dataset.class_names[int(c)], s) if c > 0 else ""
                for c, s in zip(detections[:, 4], detections[:, 5])]
    visualize.draw_boxes(
        image,
        refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
        visibilities=[2] * len(detections),
        captions=captions, title="Detections",
        ax=get_ax())

    # Proposals are in normalized coordinates
    proposals = mrcnn["proposals"][0]

    # Class ID, score, and mask per proposal
    roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
    roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
    roi_class_names = np.array(dataset.class_names)[roi_class_ids]
    roi_positive_ixs = np.where(roi_class_ids > 0)[0]

    # How many ROIs vs empty rows?
    print("{} Valid proposals out of {}".format(np.sum(np.any(proposals, axis=1)), proposals.shape[0]))
    print("{} Positive ROIs".format(len(roi_positive_ixs)))

    # Class counts
    print(list(zip(*np.unique(roi_class_names, return_counts=True))))

    # Display a random sample of proposals.
    # Proposals classified as background are dotted, and
    # the rest show their class and confidence score.
    limit = 200
    ixs = np.random.randint(0, proposals.shape[0], limit)
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[ixs], roi_scores[ixs])]
    visualize.draw_boxes(
        image,
        boxes=utils.denorm_boxes(proposals[ixs], image.shape[:2]),
        visibilities=np.where(roi_class_ids[ixs] > 0, 2, 1),
        captions=captions, title="ROIs Before Refinement",
        ax=get_ax())
    plt.savefig(os.getcwd() + save_to_folder + "rois_before_refinement.png", bbox_inches='tight')

    # Class-specific bounding box shifts.
    roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
    log("roi_bbox_specific", roi_bbox_specific)

    # Apply bounding box transformations
    # Shape: [N, (y1, x1, y2, x2)]
    refined_proposals = utils.apply_box_deltas(
        proposals, roi_bbox_specific * config.BBOX_STD_DEV)
    log("refined_proposals", refined_proposals)

    # Show positive proposals
    # ids = np.arange(roi_boxes.shape[0])  # Display all
    limit = 5
    ids = np.random.randint(0, len(roi_positive_ixs), limit)  # Display random sample
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[roi_positive_ixs][ids], roi_scores[roi_positive_ixs][ids])]
    visualize.draw_boxes(
        image, ax=get_ax(),
        boxes=utils.denorm_boxes(proposals[roi_positive_ixs][ids], image.shape[:2]),
        refined_boxes=utils.denorm_boxes(refined_proposals[roi_positive_ixs][ids], image.shape[:2]),
        visibilities=np.where(roi_class_ids[roi_positive_ixs][ids] > 0, 1, 0),
        captions=captions, title="ROIs After Refinement")
    plt.savefig(os.getcwd() + save_to_folder + "rois_after_refinement.png", bbox_inches='tight')

    # Remove boxes classified as background
    keep = np.where(roi_class_ids > 0)[0]
    print("Keep {} detections:\n{}".format(keep.shape[0], keep))

    # Remove low confidence detections
    keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
    print("Remove boxes below {} confidence. Keep {}:\n{}".format(
        config.DETECTION_MIN_CONFIDENCE, keep.shape[0], keep))

    # Apply per-class non-max suppression
    pre_nms_boxes = refined_proposals[keep]
    pre_nms_scores = roi_scores[keep]
    pre_nms_class_ids = roi_class_ids[keep]

    nms_keep = []
    for class_id in np.unique(pre_nms_class_ids):
        # Pick detections of this class
        ixs = np.where(pre_nms_class_ids == class_id)[0]
        # Apply NMS
        class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
                                               pre_nms_scores[ixs],
                                               config.DETECTION_NMS_THRESHOLD)
        # Map indicies
        class_keep = keep[ixs[class_keep]]
        nms_keep = np.union1d(nms_keep, class_keep)
        print("{:22}: {} -> {}".format(dataset.class_names[class_id][:20],
                                       keep[ixs], class_keep))

    keep = np.intersect1d(keep, nms_keep).astype(np.int32)
    print("\nKept after per-class NMS: {}\n{}".format(keep.shape[0], keep))

    # Show final detections
    ixs = np.arange(len(keep))  # Display all
    # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
    captions = ["{} {:.3f}".format(dataset.class_names[c], s) if c > 0 else ""
                for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
    visualize.draw_boxes(
        image,
        boxes=utils.denorm_boxes(proposals[keep][ixs], image.shape[:2]),
        refined_boxes=utils.denorm_boxes(refined_proposals[keep][ixs], image.shape[:2]),
        visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
        captions=captions, title="Detections after NMS",
        ax=get_ax())
    plt.savefig(os.getcwd() + save_to_folder + "rois_after_nms.png", bbox_inches='tight')

    ###############
    # MASKS
    ###############
    print('\n --------------- MASKS ---------------')

    limit = 8
    display_images(np.transpose(gt_mask[..., :limit], [2, 0, 1]), cmap="Blues")

    # Get predictions of mask head
    mrcnn = model.run_graph([image], [
        ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        ("masks", model.keras_model.get_layer("mrcnn_mask").output),
    ])

    # Get detection class IDs. Trim zero padding.
    det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
    det_count = np.where(det_class_ids == 0)[0][0]
    det_class_ids = det_class_ids[:det_count]

    print("{} detections: {}".format(
        det_count, np.array(dataset.class_names)[det_class_ids]))

    # Masks
    det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
    det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
                                  for i, c in enumerate(det_class_ids)])
    det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
                          for i, m in enumerate(det_mask_specific)])
    log("det_mask_specific", det_mask_specific)
    log("det_masks", det_masks)

    display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
    plt.savefig(os.getcwd() + save_to_folder + "masks_mini.png", bbox_inches='tight')

    display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")
    plt.savefig(os.getcwd() + save_to_folder + "masks_og.png", bbox_inches='tight')


    print("Show Plots:", args.show_plots)
    if args.show_plots: # TODO: boolean string
        plt.show()

    if args.save_output:
        sys.stdout.close()
    else:
        pass

#########################################################################################################
# MAIN
#########################################################################################################
''' --- based on https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/inspect_nucleus_model.ipynb --- '''
    
if __name__ == '__main__':

    class InferenceConfig(Affordance.AffordanceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        MEAN_PIXEL = MEAN_PIXEL_
        IMAGE_RESIZE_MODE = IMAGE_RESIZE_MODE_
        IMAGE_MIN_DIM = IMAGE_MIN_DIM_
        IMAGE_MAX_DIM = IMAGE_MAX_DIM_
    config = InferenceConfig()
    # config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    detect_and_get_masks(model, config, args)