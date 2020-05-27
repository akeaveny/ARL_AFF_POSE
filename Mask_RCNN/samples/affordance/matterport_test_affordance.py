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

# ========= load dataset (optional) =========
import matterport_dataset_affordance as Affordance


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf

# Path to trained weights file
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

def detect_and_get_masks(model, data_path, num_frames):

    dataset = Affordance.AffordanceDataset()
    dataset.load_Affordance(data_path, 'test')
    dataset.prepare()

    print("Images: {}\nClasses: {}\n".format(len(dataset.image_ids), dataset.class_names))

    for _ in range(0, num_frames):

        #  [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #  24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42]
        print("\n", dataset.image_ids)
        image_id = random.choice(dataset.image_ids)
        # image_id = 10

        # ''' ==================== DETECT ==================== '''
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        # ## ============== SYNTHETIC ===================
        if image.shape[-1] == 4:
            image = image[..., :3]

        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        print("Original image shape: ",
              modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

        # Run object detection
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

        # Display results
        r = results[0]
        log("gt_class_id", gt_class_id)
        log("gt_bbox", gt_bbox)
        log("gt_mask", gt_mask)

        # Compute AP over range 0.5 to 0.95 and print it
        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                               r['rois'], r['class_ids'], r['scores'], r['masks'],
                               verbose=1)

        visualize.display_differences(
            image,
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            dataset.class_names, ax=get_ax(),
            show_box=False, show_mask=False,
            iou_threshold=0.5, score_threshold=0.5)

        # # ================================
        # Display Ground Truth only
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names, ax=get_ax(1),
                                    show_bbox=False, show_mask=False,
                                    title="Ground Truth")

        # # ================================
        # Display OG
        # ax = get_ax(1)
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset.class_names, r['scores'], ax=ax,
        #                             title="Predictions")

        # # ========== precision-recall ============
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])

        # visualize.plot_precision_recall(AP, precisions, recalls)
        visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
                                overlaps, dataset.class_names)

        # # # ========== batch mAP ============
        # # Run on validation set
        # limit = 5
        # APs = compute_batch_ap(dataset, dataset.image_ids[:limit])
        # print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

        ''' ==================== RPN ==================== '''
        # Generate RPN trainig targets
        # target_rpn_match is 1 for positive anchors, -1 for negative anchors
        # and 0 for neutral anchors.
        # target_rpn_match, target_rpn_bbox = modellib.build_rpn_targets(
        #     image.shape, model.anchors, gt_class_id, gt_bbox, model.config)
        # log("target_rpn_match", target_rpn_match)
        # log("target_rpn_bbox", target_rpn_bbox)
        #
        # positive_anchor_ix = np.where(target_rpn_match[:] == 1)[0]
        # negative_anchor_ix = np.where(target_rpn_match[:] == -1)[0]
        # neutral_anchor_ix = np.where(target_rpn_match[:] == 0)[0]
        # positive_anchors = model.anchors[positive_anchor_ix]
        # negative_anchors = model.anchors[negative_anchor_ix]
        # neutral_anchors = model.anchors[neutral_anchor_ix]
        # log("positive_anchors", positive_anchors)
        # log("negative_anchors", negative_anchors)
        # log("neutral anchors", neutral_anchors)
        #
        # # Apply refinement deltas to positive anchors
        # refined_anchors = utils.apply_box_deltas(
        #     positive_anchors,
        #     target_rpn_bbox[:positive_anchors.shape[0]] * model.config.RPN_BBOX_STD_DEV)
        # log("refined_anchors", refined_anchors, )

        # ====================================================
        # Display positive anchors before refinement (dotted) and
        # after refinement (solid).
        # visualize.draw_boxes(image, boxes=positive_anchors, refined_boxes=refined_anchors, ax=get_ax())

        ''' ==================== ANCHORS ==================== '''
        # # ====================================================
        # # Run RPN sub-graph
        # pillar = model.keras_model.get_layer("ROI").output  # node to start searching from
        #
        # # TF 1.4 and 1.9 introduce new versions of NMS. Search for all names to support TF 1.3~1.10
        # nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression:0")
        # if nms_node is None:
        #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV2:0")
        # if nms_node is None:  # TF 1.9-1.10
        #     nms_node = model.ancestor(pillar, "ROI/rpn_non_max_suppression/NonMaxSuppressionV3:0")
        #
        # rpn = model.run_graph([image], [
        #     ("rpn_class", model.keras_model.get_layer("rpn_class").output),
        #     ("pre_nms_anchors", model.ancestor(pillar, "ROI/pre_nms_anchors:0")),
        #     ("refined_anchors", model.ancestor(pillar, "ROI/refined_anchors:0")),
        #     ("refined_anchors_clipped", model.ancestor(pillar, "ROI/refined_anchors_clipped:0")),
        #     ("post_nms_anchor_ix", nms_node),
        #     ("proposals", model.keras_model.get_layer("ROI").output),
        # ])
        #
        # # ====================================================
        # # Show top anchors by score (before refinement)
        # limit = 100
        # sorted_anchor_ids = np.argsort(rpn['rpn_class'][:, :, 1].flatten())[::-1]
        # visualize.draw_boxes(image, boxes=model.anchors[sorted_anchor_ids[:limit]], ax=get_ax())
        #
        # # ====================================================
        # # Show top anchors with refinement. Then with clipping to image boundaries
        # limit = 50
        # ax = get_ax(1, 2)
        # pre_nms_anchors = utils.denorm_boxes(rpn["pre_nms_anchors"][0], image.shape[:2])
        # refined_anchors = utils.denorm_boxes(rpn["refined_anchors"][0], image.shape[:2])
        # refined_anchors_clipped = utils.denorm_boxes(rpn["refined_anchors_clipped"][0], image.shape[:2])
        # visualize.draw_boxes(image, boxes=pre_nms_anchors[:limit],
        #                      refined_boxes=refined_anchors[:limit], ax=ax[0])
        # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[:limit], ax=ax[1])
        #
        # # ====================================================
        # # Show refined anchors after non-max suppression
        # limit = 50
        # ixs = rpn["post_nms_anchor_ix"][:limit]
        # visualize.draw_boxes(image, refined_boxes=refined_anchors_clipped[ixs], ax=get_ax())
        #
        # # ====================================================
        # # Show final proposals
        # # These are the same as the previous step (refined anchors
        # # after NMS) but with coordinates normalized to [0, 1] range.
        # limit = 50
        # # Convert back to image coordinates for display
        # h, w = config.IMAGE_SHAPE[:2]
        # proposals = rpn['proposals'][0, :limit] * np.array([h, w, h, w])
        # visualize.draw_boxes(image, refined_boxes=proposals, ax=get_ax())
        #
        # # ====================================================
        # # Measure the RPN recall (percent of objects covered by anchors)
        # # Here we measure recall for 3 different methods:
        # # - All anchors
        # # - All refined anchors
        # # - Refined anchors after NMS
        # iou_threshold = 0.7
        #
        # recall, positive_anchor_ids = utils.compute_recall(model.anchors, gt_bbox, iou_threshold)
        # print("All Anchors ({:5})       Recall: {:.3f}  Positive anchors: {}".format(
        #     model.anchors.shape[0], recall, len(positive_anchor_ids)))
        #
        # recall, positive_anchor_ids = utils.compute_recall(rpn['refined_anchors'][0], gt_bbox, iou_threshold)
        # print("Refined Anchors ({:5})   Recall: {:.3f}  Positive anchors: {}".format(
        #     rpn['refined_anchors'].shape[1], recall, len(positive_anchor_ids)))
        #
        # recall, positive_anchor_ids = utils.compute_recall(proposals, gt_bbox, iou_threshold)
        # print("Post NMS Anchors ({:5})  Recall: {:.3f}  Positive anchors: {}".format(
        #     proposals.shape[0], recall, len(positive_anchor_ids)))

        ''' ==================== MASKS ==================== '''
        # # ========================
        # display_images(np.transpose(gt_mask, [2, 0, 1]), cmap="Blues")
        #
        # # ========================
        # # Get predictions of mask head
        # mrcnn = model.run_graph([image], [
        #     ("detections", model.keras_model.get_layer("mrcnn_detection").output),
        #     ("masks", model.keras_model.get_layer("mrcnn_mask").output),
        # ])
        #
        # # Get detection class IDs. Trim zero padding.
        # det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
        # det_count = np.where(det_class_ids == 0)[0][0]
        # det_class_ids = det_class_ids[:det_count]
        #
        # print("{} detections: {}".format(
        #     det_count, np.array(dataset.class_names)[det_class_ids]))
        #
        # # ========================
        # # Masks
        # det_boxes = utils.denorm_boxes(mrcnn["detections"][0, :, :4], image.shape[:2])
        # det_mask_specific = np.array([mrcnn["masks"][0, i, :, :, c]
        #                               for i, c in enumerate(det_class_ids)])
        # det_masks = np.array([utils.unmold_mask(m, det_boxes[i], image.shape)
        #                       for i, m in enumerate(det_mask_specific)])
        # log("det_mask_specific", det_mask_specific)
        # log("det_masks", det_masks)
        #
        # # ========================
        # display_images(det_mask_specific[:4] * 255, cmap="Blues", interpolation="none")
        #
        # # ========================
        # display_images(det_masks[:4] * 255, cmap="Blues", interpolation="none")

        ''' ==================== ACTIVATIONS ==================== '''
        # model.keras_model.summary()
        # ==============================
        # Get activations of a few sample layers
        # activations = model.run_graph([image], [
        #     ("input_image", tf.identity(model.keras_model.get_layer("input_image").output)),
        #     ("res2c_out", model.keras_model.get_layer("res2c_out").output),
        #     ("res3c_out", model.keras_model.get_layer("res3c_out").output),
        #     ("res4f_out", model.keras_model.get_layer("res4f_out").output),  # for resnet50
        #     ("res5c_out", model.keras_model.get_layer("res5c_out").output),  # for resnet50
        #     ("rpn_bbox", model.keras_model.get_layer("rpn_bbox").output),
        #     ("roi", model.keras_model.get_layer("ROI").output),
        # ])
        #
        # # # ==============================
        # # Backbone feature map
        # # Input image (normalized)
        # _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
        # display_images(np.transpose(activations["res2c_out"][0, :, :, :4], [2, 0, 1]), cols=4)
        #
        # # Input image (normalized)
        # _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
        # display_images(np.transpose(activations["res3c_out"][0, :, :, :4], [2, 0, 1]), cols=4)
        #
        # # Input image (normalized)
        # _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
        # display_images(np.transpose(activations["res4f_out"][0, :, :, :4], [2, 0, 1]), cols=4)
        #
        # # Input image (normalized)
        # _ = plt.imshow(modellib.unmold_image(activations["input_image"][0], config))
        # display_images(np.transpose(activations["res5c_out"][0, :, :, :4], [2, 0, 1]), cols=4)

        plt.show()

    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
                        description='Train Mask R-CNN to detect Warehouses.')
    parser.add_argument('--data', required=False,
                        metavar="/path/to/data/",
                        help='Directory of the Warehouse dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--num_frames', type=int, default=100, help='number of images')
    
    args = parser.parse_args()

    class InferenceConfig(Affordance.AffordanceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    # ============= weights statistics ==========
    # visualize.display_weight_stats(model)

    detect_and_get_masks(model, args.data, args.num_frames)