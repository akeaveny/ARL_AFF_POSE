"""
Mask R-CNN
Train on the Warehouse dataset and implement color splash effect.

Based on the work of Waleed Abdulla (Matterport)
Modified by Dinh-Cuong Hoang

------------------------------------------------------------

python3 train.py --dataset=/Warehouse_Dataset/data --weights=coco
python3 train.py --dataset=/Warehouse_Dataset/data --weights=last

"""

import os
import sys
import itertools
import math
import logging
import json
import re
import random
import time
import concurrent.futures
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import imgaug
from imgaug import augmenters as iaa

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')

parser.add_argument('--dataset', required=False,
                    # default='/data/Akeaveny/Datasets/part-affordance_combined/real/',
                    default='/data/Akeaveny/Datasets/part-affordance_combined/ndds4/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='syn',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='val',
                    type=str,
                    metavar='train or val')
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
### assert args.dataset_type == 'real' or args.dataset_type == 'syn'
if args.dataset_type == 'real':
    import dataset_real as Affordance
    save_to_folder = '/images/dataset_images_real/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'syn':
    import dataset_syn as Affordance
    save_to_folder = '/images/dataset_images_syn/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'syn1':
    import dataset_syn1 as Affordance
    save_to_folder = '/images/dataset_images_syn1/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'hammer':
    import objects.dataset_syn_hammer as Affordance
    save_to_folder = '/images/objects/dataset_images_syn_hammer/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'hammer1':
    import objects.dataset_syn_hammer1 as Affordance
    save_to_folder = '/images/objects/dataset_images_syn_hammer1/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'scissors_real':
    import objects.dataset_real_scissors as Affordance
    save_to_folder = '/images/objects/dataset_images_real_scissors/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'scissors':
    import objects.dataset_syn_scissors as Affordance
    save_to_folder = '/images/objects/dataset_images_syn_scissors/'
    image_area_bins = [480 * 640]
elif args.dataset_type == 'scissors_20k':
    import objects.dataset_syn_scissors_20k as Affordance
    save_to_folder = '/images/objects/dataset_images_syn_scissors_20k/'
    image_area_bins = [480 * 640]


if not (os.path.exists(os.getcwd()+save_to_folder)):
    os.makedirs(os.getcwd()+save_to_folder)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
print("ROOT_DIR: ", ROOT_DIR)

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# ##################################
# ###  GPU
# ##################################

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# config_ = tf.ConfigProto()
# config_.gpu_options.allow_growth = True
# sess = tf.Session(config=config_)

from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log

from mrcnn import model as modellib, utils

############################################################
#  utility functions
############################################################

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def image_stats(image_id):
    """Returns a dict of stats for one image."""
    image = dataset.load_image(image_id)
    mask, _ = dataset.load_mask(image_id)
    bbox = utils.extract_bboxes(mask)
    # Sanity check
    assert mask.shape[:2] == image.shape[:2]
    # Return stats dict
    return {
        "id": image_id,
        "shape": list(image.shape),
        "bbox": [[b[2] - b[0], b[3] - b[1]]
                 for b in bbox
                 # Uncomment to exclude nuclei with 1 pixel width
                 # or height (often on edges)
                 # if b[2] - b[0] > 1 and b[3] - b[1] > 1
                ],
        "color": np.mean(image, axis=(0, 1)),
    }

# class RandomCropConfig(Affordance.AffordanceConfig):
#     IMAGE_RESIZE_MODE = "crop"
#     IMAGE_MIN_DIM = 400
#     IMAGE_MAX_DIM = 400

#########################################################################################################
# MAIN
#########################################################################################################
''' --- based on https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/inspect_nucleus_data.ipynb --- '''

if __name__ == '__main__':
    np.random.seed(5)

    if args.save_output:
        sys.stdout = open(os.getcwd() + save_to_folder + 'output.txt', "w")
    else:
        pass

    assert args.dataset, "Argument --dataset is required for training"

    config = Affordance.AffordanceConfig()

    dataset = Affordance.AffordanceDataset()
    dataset.load_Affordance(args.dataset, subset=args.dataset_split)
    dataset.prepare()  # Must call before using the dataset

    config.display()

    captions = np.array(dataset.class_names)
    ### print("Classes: {}".format(captions))

    ##################################
    ###  PRELIM
    ##################################
    print('\n --------------- Prelim ---------------')

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as e:
        stats = list(e.map(image_stats, dataset.image_ids))
    t_total = time.time() - t_start
    print("Total time: {:.1f} seconds".format(t_total))

    # # Image stats
    image_shape = np.array([s['shape'] for s in stats])
    image_color = np.array([s['color'] for s in stats])
    print("Class Count: {}".format(dataset.num_classes))
    print("Image Count: ", image_shape.shape[0])
    print("Color mean (RGB):{:.2f} {:.2f} {:.2f}".format(*np.mean(image_color, axis=0)))

    ##################################
    ###  Display Samples
    ##################################
    for idx_samples in range(10):
        print('\n --------------- Samples ---------------')

        num_images = 4
        # get random image
        ### image_ids = np.random.choice(dataset.image_ids, size=len(dataset.image_ids))
        image_ids = np.random.choice(len(dataset.image_ids), size=num_images)
        image_id = image_ids[0]
        print("image_id:", image_id)
        print("image_file:", dataset.image_reference(image_id))

        # Load the image multiple times to show augmentations
        limit = num_images
        ax = get_ax(rows=int(np.sqrt(limit)), cols=int(np.sqrt(limit)))
        for i in range(limit):
            image_idx =image_ids[i]
            image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(dataset, config, image_idx,
                                                                              use_mini_mask=False)
            visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names,
                                        ax=ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))],
                                        captions=captions[class_ids].tolist())

            depth = np.array(image.copy())
            print("\tDEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))

            # log("molded_image", image)
            # log("mask", mask)
            # log("class_ids", class_ids)
            ### print("captions", np.array(dataset.class_names)[class_ids].tolist())
        plt.savefig(os.getcwd() + save_to_folder + "gt_affordance_labels/gt_affordance_labels_" + np.str(idx_samples) + ".png", bbox_inches='tight')

    ##################################
    ###  Image Size Stats
    ##################################
    print('\n --------------- Image Size ---------------')

    image_shape = np.array([s['shape'] for s in stats])
    image_color = np.array([s['color'] for s in stats])
    print("Height  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 0]), np.median(image_shape[:, 0]),
        np.min(image_shape[:, 0]), np.max(image_shape[:, 0])))
    print("Width   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(image_shape[:, 1]), np.median(image_shape[:, 1]),
        np.min(image_shape[:, 1]), np.max(image_shape[:, 1])))

    # Histograms
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].set_title("Height")
    _ = ax[0].hist(image_shape[:, 0], bins=20)
    ax[1].set_title("Width")
    _ = ax[1].hist(image_shape[:, 1], bins=20)
    ax[2].set_title("Height & Width")
    _ = ax[2].hist2d(image_shape[:, 1], image_shape[:, 0], bins=10, cmap="Blues")
    plt.savefig(os.getcwd() + save_to_folder + "histogram_of_image_size.png", bbox_inches='tight')

    ##################################
    ###  Histogram
    ##################################
    print('\n --------------- Histograms ---------------')

    fig, ax = plt.subplots(1, len(image_area_bins), figsize=(16, 4))
    area_threshold = 0
    for i, image_area in enumerate(image_area_bins):
        object_shape = np.array([
            b
            for s in stats if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area
            for b in s['bbox']])
        object_area = object_shape[:, 0] * object_shape[:, 1]
        area_threshold = image_area

        print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)))
        print("  Total Objects: ", object_shape.shape[0])
        print("  Object Height. mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
            np.mean(object_shape[:, 0]), np.median(object_shape[:, 0]),
            np.min(object_shape[:, 0]), np.max(object_shape[:, 0])))
        print("  Object Width.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
            np.mean(object_shape[:, 1]), np.median(object_shape[:, 1]),
            np.min(object_shape[:, 1]), np.max(object_shape[:, 1])))
        print("  Object Area.   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
            np.mean(object_area), np.median(object_area),
            np.min(object_area), np.max(object_area)))

        # Show 2D histogram
        _ = ax.hist2d(object_shape[:, 1], object_shape[:, 0], bins=20, cmap="Blues")
        plt.savefig(os.getcwd() + save_to_folder + "histogram_of_shapes.png", bbox_inches='tight')

    # Objects height/width ratio
    object_aspect_ratio = object_shape[:, 0] / (object_shape[:, 1]+1e-6)
    print("Object Aspect Ratio.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
        np.mean(object_aspect_ratio), np.median(object_aspect_ratio),
        np.min(object_aspect_ratio), np.max(object_aspect_ratio)))
    plt.figure(figsize=(15, 5))
    _ = plt.hist(object_aspect_ratio, bins=100, range=[0, 5])
    plt.savefig(os.getcwd() + save_to_folder + "height_width_ratio.png", bbox_inches='tight')

    ##################################
    # IMG AUG
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    ##################################
    print('\n --------------- IMGAUG ---------------')

    # augmentation = iaa.Sometimes(0.9, [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.Multiply((0.8, 1.2)),
    #     iaa.GaussianBlur(sigma=(0.0, 2.0)),
    #     iaa.OneOf([iaa.Affine(rotate=90),
    #                iaa.Affine(rotate=180),
    #                iaa.Affine(rotate=270)]),
    # ])

    augmentation = iaa.Sometimes(0.833, iaa.Sequential([
        #########################
        # COLOR & MASK
        #########################
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-25, 25),
            # shear=(-8, 8)
        ),
        #########################
        # ONLY COLOR !!!
        #########################
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.25)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True))  # apply augmenters in random order

    # Load the image multiple times to show augmentations
    limit = 16
    ax = get_ax(rows=int(np.sqrt(limit)), cols=int(np.sqrt(limit)))
    for i in range(limit):
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False, augment=False, augmentation=augmentation)
        visualize.display_instances(image, bbox, mask, class_ids,
                                    dataset.class_names, ax=ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))],
                                    show_mask=False, show_bbox=False)
    plt.savefig(os.getcwd() + save_to_folder + "imgaug.png", bbox_inches='tight')

    ##################################
    # CROPPING
    ##################################

    crop_config = config
    # Load the image multiple times to show augmentations
    limit = num_images
    ax = get_ax(rows=int(np.sqrt(limit)), cols=int(np.sqrt(limit)))
    for i in range(limit):
        image_idx = image_ids[i]
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt( dataset, crop_config, image_id,
                                                                           augmentation=augmentation, use_mini_mask=False)
        visualize.display_instances(image, bbox, mask, class_ids,
                                    dataset.class_names, ax=ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))],
                                    show_mask=True, show_bbox=True,
                                    captions=captions[class_ids].tolist())
    plt.savefig(os.getcwd() + save_to_folder + "cropped_affordance_labels.png", bbox_inches='tight')

    ##################################
    # Masks
    ##################################
    for i in range(4):
        print('\n --------------- Masks ---------------')

        image_id_ = int(np.random.choice(len(dataset.image_ids), size=1))
        image = dataset.load_image(image_id_)
        mask, class_ids = dataset.load_mask(image_id_)
        original_shape = image.shape

        ### config = RandomCropConfig()
        print("Min: ", config.IMAGE_MIN_DIM)
        print("Max: ", config.IMAGE_MAX_DIM)
        print("Resize Mode: ", config.IMAGE_RESIZE_MODE)

        # Resize
        image, window, scale, padding, _ = utils.resize_image(
            image, min_dim=config.IMAGE_MIN_DIM, max_dim=config.IMAGE_MAX_DIM, mode=config.IMAGE_RESIZE_MODE)
        mask = utils.resize_mask(mask, scale, padding)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)

        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset, config, image_id_, use_mini_mask=False)
        log("Original mask: ", mask)

        display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
        plt.savefig(os.getcwd() + save_to_folder + "masks/mask_og_" + np.str(i) +".png", bbox_inches='tight')

        # Add augmentation and mask resizing.
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset, config, image_id_, use_mini_mask=True)
        log("Mini Mask", mask)

        display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
        plt.savefig(os.getcwd() + save_to_folder + "masks/mask_mini_" + np.str(i) +".png", bbox_inches='tight')

    ##################################
    # Anchors
    ##################################
    print('\n --------------- Anchors ---------------')

    ### Visualize anchors of one cell at the center of the feature map
    image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)

    ### Generate Anchors
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    ### Print summary of anchors
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))

    ### Display
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    levels = len(backbone_shapes)
    for level in range(levels):
        colors = visualize.random_colors(levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level])  # sum of anchors of previous levels
        level_anchors = anchors[level_start:level_start + anchors_per_level[level]]
        print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
                                                                      backbone_shapes[level]))
        center_cell = backbone_shapes[level] // 2
        center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
        level_center = center_cell_index * anchors_per_cell
        center_anchor = anchors_per_cell * (
                (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE ** 2) \
                + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center + anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, facecolor='none',
                                  edgecolor=(i + 1) * np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)
    plt.savefig(os.getcwd() + save_to_folder + "anchors.png", bbox_inches='tight')

    #########################################
    # POSTIVE, NEGATIVE & NEUTRAL ANCHORS
    #########################################

    # Create data generator
    random_rois = config.TRAIN_ROIS_PER_IMAGE
    g = modellib.data_generator(
        dataset, crop_config, shuffle=True,
        random_rois=random_rois,
        batch_size=4,
        detection_targets=True)

    # Get Next Image
    if random_rois:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
        [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)

        log("rois", rois)
        log("mrcnn_class_ids", mrcnn_class_ids)
        log("mrcnn_bbox", mrcnn_bbox)
        log("mrcnn_mask", mrcnn_mask)
    else:
        [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)

    log("gt_class_ids", gt_class_ids)
    log("gt_boxes", gt_boxes)
    log("gt_masks", gt_masks)
    log("rpn_match", rpn_match[:len(dataset.image_ids)])
    log("rpn_bbox", rpn_bbox)
    # image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
    # print("image_id: ", image_id, dataset.image_reference(image_id))

    # Remove the last dim in mrcnn_class_ids. It's only added
    # to satisfy Keras restriction on target shape.
    mrcnn_class_ids = mrcnn_class_ids[:, :, 0]

    b = 0

    # Restore original image (reverse normalization)
    sample_image = modellib.unmold_image(normalized_images[b], config)

    # Compute anchor shifts.
    indices = np.where(rpn_match[b] == 1)[0]
    print("indices", indices)

    refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    log("anchors", anchors)
    log("refined_anchors", refined_anchors)

    # Get list of positive anchors
    positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    print("Positive anchors: {}".format(len(positive_anchor_ids)))
    negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    print("Negative anchors: {}".format(len(negative_anchor_ids)))
    neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    print("Neutral anchors: {}".format(len(neutral_anchor_ids)))

    # ROI breakdown by class
    print("Breakdwon of ROIs per class")
    for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
        if n:
            print("{:23}: {}".format(c[:20], n))

    # Show positive anchors
    fig, ax = plt.subplots(1, figsize=(16, 16))
    visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
                         refined_boxes=refined_anchors, ax=ax)
    plt.savefig(os.getcwd() + save_to_folder + "anchors_positive.png", bbox_inches='tight')

    # Show negative anchors
    visualize.draw_boxes(sample_image, boxes=anchors[negative_anchor_ids])
    plt.savefig(os.getcwd() + save_to_folder + "anchors_negative.png", bbox_inches='tight')

    # Show neutral anchors. They don't contribute to training.
    visualize.draw_boxes(sample_image, boxes=anchors[np.random.choice(neutral_anchor_ids, 100)])
    plt.savefig(os.getcwd() + save_to_folder + "anchors_neutral.png", bbox_inches='tight')

    #########################################
    # RoIs
    #########################################
    print('\n --------------- RoIs ---------------')

    ### Check ratio of positive ROIs in a set of images.
    if random_rois:
        limit = 10
        temp_g = modellib.data_generator(
            dataset, crop_config, shuffle=True, random_rois=10000,
            batch_size=1, detection_targets=True)
        total = 0
        for i in range(limit):
            _, [ids, _, _] = next(temp_g)
            positive_rois = np.sum(ids[0] > 0)
            total += positive_rois
            print("{:5} {:5.2f}".format(positive_rois, positive_rois / ids.shape[1]))
        print("Average percent: {:.2f}".format(total / (limit * ids.shape[1])))

    ### Dispalay ROIs and corresponding masks and bounding boxes
    if random_rois:
        # Class aware bboxes
        bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]

        # Refined ROIs
        refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:, :4] * config.BBOX_STD_DEV)

        # Class aware masks
        mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]

        visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)

        # Any repeated ROIs?
        rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
        _, idx = np.unique(rows, return_index=True)
        print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))

    if random_rois:
        # Dispalay ROIs and corresponding masks and bounding boxes
        ids = random.sample(range(rois.shape[1]), 8)

        images = []
        titles = []
        for i in ids:
            image = visualize.draw_box(sample_image.copy(), rois[b, i, :4].astype(np.int32), [255, 0, 0])
            image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
            images.append(image)
            titles.append("ROI {}".format(i))
            images.append(mask_specific[i] * 255)
            titles.append(dataset.class_names[mrcnn_class_ids[b, i]][:20])

        display_images(images, titles, cols=4, cmap="Blues", interpolation="none")
    plt.savefig(os.getcwd() + save_to_folder + "random_rois.png", bbox_inches='tight')

    print("show_plots", args.show_plots)
    if args.show_plots: # TODO: boolean string
        plt.show()

    if args.save_output:
        sys.stdout.close()
    else:
        pass

