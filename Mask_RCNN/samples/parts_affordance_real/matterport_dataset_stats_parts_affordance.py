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

# ========= load dataset (optional) =========
import matterport_dataset_parts_affordance as PartsAffordance

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib, utils
from mrcnn.model import log

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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

class RandomCropConfig(PartsAffordance.PartsAffordanceConfig):
    IMAGE_RESIZE_MODE = "none" # "crop"
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

############################################################
#  get stats
############################################################

if __name__ == '__main__':

    '''================= LOAD ================='''
    import argparse
    # =====================
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect PartsAffordance.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/PartsAffordance/dataset/",
                        help='Directory of the PartsAffordance dataset')
    args = parser.parse_args()

    assert args.dataset, "Argument --dataset is required for training"
    print("Dataset: ", args.dataset)

    # =====================
    config = PartsAffordance.PartsAffordanceConfig()
    dataset = PartsAffordance.PartsAffordanceDataset()
    dataset.load_PartsAffordance(args.dataset, subset="val") # TODO
    # Must call before using the dataset
    dataset.prepare()
    config.display()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    '''================= DISPLAY ================='''
    ## =====================
    # Load and display random samples
    # image_ids = np.random.choice(dataset.image_ids, 1)
    # for image_id in image_ids:
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)

        ## =====================
        # Load and display
        image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
            dataset, config, image_id, use_mini_mask=False)
        log("molded_image", image)
        log("mask", mask)
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names,
                                    show_bbox=False)

    '''================= Image Augmentation ====================='''
    # # Image augmentation
    # # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    # augmentation = iaa.Sometimes(0.9, [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    # #     # iaa.Multiply((0.8, 1.2)),
    # #     # iaa.GaussianBlur(sigma=(0.0, 5.0)),
    # #     # iaa.OneOf([iaa.Affine(rotate=90),
    # #     #            iaa.Affine(rotate=180),
    # #     #            iaa.Affine(rotate=270)]),
    # ])
    #
    # image_id = 3
    # # Load the image multiple times to show augmentations
    # limit = 16
    # ax = get_ax(rows=4, cols=4)
    # for i in range(limit):
    #     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    #         dataset, config, image_id, use_mini_mask=False, augment=False, augmentation=augmentation)
    #     visualize.display_instances(image, bbox, mask, class_ids,
    #                                 dataset.class_names, ax=ax[i // 4, i % 4],
    #                                 show_mask=False, show_bbox=False)

    # # ============== CROPPING ==============
    # crop_config = RandomCropConfig()
    # # Load the image multiple times to show augmentations
    # limit = 4
    # image_id = np.random.choice(dataset.image_ids, 1)[0]
    # ax = get_ax(rows=2, cols=limit // 2)
    # for i in range(limit):
    #     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    #         dataset, crop_config, image_id, use_mini_mask=False)
    #     visualize.display_instances(image, bbox, mask, class_ids,
    #                                 dataset.class_names, ax=ax[i // 2, i % 2],
    #                                 show_mask=False, show_bbox=False)

    '''================= image stats ====================='''
    # # # ====================
    # # Loop through the dataset and compute stats over multiple threads
    # # This might take a few minutes
    # t_start = time.time()
    # with concurrent.futures.ThreadPoolExecutor() as e:
    #     stats = list(e.map(image_stats, dataset.image_ids))
    # t_total = time.time() - t_start
    # print("Total time: {:.1f} seconds".format(t_total))
    #
    # # # ====================
    # # # Image stats
    # image_shape = np.array([s['shape'] for s in stats])
    # image_color = np.array([s['color'] for s in stats])
    # print("Image Count: ", image_shape.shape[0])
    # # print("Height  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    # #     np.mean(image_shape[:, 0]), np.median(image_shape[:, 0]),
    # #     np.min(image_shape[:, 0]), np.max(image_shape[:, 0])))
    # # print("Width   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    # #     np.mean(image_shape[:, 1]), np.median(image_shape[:, 1]),
    # #     np.min(image_shape[:, 1]), np.max(image_shape[:, 1])))
    # print("Color   mean (RGB): {:.2f} {:.2f} {:.2f}".format(*np.mean(image_color, axis=0)))

    '''================= Histograms ====================='''
    # # Segment by image area
    # image_area_bins = [480*640]
    #
    # # # ============ ASPECT RATIOS ============
    # fig, ax = plt.subplots(1, len(image_area_bins), figsize=(16, 4))
    # area_threshold = 0
    # for i, image_area in enumerate(image_area_bins):
    #     nucleus_shape = np.array([
    #         b
    #         for s in stats if area_threshold < (s['shape'][0] * s['shape'][1]) <= image_area
    #         for b in s['bbox']])
    #     nucleus_area = nucleus_shape[:, 0] * nucleus_shape[:, 1]
    #     area_threshold = image_area
    #
    #     print("\nImage Area <= {:.0f}**2".format(np.sqrt(image_area)))
    #     print("  Total Nuclei: ", nucleus_shape.shape[0])
    #     print("  Nucleus Height. mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    #         np.mean(nucleus_shape[:, 0]), np.median(nucleus_shape[:, 0]),
    #         np.min(nucleus_shape[:, 0]), np.max(nucleus_shape[:, 0])))
    #     print("  Nucleus Width.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    #         np.mean(nucleus_shape[:, 1]), np.median(nucleus_shape[:, 1]),
    #         np.min(nucleus_shape[:, 1]), np.max(nucleus_shape[:, 1])))
    #     print("  Nucleus Area.   mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    #         np.mean(nucleus_area), np.median(nucleus_area),
    #         np.min(nucleus_area), np.max(nucleus_area)))
    #
    #     # Show 2D histogram
    #     _ = ax.hist2d(nucleus_shape[:, 1], nucleus_shape[:, 0], bins=20, cmap="Blues")
    #
    # # Nuclei height/width ratio
    # nucleus_aspect_ratio = nucleus_shape[:, 0] / nucleus_shape[:, 1]
    # print("Nucleus Aspect Ratio.  mean: {:.2f}  median: {:.2f}  min: {:.2f}  max: {:.2f}".format(
    #     np.mean(nucleus_aspect_ratio), np.median(nucleus_aspect_ratio),
    #     np.min(nucleus_aspect_ratio), np.max(nucleus_aspect_ratio)))
    # plt.figure(figsize=(15, 5))
    # _ = plt.hist(nucleus_aspect_ratio, bins=100, range=[0, 5])

    '''================= Anchors ====================='''
    # image_id = np.random.choice(dataset.image_ids, 1)[0]
    # # for image_id in range(len(dataset.image_ids)):
    # crop_config = RandomCropConfig()
    # ## Visualize anchors of one cell at the center of the feature map
    #
    # # Load and display random image
    # image_id = np.random.choice(dataset.image_ids, 1)[0]
    # image, image_meta, _, _, _ = modellib.load_image_gt(dataset, crop_config, image_id)
    #
    # # Generate Anchors
    # backbone_shapes = modellib.compute_backbone_shapes(config, image.shape)
    # anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
    #                                          config.RPN_ANCHOR_RATIOS,
    #                                          backbone_shapes,
    #                                          config.BACKBONE_STRIDES,
    #                                          config.RPN_ANCHOR_STRIDE)
    #
    # # Print summary of anchors
    # num_levels = len(backbone_shapes)
    # anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    # print("Count: ", anchors.shape[0])
    # print("Scales: ", config.RPN_ANCHOR_SCALES)
    # print("ratios: ", config.RPN_ANCHOR_RATIOS)
    # print("Anchors per Cell: ", anchors_per_cell)
    # print("Levels: ", num_levels)
    # anchors_per_level = []
    # for l in range(num_levels):
    #     num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
    #     anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE ** 2)
    #     print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
    #
    # # Display
    # fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(image)
    #
    # levels = len(backbone_shapes)
    # for level in range(levels):
    #     colors = visualize.random_colors(levels)
    #     # Compute the index of the anchors at the center of the image
    #     level_start = sum(anchors_per_level[:level])  # sum of anchors of previous levels
    #     level_anchors = anchors[level_start:level_start + anchors_per_level[level]]
    #     print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0],
    #                                                                   backbone_shapes[level]))
    #     center_cell = backbone_shapes[level] // 2
    #     center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
    #     level_center = center_cell_index * anchors_per_cell
    #     center_anchor = anchors_per_cell * (
    #             (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE ** 2) \
    #             + center_cell[1] / config.RPN_ANCHOR_STRIDE)
    #     level_center = int(center_anchor)
    #
    #     # Draw anchors. Brightness show the order in the array, dark to bright.
    #     for i, rect in enumerate(level_anchors[level_center:level_center + anchors_per_cell]):
    #         y1, x1, y2, x2 = rect
    #         p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, facecolor='none',
    #                               edgecolor=(i + 1) * np.array(colors[level]) / anchors_per_cell)
    #         ax.add_patch(p)
    # # plt.show()

    '''================= Masks ====================='''
    # # =======================
    # image_id = np.random.choice(dataset.image_ids, 1)[0]
    # for image_id in range(len(dataset.image_ids)):
    #
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     original_shape = image.shape
    #
    #     print("Min: ", config.IMAGE_MIN_DIM)
    #     print("Max: ", config.IMAGE_MAX_DIM)
    #     print("Resize Mode: ", config.IMAGE_RESIZE_MODE)
    #
    #     # =======================
    #     # Resize
    #     image, window, scale, padding, _ = utils.resize_image(
    #         image,
    #         min_dim=config.IMAGE_MIN_DIM,
    #         max_dim=config.IMAGE_MAX_DIM,
    #         mode=config.IMAGE_RESIZE_MODE)
    #     mask = utils.resize_mask(mask, scale, padding)
    #     # Compute Bounding box
    #     bbox = utils.extract_bboxes(mask)
    #
    #     # Display image and additional stats
    #     print("image_id: ", image_id, dataset.image_reference(image_id))
    #     print("Original shape: ", original_shape)
    #     log("image", image)
    #     log("mask", mask)
    #     log("class_ids", class_ids)
    #     log("bbox", bbox)
    #     # Display image and instances
    #     visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
    #
    #     # # =======================
    #     # image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    #     #     dataset, config, image_id, use_mini_mask=False)
    #     #
    #     # log("image", image)
    #     # log("image_meta", image_meta)
    #     # log("class_ids", class_ids)
    #     # log("bbox", bbox)
    #     # log("mask", mask)
    #     #
    #     # # =======================
    #     # display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
    #     #
    #     # # =======================
    #     # # Add augmentation and mask resizing.
    #     # image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    #     #     dataset, config, image_id, augment=True, use_mini_mask=True)
    #     # log("mask", mask)
    #     # display_images([image] + [mask[:, :, i] for i in range(min(mask.shape[-1], 7))])
    #
    # '''================= ROIs ====================='''
    # # for image_id in range(len(dataset.image_ids)):
    # for image_id in range(1):
    #     crop_config = RandomCropConfig()
    #     # ==================
    #     # Create data generator
    #     random_rois = 512
    #     g = modellib.data_generator(
    #         dataset, crop_config, shuffle=True, random_rois=random_rois,
    #         batch_size=2,
    #         detection_targets=True)
    #
    #     # ==================
    #     # Get Next Image
    #     if random_rois:
    #         [normalized_images, image_meta, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, rpn_rois, rois], \
    #         [mrcnn_class_ids, mrcnn_bbox, mrcnn_mask] = next(g)
    #
    #         log("rois", rois)
    #         log("mrcnn_class_ids", mrcnn_class_ids)
    #         log("mrcnn_bbox", mrcnn_bbox)
    #         log("mrcnn_mask", mrcnn_mask)
    #     else:
    #         [normalized_images, image_meta, rpn_match, rpn_bbox, gt_boxes, gt_masks], _ = next(g)
    #
    #     log("gt_class_ids", gt_class_ids)
    #     log("gt_boxes", gt_boxes)
    #     log("gt_masks", gt_masks)
    #     log("rpn_match", rpn_match, )
    #     log("rpn_bbox", rpn_bbox)
    #     image_id = modellib.parse_image_meta(image_meta)["image_id"][0]
    #     print("image_id: ", image_id, dataset.image_reference(image_id))
    #
    #     # Remove the last dim in mrcnn_class_ids. It's only added
    #     # to satisfy Keras restriction on target shape.
    #     mrcnn_class_ids = mrcnn_class_ids[:, :, 0]
    #
    #
    #     # ==================
    #     b = 0
    #
    #     # Restore original image (reverse normalization)
    #     sample_image = modellib.unmold_image(normalized_images[b], config)
    #
    #     # # Compute anchor shifts.
    #     indices = np.where(rpn_match[b] == 1)
    #     #
    #     # print("indices: ", indices)
    #     # print("anchors: ", anchors.shape)
    #     # print("rpn_bbox: ", rpn_bbox.shape)
    #     #
    #     # refined_anchors = utils.apply_box_deltas(anchors[indices], rpn_bbox[b, :len(indices)] * config.RPN_BBOX_STD_DEV)
    #     # log("anchors", anchors)
    #     # log("refined_anchors", refined_anchors)
    #     #
    #     # # Get list of positive anchors
    #     # positive_anchor_ids = np.where(rpn_match[b] == 1)[0]
    #     # print("Positive anchors: {}".format(len(positive_anchor_ids)))
    #     # negative_anchor_ids = np.where(rpn_match[b] == -1)[0]
    #     # print("Negative anchors: {}".format(len(negative_anchor_ids)))
    #     # neutral_anchor_ids = np.where(rpn_match[b] == 0)[0]
    #     # print("Neutral anchors: {}".format(len(neutral_anchor_ids)))
    #
    #     # # ROI breakdown by class
    #     # for c, n in zip(dataset.class_names, np.bincount(mrcnn_class_ids[b].flatten())):
    #     #     if n:
    #     #         print("{:23}: {}".format(c[:20], n))
    #     #
    #     # # Show positive anchors
    #     # fig, ax = plt.subplots(1, figsize=(16, 16))
    #     # visualize.draw_boxes(sample_image, boxes=anchors[positive_anchor_ids],
    #     #                      refined_boxes=refined_anchors, ax=ax)
    #
    #     # if random_rois:
    #     #     # Class aware bboxes
    #     #     bbox_specific = mrcnn_bbox[b, np.arange(mrcnn_bbox.shape[1]), mrcnn_class_ids[b], :]
    #     #
    #     #     # Refined ROIs
    #     #     refined_rois = utils.apply_box_deltas(rois[b].astype(np.float32), bbox_specific[:,:4] * config.BBOX_STD_DEV)
    #     #
    #     #     # Class aware masks
    #     #     mask_specific = mrcnn_mask[b, np.arange(mrcnn_mask.shape[1]), :, :, mrcnn_class_ids[b]]
    #     #
    #     #     visualize.draw_rois(sample_image, rois[b], refined_rois, mask_specific, mrcnn_class_ids[b], dataset.class_names)
    #     #
    #     #     # Any repeated ROIs?
    #     #     rows = np.ascontiguousarray(rois[b]).view(np.dtype((np.void, rois.dtype.itemsize * rois.shape[-1])))
    #     #     _, idx = np.unique(rows, return_index=True)
    #     #     print("Unique ROIs: {} out of {}".format(len(idx), rois.shape[1]))
    #     #
    #     # if random_rois:
    #     #     # Dispalay ROIs and corresponding masks and bounding boxes
    #     #     ids = random.sample(range(rois.shape[1]), 8)
    #     #
    #     #     images = []
    #     #     titles = []
    #     #     for i in ids:
    #     #         image = visualize.draw_box(sample_image.copy(), rois[b, i, :4].astype(np.int32), [255, 0, 0])
    #     #         image = visualize.draw_box(image, refined_rois[i].astype(np.int64), [0, 255, 0])
    #     #         images.append(image)
    #     #         titles.append("ROI {}".format(i))
    #     #         images.append(mask_specific[i] * 255)
    #     #         titles.append(dataset.class_names[mrcnn_class_ids[b, i]][:20])
    #     #
    #     #     display_images(images, titles, cols=4, cmap="Blues", interpolation="none")
    #
    #     if random_rois:
    #         limit = 10
    #         temp_g = modellib.data_generator(
    #             dataset, crop_config, shuffle=True, random_rois=10000,
    #             batch_size=1, detection_targets=True)
    #         total = 0
    #         for i in range(limit):
    #             _, [ids, _, _] = next(temp_g)
    #             positive_rois = np.sum(ids[0] > 0)
    #             total += positive_rois
    #             print("{:5} {:5.2f}".format(positive_rois, positive_rois / ids.shape[1]))
    #         print("Average percent: {:.2f}".format(total / (limit * ids.shape[1])))
    #
    #     # plt.show()

    plt.show()

