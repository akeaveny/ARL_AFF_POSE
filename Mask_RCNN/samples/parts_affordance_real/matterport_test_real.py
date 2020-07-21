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
import matterport_dataset_real as Affordance
# ========== GPU config ================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    ARs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)

        # # ## ============== SYNTHETIC ===================
        # if image.shape[-1] == 4:
        #     image = image[..., :3]

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

    # # # ========== batch mAP ============
    # Run on validation set
    limit = len(dataset.image_ids)
    # limit = 29 # real images
    APs = compute_batch_ap(dataset, dataset.image_ids[:limit])
    print("Mean AP over {} SYN images: {:.4f}".format(len(APs), np.mean(APs)))

    for idx in range(len(dataset.image_ids)):
        # print("\n", dataset.image_ids)
        # image_id = random.choice(dataset.image_ids)
        # image_id = 9
        image_id = idx

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        ## ============== SYNTHETIC ===================
        if image.shape[-1] == 4:
            image = image[..., :3]

        info = dataset.image_info[image_id]
        print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                               dataset.image_reference(image_id)))
        print("Original image shape: ",
              modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

        ''' ==================== DETECT ==================== '''
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

        # ''' ==================== VIS ==================== '''
        # IOU
        visualize.display_differences(
            image,
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            dataset.class_names, ax=get_ax(),
            show_box=False, show_mask=False,
            iou_threshold=0.5, score_threshold=0.5)

        # # Ground Truth
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id,
                                    dataset.class_names, ax=get_ax(1),
                                    show_bbox=False, show_mask=False,
                                    title="Ground Truth")

        # # OBJECT DETECTION
        # ax = get_ax(1)
        # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             dataset.class_names, r['scores'], ax=ax,
        #                             title="Predictions")

        # # ========== precision-recall ============
        # AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
        #                                                      r['rois'], r['class_ids'], r['scores'], r['masks'])
        #
        # visualize.plot_precision_recall(AP, precisions, recalls)
        # visualize.plot_overlaps(gt_class_id, r['class_ids'], r['scores'],
        #                         overlaps, dataset.class_names)
        plt.show()

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
    parser.add_argument('--num_frames', type=int, default=1, help='number of images')
    
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