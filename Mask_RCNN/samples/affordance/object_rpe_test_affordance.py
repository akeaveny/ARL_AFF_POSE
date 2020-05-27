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

# Path to trained weights file
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

###########################################################
# Test
###########################################################

def seq_get_masks(image, pre_detection, cur_detection):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']

    pre_masks = pre_detection['masks']
    pre_class_ids = pre_detection['class_ids']
    pre_rois = pre_detection['rois']

    new_masks = pre_detection['masks']
    new_class_ids = pre_detection['class_ids']
    new_rois = pre_detection['rois']

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    good_detection = True
    print('cur_masks.shape: {}'.format(cur_masks.shape[-1]))

    if cur_masks.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(cur_masks.shape[-1]):
            sub = np.abs(pre_rois-cur_rois[i])
            dist = sum(sub.transpose())
            print('cur_rois[i]: {}'.format(cur_rois[i]))
            mask_index = dist.argmin()
            if dist.min() < 50:
                if new_class_ids[mask_index] != cur_class_ids[i]: # bad classification
                    good_detection = False
                    pass
                else:
                    new_rois[mask_index,:] = cur_rois[i,:] # change order of current masks to follow the mask order of previous prediction
            else:
                good_detection = False
                pass

            semantic_mask = semantic_mask_one * cur_class_ids[i]
            semantic_masks = np.where(cur_masks[:, :, i], semantic_mask, semantic_masks).astype(np.uint8)

            instance_mask = instance_mask_one * (mask_index+1)
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    #print('old rois: \n {}'.format(cur_rois))
    #print('new rois: \n {}'.format(new_rois))

    instance_to_color = Affordance.color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    if good_detection:
        pre_detection['masks'] = new_masks
        pre_detection['class_ids'] = new_class_ids
        pre_detection['rois'] = new_rois
    return semantic_masks, instance_masks, color_masks, pre_detection, good_detection

def detect_and_get_masks(model, data_path, num_frames):

    classes_file_dir = '/data/Akeaveny/Datasets/part-affordance-dataset/classes.txt'
    class_id_file_dir = '/data/Akeaveny/Datasets/part-affordance-dataset/class_ids.txt'
    offset = 0
    skip = 10

    for i in range(0, num_frames):
        assign_first_pre_detect = True

        ## ============== clutter ===================
        # folder_to_load = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/combined_val_375k/'
        folder_to_save = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/masks/'
        count = 100000 + (i*skip + offset)
        str_num = str(count)[1:]
        rgb_addr = data_path + str_num + '_rgb.png'
        depth_addr = data_path + str_num + '_depth.png'
        print(rgb_addr, depth_addr)

        if os.path.isfile(rgb_addr) == False: 
            continue;
        if os.path.isfile(depth_addr) == False: 
            continue;

        # Read image
        image = skimage.io.imread(rgb_addr)
        depth = skimage.io.imread(depth_addr)

        # ## ============== SYNTHETIC ===================
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Detect objects
        cur_detect = model.detect([image], verbose=0)[0]
        if assign_first_pre_detect and cur_detect['masks'].shape[-1] > 0:
            assign_first_pre_detect = False
            pre_detect = cur_detect
            with open(class_id_file_dir, 'w') as the_file:
                for j in range(cur_detect['class_ids'].shape[0]):
                    detected_class = str(cur_detect['class_ids'][j])
                    the_file.write(str(detected_class))
                    the_file.write('\n')

        # get instance_masks
        if not assign_first_pre_detect:
            semantic_masks, instance_masks, color_masks, pre_detect, good_detect = seq_get_masks(image, pre_detect, cur_detect)

            if good_detect:
                mask_addr = folder_to_save + str_num + '.mask-color.png'
                skimage.io.imsave(mask_addr, color_masks)
                mask_addr = folder_to_save + str_num + '.mask.png'
                skimage.io.imsave(mask_addr, instance_masks)

                # ========== tutorials ================
                results = model.detect([image], verbose=1)
                r = results[0]
                class_names = np.loadtxt(classes_file_dir, dtype=np.str)
                class_names = np.array(class_names)
                class_ids = r['class_ids'] - 1

                print("-------------------------")
                print("class_names: ", class_names)

                # ========== detect model ============
                results = model.detect([image], verbose=1)
                r = results[0]
                visualize.display_instances(image, r['rois'], r['masks'], class_ids,
                                            class_names, r['scores'], figsize=(10, 10), ax=None,
                                            title="Predictions", show_bbox=True, show_mask=True)

        # # ========== plot ============
        # plt.subplot(2, 2, 1)
        # plt.title('rgb')
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.title('depth')
        # plt.imshow(depth)
        # plt.subplot(2, 2, 3)
        # plt.title('mask')
        # plt.imshow(instance_masks)
        # plt.subplot(2, 2, 4)
        # plt.title('label')
        # plt.imshow(semantic_masks)
        # plt.ioff()
        # plt.pause(0.001)
        # plt.show()

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