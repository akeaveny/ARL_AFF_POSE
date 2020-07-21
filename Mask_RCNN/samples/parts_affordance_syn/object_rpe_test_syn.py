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

# ========= load dataset =========
import matterport_dataset_syn as Affordance
import scipy.io as scio
# ========== GPU config ================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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
            # print('mask_index: {}'.format(mask_index))
            if dist.min() < 50:
                if new_class_ids[mask_index] != cur_class_ids[i]: # bad classification
                    good_detection = False
                    pass
                elif new_class_ids[mask_index] == 7:
                    pass
                else:
                    new_rois[mask_index,:] = cur_rois[i,:] # change order of current masks to follow the mask order of previous prediction
            else:
                good_detection = False
                pass

            # print("cur_class_ids[i] :", cur_class_ids[i])
            semantic_mask = semantic_mask_one * map_affordance_label(cur_class_ids[i])
            semantic_masks = np.where(cur_masks[:, :, i], semantic_mask, semantic_masks).astype(np.uint8)

            # instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * map_affordance_label(cur_class_ids[i])
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

    classes_file_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/object_rpe_classes_.txt'
    class_id_file_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/object_rpe_class_ids_.txt'

    gt_label_addr = data_path + '??????' + '_label.png'
    files = sorted(glob.glob(gt_label_addr))
    print("Loaded files: ", len(files))

    for i, file in enumerate(files[0:num_frames]):
        print("Iteration: {}/{}".format(i, num_frames))

        str_num = file.split(data_path)[1]
        str_num = str_num.split('_label.png')[0]

        ## ============== syn ===================
        folder_to_save = data_path
        # str_num = str(count)[1:]
        rgb_addr = data_path + str_num + '_rgb.jpg'
        depth_addr = data_path + str_num + '_depth.png'
        gt_label_addr = data_path + str_num + '_label.png'
        print("Image: ", str_num)

        if os.path.isfile(rgb_addr) == False:
            continue;
        if os.path.isfile(depth_addr) == False:
            continue;

        # Read image
        image = skimage.io.imread(rgb_addr)
        # depth = skimage.io.imread(depth_addr)
        gt_label = skimage.io.imread(gt_label_addr)

        # ## ============== SYNTHETIC ===================
        if image.shape[-1] == 4:
            image = image[..., :3]

        # ## ============== OBJECT ID ===================
        # meta_addr = data_path + str_num + "-meta.mat"
        # meta = scio.loadmat(meta_addr)
        # object_id = meta['Object_ID'].flatten().astype(np.int32)[0]

        # Detect objects
        cur_detect = model.detect([image], verbose=0)[0]
        # if cur_detect['masks'].shape[-1] > 0:
        #     pre_detect = cur_detect
        #     with open(class_id_file_dir, 'w') as the_file:
        #         # print("object_id: ", object_id)
        #         # the_file.write(str(object_id))
        #         # the_file.write('\n')
        #
        #         for idx, object_id in enumerate(cur_detect['class_ids']):
        #             affordance_label = map_affordance_label(object_id)
        #             # if affordance_label != 2 and affordance_label != 3 \
        #             #         and affordance_label != 4 and affordance_label != 5 \
        #             #             and affordance_label != 6:
        #             print("object idx: ", str(object_id))
        #             print("affordance_label: ", str(affordance_label))
        #             the_file.write(str(object_id))
        #             the_file.write('\n')
        #
        # # get instance_masks
        # semantic_masks, instance_masks, color_masks, pre_detect, good_detect = seq_get_masks(image, pre_detect, cur_detect)
        #
        # # if good_detect:
        # mask_addr = folder_to_save + str_num + '.mask-color.png'
        # skimage.io.imsave(mask_addr, color_masks)
        # mask_addr = folder_to_save + str_num + '.mask.png'
        # skimage.io.imsave(mask_addr, instance_masks)

        ### ========== tutorials ================
        results = model.detect([image], verbose=1)
        r = results[0]
        class_names = np.loadtxt(classes_file_dir, dtype=np.str)
        class_names = np.array(class_names)
        class_ids = r['class_ids'] - 1

        ### ========== detect model ============
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], class_ids,
                                    class_names, r['scores'], figsize=(10, 10), ax=None,
                                    title="Predictions", show_bbox=True, show_mask=True)

        # print("gt", np.unique(gt_label))
        # print("pred" , np.unique(instance_masks))
        # # # ========== plot ============
        # plt.subplot(3, 2, 1)
        # plt.title('rgb')
        # plt.imshow(image)
        # # plt.subplot(3, 2, 2)
        # # plt.title('depth')
        # # plt.imshow(depth)
        # plt.subplot(3, 2, 3)
        # plt.title('gt label')
        # plt.imshow(gt_label)
        # plt.subplot(3, 2, 5)
        # plt.title('MaskRCNN Color')
        # plt.imshow(color_masks)
        # plt.subplot(3, 2, 6)
        # plt.title('MaskRCNN Mask')
        # plt.imshow(instance_masks)
        # plt.ioff()
        # plt.pause(0.001)
        # plt.show()

def map_affordance_label(current_id):

    # 1
    grasp = [
        20, # 'hammer-grasp'
        22,
        24,
        26,
        28, # 'knife-grasp'
        30,
        32,
        34,
        36,
        38,
        40,
        42,
        44,
        46,
        48,
        50,
        52, # 'ladle-grasp'
        54,
        56,
        58, # 'mallet-grasp'
        60,
        62,
        64,
        66, # 'mug-grasp'
        69,
        72,
        75,
        78,
        81,
        84,
        87,
        90,
        93,
        96,
        99,
        102,
        105,
        108,
        111,
        114,
        117,
        120,
        123,
        130, # 'saw-grasp'
        132,
        134,
        136, # 'scissors-grasp'
        138,
        140,
        142,
        144,
        146,
        148,
        150,
        152, # 'scoop-grasp'
        154,
        156, # 'shears-grasp'
        158,
        160, # 'shovel-grasp'
        162,
        164, # 'spoon-grasp'
        166,
        168,
        170,
        172,
        174,
        176,
        178,
        180,
        182,
        184, # 'tenderizer-grasp'
        186, # 'trowel-grasp'
        188,
        190,
        192, # 'turner-grasp'
        194,
        196,
        198,
        200,
        202,
        204,
    ]

    # 2
    cut = [
        28+1, # "knife-cut"
        30+1,
        32+1,
        34+1,
        36+1,
        38+1,
        40+1,
        42+1,
        44+1,
        46+1,
        48+1,
        50+1,
        130+1, # "saw-cut"
        132+1,
        134+1,
        136 + 1, # "scissors-cut"
        138 + 1,
        140 + 1,
        142 + 1,
        144 + 1,
        146 + 1,
        148 + 1,
        150 + 1,
        156 + 1, # "shears-cut"
        158 + 1,
    ]

    # 3
    scoop = [
        152 + 1, # "scoop-scoop"
        154 + 1,
        160 + 1, # "shovel-scoop"
        162 + 1,
        164 + 1, # "spoon-scoop"
        166 + 1,
        168 + 1,
        170 + 1,
        172 + 1,
        174 + 1,
        176 + 1,
        178 + 1,
        180 + 1,
        182 + 1,
        186 + 1, # "trowel-scoop"
        188 + 1,
        190 + 1,
    ]

    # 4
    contain = [
        1, #"bowl-contain"
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        10, # "cup-contain"
        12,
        14,
        16,
        18,
        52 + 1, # "ladle-contain"
        54 + 1,
        56 + 1,
        66 + 1, # "mug-contain"
        69 + 1,
        72 + 1,
        75 + 1,
        78 + 1,
        81 + 1,
        84 + 1,
        87 + 1,
        90 + 1,
        93 + 1,
        96 + 1,
        99 + 1,
        102 + 1,
        105 + 1,
        108 + 1,
        111 + 1,
        114 + 1,
        117 + 1,
        120 + 1,
        123 + 1,
        126, # "pot-contain"
        128,
    ]

    # 5
    pound = [
        20 + 1, #"hammer-pound"
        22 + 1, #"hammer-pound"
        24 + 1, #"hammer-pound"
        26 + 1, #"hammer-pound"
        58 + 1, #'mallet-pound'
        60 + 1, #'mallet-pound'
        62 + 1, #'mallet-pound'
        64 + 1, #'mallet-pound'
        184 + 1, #'tenderizer-pound'
    ]

    # 6
    support = [
        192 + 1, # "turner-support"
        194 + 1,
        196 + 1,
        198 + 1,
        200 + 1,
        202 + 1,
        204 + 1,
    ]

    # 7
    wrap_grasp = [
        8 + 1, # "cup-wrap_grasp"
        10 + 1, # "cup-wrap_grasp"
        12 + 1, # "cup-wrap_grasp"
        14 + 1, # "cup-wrap_grasp"
        16 + 1, # "cup-wrap_grasp"
        18 + 1, # "cup-wrap_grasp"
        66 + 2, # "mug-wrap_grasp"
        69 + 2, # "mug-wrap_grasp"
        72 + 2, # "mug-wrap_grasp"
        75 + 2, # "mug-wrap_grasp"
        78 + 2, # "mug-wrap_grasp"
        81 + 2, # "mug-wrap_grasp"
        84 + 2, # "mug-wrap_grasp"
        87 + 2, # "mug-wrap_grasp"
        90 + 2, # "mug-wrap_grasp"
        93 + 2, # "mug-wrap_grasp"
        96 + 2, # "mug-wrap_grasp"
        99 + 2, # "mug-wrap_grasp"
        102 + 2, # "mug-wrap_grasp"
        105 + 2, # "mug-wrap_grasp"
        108 + 2, # "mug-wrap_grasp"
        111 + 2, # "mug-wrap_grasp"
        114 + 2, # "mug-wrap_grasp"
        117 + 2, # "mug-wrap_grasp"
        120 + 2, # "mug-wrap_grasp"
        123 + 2, # "mug-wrap_grasp"
        126 + 1, # "pot-wrap_grasp"
        128 + 1, # "pot-wrap_grasp"
    ]

    if current_id in grasp:
        return 1
    elif current_id in cut:
        return 2
    elif current_id in scoop:
        return 3
    elif current_id in contain:
        return 4
    elif current_id in pound:
        return 5
    elif current_id in support:
        return 6
    elif current_id in wrap_grasp:
        return 7
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        print(current_id)
        exit(1)

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