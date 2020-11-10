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

UMD_DEPTH_MAX = 3626



import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')

parser.add_argument('--detect', required=False, default='rgbd+',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False,
                    default='/data/Akeaveny/Datasets/part-affordance_combined/real/',
                    # default='/data/Akeaveny/Datasets/part-affordance_combined/ndds4/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")

parser.add_argument('--dataset_type', required=False, default='real',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')

parser.add_argument('--is_umd_real', required=False,
                    default=True,
                    # default=False,
                    type=bool,
                    metavar="/path/to/YCB/dataset/")

parser.add_argument('--save_inference_images', required=False,
                    default='/data/Akeaveny/Datasets/part-affordance_combined/real/test_maskrcnn_real/',
                    # default='/data/Akeaveny/Datasets/part-affordance_combined/ndds4/test_maskrcnn_syn/',
                    type=str,
                    metavar="/path/to/YCB/dataset/")

parser.add_argument('--num_frames', required=False, default=100,
                    type=int,
                    metavar='test or val')

parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/ or Logs and checkpoints directory (default=logs/)")

parser.add_argument('--show_plots', required=False, default=False,
                    type=bool,
                    metavar='show plots from matplotlib')
parser.add_argument('--save_output', required=False, default=False,
                    type=bool,
                    metavar='save terminal output to text file')

args = parser.parse_args()


######################################
# import dataset_syn1 as Affordance
# save_to_folder = '/images/test_images_syn1/'
# MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
# RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
# ### config ###
# MAX_GT_INSTANCES_ = 2
# DETECTION_MAX_INSTANCES_ = 2
# DETECTION_MIN_CONFIDENCE_ = 0.985
# POST_NMS_ROIS_INFERENCE_ = 100
# RPN_NMS_THRESHOLD_ = 0.8
# DETECTION_NMS_THRESHOLD_ = 0.5
# ### crop ###
# CROP = True
# IMAGE_RESIZE_MODE_ = "crop"
# IMAGE_MIN_DIM_ = 384
# IMAGE_MAX_DIM_ = 384
# ### sqaure ###
# # CROP = False
# # IMAGE_RESIZE_MODE_ = "square"
# # IMAGE_MIN_DIM_ = 640
# # IMAGE_MAX_DIM_ = 640
######################################

############################################################
#  REAL OR SYN
############################################################
# assert args.dataset_type == 'real' or args.dataset_type == 'syn' or args.dataset_type == 'syn1' or args.dataset_type == 'hammer' or args.dataset_type == 'hammer1'
if args.dataset_type == 'real':
    import dataset_real as Affordance
    save_to_folder = '/images/test_images_real/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5 # 0.975
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    # CROP = True
    # IMAGE_RESIZE_MODE_ = "crop"
    # IMAGE_MIN_DIM_ = 384
    # IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    CROP = False
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn':
    import dataset_syn as Affordance
    save_to_folder = '/images/test_images_syn/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 20
    DETECTION_MAX_INSTANCES_ = 20
    DETECTION_MIN_CONFIDENCE_ = 0.975 # 0.975
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    # CROP = True
    # IMAGE_RESIZE_MODE_ = "crop"
    # IMAGE_MIN_DIM_ = 384
    # IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    CROP = False
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'syn1':
    import dataset_syn1 as Affordance
    save_to_folder = '/images/test_images_syn1/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 20 # 2
    DETECTION_MAX_INSTANCES_ = 20 # 2
    DETECTION_MIN_CONFIDENCE_ = 0.975
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    # CROP = True
    # IMAGE_RESIZE_MODE_ = "crop"
    # IMAGE_MIN_DIM_ = 384
    # IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    CROP = False
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'hammer':
    import objects.dataset_syn_hammer as Affordance
    save_to_folder = '/images/objects/test_images_syn_hammer/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'hammer1':
    import objects.dataset_syn_hammer1 as Affordance
    save_to_folder = '/images/objects/test_images_syn_hammer1/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 896
    # IMAGE_MAX_DIM_ = 896
elif args.dataset_type == 'scissors_real':
    import objects.dataset_real_scissors as Affordance
    save_to_folder = '/images/objects/test_images_real_scissors/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'scissors':
    import objects.dataset_syn_scissors as Affordance
    save_to_folder = '/images/objects/test_images_syn_scissors/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
elif args.dataset_type == 'scissors_20k':
    import objects.dataset_syn_scissors_20k as Affordance
    save_to_folder = '/images/objects/test_images_syn_scissors_20k/'
    MEAN_PIXEL_ = np.array([91.13, 88.92, 98.65])  ### REAL RGB
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.5
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### crop ###
    CROP = True
    IMAGE_RESIZE_MODE_ = "crop"
    IMAGE_MIN_DIM_ = 384
    IMAGE_MAX_DIM_ = 384
    ### sqaure ###
    # CROP = False
    # IMAGE_RESIZE_MODE_ = "square"
    # IMAGE_MIN_DIM_ = 640
    # IMAGE_MAX_DIM_ = 640
else:
    print("*** No Dataset Type Selected ***")
    exit(1)

if not (os.path.exists(os.getcwd()+save_to_folder)):
    os.makedirs(os.getcwd()+save_to_folder)

from mrcnn.config import Config
from mrcnn.model import log
from mrcnn.visualize import display_images
import tensorflow as tf

if args.detect == 'rgb':
    from mrcnn import model as modellib, utils, visualize
if args.detect == 'rgbd':
    from mrcnn import modeldepth as modellib, utils, visualize
elif args.detect == 'rgbd+':
    from mrcnn import modeldepthv2 as modellib, utils, visualize
else:
    print("*** No Model Selected ***")
    exit(1)

###########################################################
# Test
###########################################################

def seq_get_masks(image, cur_detection, gt_mask, args):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']
    good_detect = False

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    print("\tObject_ids", cur_class_ids)
    if cur_masks.shape[-1] > 0:

        for i in range(cur_masks.shape[-1]):
            good_detect = True

            if args.dataset_type == 'real' or args.dataset_type == 'syn1':
                cur_class_ids[i] = cur_class_ids[i]
            # elif args.dataset_type == 'syn' or args.dataset_type == 'hammer' or args.dataset_type == 'hammer1' or args.dataset_type == 'scissors':
            #     cur_class_ids[i] = map_affordance_label(cur_class_ids[i])
            print("\tPred Affordance Label:", cur_class_ids[i])

            ### instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * cur_class_ids[i]
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    ########################
    #  add color to masks
    ########################
    instance_to_color = Affordance.color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    return instance_masks, color_masks, good_detect

def detect_and_get_masks(model, config, args):

    ########################
    #  Load test images
    ########################

    print("args.dataset_split", args.dataset_split)
    dataset = Affordance.AffordanceDataset()
    dataset.load_Affordance(args.dataset, args.dataset_split)
    dataset.prepare()

    config.display()

    print("Num of Test Images: {}".format(len(dataset.image_ids)))

    # select random test images
    np.random.seed(0)
    test_idx = np.random.choice(np.arange(0, len(dataset.image_ids), 1), size=int(args.num_frames), replace=False)
    print("Chosen Files \n", len(test_idx))

    for num_image, idx in enumerate(test_idx):

        print("Running Inference on Image {} ..".format(num_image))

        ##############################
        #  Address for saving mask
        ##############################
        # for image_id in range(len(dataset.image_ids)):
        # print("\nimage_file:", dataset.image_reference(image_id))

        image_file1 = dataset.image_reference(idx)
        image_file2 = image_file1.split(args.dataset)[1]  # remove dataset path
        idx = image_file2.split('_rgb')[0]  # remove _rgb label

        ##############################
        ##############################
        if args.is_umd_real:
            rgb_addr = args.dataset + idx + '_rgb.jpg'
            depth_addr = args.dataset + idx + '_depth.png'
            gt_mask_addr = args.dataset + idx + '_label.png' # '_gt_affordance.png' or '_label.png'
        else:
            rgb_addr = args.dataset + idx + '_rgb.png'
            depth_addr = args.dataset + idx + '_depth.png'
            gt_mask_addr = args.dataset + idx + '_gt_affordance.png'  # '_gt_affordance.png' or '_label.png'
        # print("gt_mask_addr: ", gt_mask_addr)

        if os.path.isfile(rgb_addr) == False:
            continue
        if os.path.isfile(depth_addr) == False:
            continue
        if os.path.isfile(gt_mask_addr) == False:
            continue

        mask_addr = args.save_inference_images + str(num_image) + '_mask_og.png'
        color_mask_addr = args.save_inference_images + str(num_image) + '_mask_color.png'
        cropped_mask_addr = args.save_inference_images + str(num_image) + '_mask_cropped.png'
        print("\tmask_addr:", mask_addr)

        ##############################
        ### ground truth
        ##############################

        rgb = np.array(skimage.io.imread(rgb_addr))
        depth = np.array(skimage.io.imread(depth_addr))
        gt_label = np.array(skimage.io.imread(gt_mask_addr))
        ### print("GT RGB SHAPE: ", rgb.shape)
        print("\tGT Affordance Label:", np.unique(gt_label))

        ######################
        # configure depth
        ######################

        depth[np.isnan(depth)] = 0
        depth[depth == -np.inf] = 0
        depth[depth == np.inf] = 0

        # convert to 8-bit image
        # depth = depth * (2 ** 16 -1) / UMD_DEPTH_MAX  ### 16 bit
        # depth = depth * (2 ** 8 - 1) / UMD_DEPTH_MAX  ### 8 bit
        # depth = np.array(depth, dtype=np.uint16)

        # print("depth min: ", np.min(np.array(depth)))
        # print("depth max: ", np.max(np.array(depth)))
        #
        # print("depth type: ", depth.dtype)
        # print("depth shape: ", depth.shape)

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        rgb, depth = rgb[..., :3], skimage.color.gray2rgb(depth)

        ##############################
        #  Resize
        ##############################
        ### rgb = cv2.resize(rgb, dsize=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM), interpolation=cv2.INTER_CUBIC)
        ### depth = cv2.resize(depth, dsize=(config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM), interpolation=cv2.INTER_CUBIC)
        ### gt_label = cv2.resize(gt_label, dsize=(1280, 1280), interpolation=cv2.INTER_CUBIC)

        ##############################
        #  CROP
        ##############################

        if CROP == True:
            # Pick a random crop
            h, w = rgb.shape[:2]

            x = (w - config.IMAGE_MIN_DIM) // 2
            y = (h - config.IMAGE_MIN_DIM) // 2

            rgb = rgb[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            depth = depth[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            gt_label = gt_label[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]

            if gt_label.shape != (config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM):
                print("\tGT shape:", gt_label.shape)
                print("\tgt label addr: ", cropped_mask_addr)
                exit()
        # print("gt_label: ", gt_label.shape)

        # plt.subplot(1, 2, 1)
        # plt.title("rgb")
        # plt.imshow(rgb_og)
        # plt.subplot(1, 2, 2)
        # plt.title("rgb1")
        # plt.imshow(rgb)
        # plt.show()
        # plt.ioff()


        ##############################
        #  Detect
        ##############################

        if args.detect == 'rgb':
            ### load
            ### image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            ###     modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
            # run detect
            cur_detect = model.detect([rgb], verbose=0)[0]

        elif args.detect == 'rgbd' or args.detect == 'rgbd+':
            ### load
            ### image, depthimage, image_meta, gt_class_id, gt_bbox, gt_mask = \
            ###    modellib.load_images_gt(dataset, config, image_id, use_mini_mask=False)
            # run detect
            cur_detect = model.detectWdepth([rgb], [depth], verbose=0)[0]

        # get instance_masks
        instance_mask, color_mask, good_detect = seq_get_masks(rgb, cur_detect, gt_label, args)

        ##############################
        #  Resize
        ##############################
        ### resized_instance_mask = cv2.resize(instance_mask, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        ### resized_color_mask = cv2.resize(color_mask, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
        ### cv2.imwrite(mask_addr, resized_instance_mask)  # TODO: instance_masks is better F_wb
        ### cv2.imwrite(color_mask_addr, resized_color_mask)

        ####################

        cv2.imwrite(mask_addr, instance_mask)
        cv2.imwrite(color_mask_addr, color_mask)
        cv2.imwrite(cropped_mask_addr, gt_label )

        if args.show_plots:  # TODO: boolean string
            print("\tGT shape:", gt_label.shape)
            print("\tPred shape:", instance_mask.shape)
            print("\tGT Label:", np.unique(gt_label))
            print("\tPred Labels:", np.unique(instance_mask))

            cv2.imshow("gt", gt_label * 25)
            cv2.imshow("resize pred", instance_mask * 25)
            cv2.waitKey(0)

###########################################################
# FOR SYTHENTIC IMAGES
# LOOKUP FROM OBJECT ID TO AFFORDANCE LABEL
###########################################################
def map_affordance_label(current_id):

    # 1
    grasp = [
        20,  # "hammer-grasp"
        22,  # "hammer-grasp"
        24,  # "hammer-grasp"
        26,  # "hammer-grasp"
        #
        28,  # "knife-grasp"
        30,  # "knife-grasp"
        32,  # "knife-grasp"
        34,  # "knife-grasp"
        36,  # "knife-grasp"
        38,  # "knife-grasp"
        40,  # "knife-grasp"
        42,  # "knife-grasp"
        44,  # "knife-grasp"
        46,  # "knife-grasp"
        48,  # "knife-grasp"
        50,  # "knife-grasp"
        #
        52,  # "ladle-grasp"
        54,  # "ladle-grasp"
        56,  # "ladle-grasp"
        #
        58,  # "mallet-grasp"
        60,  # "mallet-grasp"
        62,  # "mallet-grasp"
        64,  # "mallet-grasp"
        #
        66,  # "mug-grasp"
        69,  # "mug-grasp"
        72,  # "mug-grasp"
        75,  # "mug-grasp"
        78,  # "mug-grasp"
        81,  # "mug-grasp"
        84,  # "mug-grasp"
        87,  # "mug-grasp"
        90,  # "mug-grasp"
        93,  # "mug-grasp"
        96,  # "mug-grasp"
        99,  # "mug-grasp"
        102,  # "mug-grasp"
        105,  # "mug-grasp"
        108,  # "mug-grasp"
        111,  # "mug-grasp"
        114,  # "mug-grasp"
        117,  # "mug-grasp"
        120,  # "mug-grasp"
        123,  # "mug-grasp"
        #
        130,  # "saw-grasp"
        132,  # "saw-grasp"
        134,  # "saw-grasp"
        #
        136,  # "scissors-grasp"
        138,  # "scissors-grasp"
        140,  # "scissors-grasp"
        142,  # "scissors-grasp"
        144,  # "scissors-grasp"
        146,  # "scissors-grasp"
        148,  # "scissors-grasp"
        150,  # "scissors-grasp"
        #
        152,  # "scoop-grasp"
        154,  # "scoop-grasp"
        #
        156,  # "shears-grasp"
        158,  # "shears-grasp"
        #
        160,  # "shovel-grasp"
        162,  # "shovel-grasp"
        #
        164,  # "spoon-grasp"
        166,  # "spoon-grasp"
        168,  # "spoon-grasp"
        170,  # "spoon-grasp"
        172,  # "spoon-grasp"
        174,  # "spoon-grasp"
        176,  # "spoon-grasp"
        178,  # "spoon-grasp"
        180,  # "spoon-grasp"
        182,  # "spoon-grasp"
        #
        184,  # "tenderizer-grasp"
        #
        186,  # "trowel-grasp"
        188,  # "trowel-grasp"
        190,  # "trowel-grasp"
        #
        192,  # "turner-grasp"
        194,  # "turner-grasp"
        196,  # "turner-grasp"
        198,  # "turner-grasp"
        200,  # "turner-grasp"
        202,  # "turner-grasp"
        204,  # "turner-grasp"
    ]

    # 2
    cut = [
        28 + 1,  # "knife-cut"
        30 + 1,  # "knife-cut"
        32 + 1,  # "knife-cut"
        34 + 1,  # "knife-cut"
        36 + 1,  # "knife-cut"
        38 + 1,  # "knife-cut"
        40 + 1,  # "knife-cut"
        42 + 1,  # "knife-cut"
        44 + 1,  # "knife-cut"
        46 + 1,  # "knife-cut"
        48 + 1,  # "knife-cut"
        50 + 1,  # "knife-cut"
        #
        130 + 1,  # "saw-cut"
        132 + 1,  # "saw-cut"
        134 + 1,  # "saw-cut"
        #
        136 + 1,  # "scissors-cut"
        138 + 1,  # "scissors-cut"
        140 + 1,  # "scissors-cut"
        142 + 1,  # "scissors-cut"
        144 + 1,  # "scissors-cut"
        146 + 1,  # "scissors-cut"
        148 + 1,  # "scissors-cut"
        150 + 1,  # "scissors-cut"
        #
        156 + 1, # "shears-cut"
        158 + 1,
    ]

    # 3
    scoop = [
        152 + 1,  # "scoop-scoop"
        154 + 1,  # "scoop-scoop"
        #
        160 + 1,  # "shovel-scoop"
        162 + 1,  # "shovel-scoop"
        #
        164 + 1,  # "spoon-scoop"
        166 + 1,  # "spoon-scoop"
        168 + 1,  # "spoon-scoop"
        170 + 1,  # "spoon-scoop"
        172 + 1,  # "spoon-scoop"
        174 + 1,  # "spoon-scoop"
        176 + 1,  # "spoon-scoop"
        178 + 1,  # "spoon-scoop"
        180 + 1,  # "spoon-scoop"
        182 + 1,  # "spoon-scoop"
        #
        186 + 1,  # "trowel-scoop"
        188 + 1,  # "trowel-scoop"
        190 + 1,  # "trowel-scoop"
    ]

    # 4
    contain = [
        1,  # "bowl-contain"
        2,  # "bowl-contain"
        3,  # "bowl-contain"
        4,  # "bowl-contain"
        5,  # "bowl-contain"
        6,  # "bowl-contain"
        7,  # "bowl-contain"
        #
        8,  # "cup-contain"
        10,  # "cup-contain"
        12,  # "cup-contain"
        14,  # "cup-contain"
        16,  # "cup-contain"
        18,  # "cup-contain"
        #
        52 + 1,  # "ladle-contain"
        54 + 1,  # "ladle-contain"
        56 + 1,  # "ladle-contain"
        66 + 1,  # "mug-contain"
        69 + 1,  # "mug-contain"
        72 + 1,  # "mug-contain"
        75 + 1,  # "mug-contain"
        78 + 1,  # "mug-contain"
        81 + 1,  # "mug-contain"
        84 + 1,  # "mug-contain"
        87 + 1,  # "mug-contain"
        90 + 1,  # "mug-contain"
        93 + 1,  # "mug-contain"
        96 + 1,  # "mug-contain"
        99 + 1,  # "mug-contain"
        #
        102 + 1,  # "mug-contain"
        105 + 1,  # "mug-contain"
        108 + 1,  # "mug-contain"
        111 + 1,  # "mug-contain"
        114 + 1,  # "mug-contain"
        117 + 1,  # "mug-contain"
        120 + 1,  # "mug-contain"
        123 + 1,  # "mug-contain"
        #
        126,  # "pot-contain"
        128,  # "pot-contain"
    ]

    # 5
    pound = [
        20 + 1, #"hammer-pound"
        22 + 1, #"hammer-pound"
        24 + 1, #"hammer-pound"
        26 + 1, #"hammer-pound"
        #
        58 + 1, #'mallet-pound'
        60 + 1, #'mallet-pound'
        62 + 1, #'mallet-pound'
        64 + 1, #'mallet-pound'
        #
        184 + 1, #'tenderizer-pound'
    ]

    # 6
    support = [
        192 + 1,  # "turner-support"
        194 + 1,  # "turner-support"
        196 + 1,  # "turner-support"
        198 + 1,  # "turner-support"
        200 + 1,  # "turner-support"
        202 + 1,  # "turner-support"
        204 + 1,  # "turner-support"
    ]

    # 7
    wrap_grasp = [
        8 + 1, # "cup-wrap_grasp"
        10 + 1, # "cup-wrap_grasp"
        12 + 1, # "cup-wrap_grasp"
        14 + 1, # "cup-wrap_grasp"
        16 + 1, # "cup-wrap_grasp"
        18 + 1, # "cup-wrap_grasp"
        #
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
        #
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

###########################################################
# 
###########################################################
if __name__ == '__main__':


    class InferenceConfig(Affordance.AffordanceConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        MEAN_PIXEL = MEAN_PIXEL_
        USE_MINI_MASK = False
        RPN_ANCHOR_SCALES = RPN_ANCHOR_SCALES_
        IMAGE_RESIZE_MODE = IMAGE_RESIZE_MODE_
        IMAGE_MIN_DIM = IMAGE_MIN_DIM_
        IMAGE_MAX_DIM = IMAGE_MAX_DIM_
        MAX_GT_INSTANCES = MAX_GT_INSTANCES_
        DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES_
        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE_
        POST_NMS_ROIS_INFERENCE = POST_NMS_ROIS_INFERENCE_
        RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD_
        DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD_
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    detect_and_get_masks(model, config, args)
