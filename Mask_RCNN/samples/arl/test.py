"""
------------------------------------------------------------
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

parser.add_argument('--detect', required=False, default='rgb',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='real',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')

parser.add_argument('--save_inference_images', required=False, default='test_maskrcnn_real/',
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

############################################################
#  REAL OR SYN
############################################################
if args.dataset_type == 'real':
    import dataset_real as ARL
    save_to_folder = '/images/test_images_real/'
    MEAN_PIXEL_ = np.array([114.34, 109.86, 101.07])  ### REAL
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.9  # 0.975
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
    import dataset_syn as ARL
    save_to_folder = '/images/test_images_syn/'
    # MEAN_PIXEL_ = np.array([157.72, 151.18, 155.02])  ### SYN
    MEAN_PIXEL_ = np.array([114.34, 109.86, 101.07])  ### REAL
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 2
    DETECTION_MAX_INSTANCES_ = 2
    DETECTION_MIN_CONFIDENCE_ = 0.9  # 0.975
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

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    print("\tobject_ids", cur_class_ids)
    if cur_masks.shape[-1] > 0:

        for i in range(cur_masks.shape[-1]):
            print("\tPred Label:", cur_class_ids[i])

            ### instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * cur_class_ids[i]
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    ########################
    #  add color to masks
    ########################
    instance_to_color = ARL.color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    return instance_masks, color_masks

def detect_and_get_masks(model, config, args):

    ########################
    #  Load test images
    ########################

    print("args.dataset_split", args.dataset_split)
    dataset = ARL.ARLDataset()
    dataset.load_ARL(args.dataset, args.dataset_split)
    dataset.prepare()

    config.display()

    print("Num of Test Images: {}".format(len(dataset.image_ids)))

    # select random test images
    np.random.seed(0)
    test_idx = np.random.choice(np.arange(0, len(dataset.image_ids), 1), size=int(args.num_frames), replace=False)
    print("Chosen Files \n", len(test_idx))

    for num_image, idx in enumerate(test_idx):

        print("Running Inference on Image {} ..".format(num_image))

        image_file1 = dataset.image_reference(idx)
        image_file2 = image_file1.split(args.dataset)[1]  # remove dataset path
        idx = image_file2.split('_rgb')[0] # remove _rgb label

        ##############################
        ##############################

        rgb_addr = args.dataset + idx + '_rgb.png'
        depth_addr = args.dataset + idx + '_depth.png'
        gt_mask_addr = args.dataset + idx + '_label.png'
        # print(rgb_addr)

        if os.path.isfile(rgb_addr) == False:
            continue
        if os.path.isfile(depth_addr) == False:
            continue
        if os.path.isfile(gt_mask_addr) == False:
            continue

        mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_og.png'
        color_mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_color.png'
        cropped_mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_cropped.png'
        print("\tmask_addr:", mask_addr)

        ##############################
        ### ground truth
        ##############################

        rgb = np.array(skimage.io.imread(rgb_addr))
        depth = np.array(skimage.io.imread(depth_addr))
        gt_label = np.array(skimage.io.imread(gt_mask_addr))
        print("\tGT Label:", np.unique(gt_label))

        ######################
        # configure depth
        ######################
        print("\tDEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))
        # depth = np.array(depth, dtype=np.int32)
        #
        # NDDS_DEPTH_CONST = 10e3 / (2 ** 8 - 1)
        # depth = depth * NDDS_DEPTH_CONST
        # depth = np.array(depth, dtype=np.int32)
        #
        # print("DEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        rgb, depth = rgb[..., :3], skimage.color.gray2rgb(depth)

        ##############################
        #  CROP
        ##############################
        rgb_og = rgb

        if CROP == True:
            # Pick a random crop
            h, w = rgb.shape[:2]

            x = (w - config.IMAGE_MIN_DIM) // 2
            y = (h - config.IMAGE_MIN_DIM) // 2

            rgb = rgb[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            depth = depth[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]
            gt_label = gt_label[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]

        # plt.subplot(1, 3, 1)
        # plt.title("rgb")
        # plt.imshow(rgb_og)
        # plt.subplot(1, 3, 2)
        # plt.title("rgb cropped")
        # plt.imshow(rgb)
        # plt.subplot(1, 3, 3)
        # plt.title("depth cropped")
        # plt.imshow(np.array(depth, dtype=np.uint8))
        # plt.show()
        # plt.ioff()

        ##############################
        #  Detect
        ##############################

        if args.detect == 'rgb':
            cur_detect = model.detect([rgb], verbose=0)[0]

        elif args.detect == 'rgbd' or args.detect == 'rgbd+':
            cur_detect = model.detectWdepth([rgb], [depth], verbose=0)[0]

        # get instance_masks
        instance_mask, color_mask = seq_get_masks(rgb, cur_detect, gt_label, args)

        ####################

        cv2.imwrite(mask_addr, instance_mask)
        cv2.imwrite(color_mask_addr, color_mask)
        cv2.imwrite(cropped_mask_addr, gt_label )

        if args.show_plots:  # TODO: boolean string
            print("GT shape:", gt_label.shape)
            print("Pred shape:", instance_mask.shape)
            print("resize_pred shape:", instance_mask.shape)

            cv2.imshow("gt", gt_label * 25)
            cv2.imshow("resize pred", instance_mask * 25)
            cv2.waitKey(0)

###########################################################
# 
###########################################################
if __name__ == '__main__':


    class InferenceConfig(ARL.ARLConfig):
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
        # POST_NMS_ROIS_INFERENCE = POST_NMS_ROIS_INFERENCE_
        # RPN_NMS_THRESHOLD = RPN_NMS_THRESHOLD_
        # DETECTION_NMS_THRESHOLD = DETECTION_NMS_THRESHOLD_
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    detect_and_get_masks(model, config, args)
