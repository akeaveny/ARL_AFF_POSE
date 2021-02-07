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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser(description='Get Stats from Image Dataset')

parser.add_argument('--detect', required=False, default='rgb',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/elevator_dataset/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='real',
                    type=str,
                    metavar='real or syn')
parser.add_argument('--dataset_split', required=False, default='test',
                    type=str,
                    metavar='test or val')

parser.add_argument('--save_inference_images', required=False,
                    default='test_maskrcnn/',
                    type=str,
                    metavar="/path/to/YCB/dataset/")

parser.add_argument('--num_frames', required=False, default=50,
                    type=int,
                    metavar='test or val')

parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/ or Logs and checkpoints directory (default=logs/)")

parser.add_argument('--show_plots', required=False, default=True,
                    type=bool,
                    metavar='show plots from matplotlib')
parser.add_argument('--save_output', required=False, default=False,
                    type=bool,
                    metavar='save terminal output to text file')

args = parser.parse_args()

#####################
# clear old results
#####################
for img in os.listdir(args.dataset + args.save_inference_images):
    os.remove(os.path.join(args.dataset + args.save_inference_images, img))

############################################################
#  REAL OR SYN
############################################################
if args.dataset_type == 'real':
    import dataset_real as ARLElevator
    save_to_folder = '/images/test_images_real/'
    # MEAN_PIXEL_ = np.array([103.57, 103.38, 103.52])  ### REAL
    MEAN_PIXEL_ = np.array([93.70, 92.43, 89.58])  ### TEST
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 1
    DETECTION_MAX_INSTANCES_ = 1
    DETECTION_MIN_CONFIDENCE_ = 0.8  ### SYN TOOLS: 0.7 real / 0.85 test, TEST Tools: 0.7 real / 0.975 test
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.5
    ### sqaure ###
    CROP = False
    IMAGE_RESIZE_MODE_ = "square"
    IMAGE_MIN_DIM_ = 640
    IMAGE_MAX_DIM_ = 640

elif args.dataset_type == 'syn':
    import dataset_syn as ARLElevator
    save_to_folder = '/images/test_images_syn/'
    MEAN_PIXEL_ = np.array([124.65, 119.64, 113.10])  ### SYN
    RPN_ANCHOR_SCALES_ = (16, 32, 64, 128, 256)
    ### config ###
    MAX_GT_INSTANCES_ = 10
    DETECTION_MAX_INSTANCES_ = 10
    DETECTION_MIN_CONFIDENCE_ = 0.9
    POST_NMS_ROIS_INFERENCE_ = 100
    RPN_NMS_THRESHOLD_ = 0.8
    DETECTION_NMS_THRESHOLD_ = 0.8
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
elif args.detect == 'rgbd':
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

    # print("\tobject_ids", cur_class_ids)
    if cur_masks.shape[-1] > 0:

        for i in range(cur_masks.shape[-1]):
            ### object ID to affordance ID
            cur_class_ids[i] = map_affordance_label(cur_class_ids[i])
            # print("\tPred Aff Label:", cur_class_ids[i])

            ### instance_mask = instance_mask_one * (mask_index+1)
            instance_mask = instance_mask_one * cur_class_ids[i]
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    print("\tPred aff_label:", np.unique(instance_masks)[1:])

    ########################
    #  add color to masks
    ########################
    instance_to_color = ARLElevator.color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    return instance_masks, color_masks

def detect_and_get_masks(model, config, args):

    ########################
    #  Load test images
    ########################

    print("args.dataset_split", args.dataset_split)
    dataset = ARLElevator.ARLElevatorDataset()
    dataset.load_ARLElevator(args.dataset, args.dataset_split)
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

        mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_gt.png'
        color_mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_pred_color.png'
        cropped_mask_addr = args.dataset + args.save_inference_images + str(num_image) + '_mask_pred.png'
        print("\tmask_addr:", mask_addr)

        ##############################
        ### ground truth
        ##############################

        rgb = np.array(skimage.io.imread(rgb_addr))
        depth = np.array(skimage.io.imread(depth_addr))
        gt_label = np.array(skimage.io.imread(gt_mask_addr))

        ##############################
        ### OBJECT IDS TO AFF LABELS
        ##############################

        object_ids = np.unique(gt_label)[1:]
        # print("\tGT object_ids:", object_ids)

        gt_aff_mask = np.zeros((gt_label.shape[0], gt_label.shape[1]), dtype=np.uint8)
        gt_mask_one = np.ones((gt_label.shape[0], gt_label.shape[1]), dtype=np.uint8)

        for object_id in object_ids:
            aff_label = map_affordance_label(object_id)
            # print("\tGT aff_label:", aff_label)

            gt_mask = gt_mask_one * aff_label
            gt_aff_mask = np.where(gt_label == object_id, gt_mask, gt_aff_mask).astype(np.uint8)

        print("\tGT aff_label:", np.unique(gt_aff_mask)[1:])

        # plt.subplot(1, 2, 1)
        # plt.title("gt_label")
        # print("Object ids", np.unique(gt_label))
        # plt.imshow(gt_label)
        # plt.subplot(1, 2, 2)
        # plt.title("gt_masks")
        # print("Aff labels", np.unique(gt_aff_mask))
        # plt.imshow(gt_aff_mask)
        # plt.show()
        # plt.ioff()

        ######################
        # configure depth
        ######################
        print("\tDEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))

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
            gt_aff_mask = gt_aff_mask[y:y + config.IMAGE_MIN_DIM, x:x + config.IMAGE_MIN_DIM]

        ##############################
        #  Detect
        ##############################

        if args.detect == 'rgb':
            cur_detect = model.detect([rgb], verbose=0)[0]

        elif args.detect == 'rgbd' or args.detect == 'rgbd+':
            cur_detect = model.detectWdepth([rgb], [depth], verbose=0)[0]

        # get instance_masks
        instance_mask, color_mask = seq_get_masks(rgb, cur_detect, gt_label, args)

        ##############################
        #  SAVE IMAGES
        ##############################

        cv2.imwrite(mask_addr, gt_aff_mask)
        cv2.imwrite(color_mask_addr, color_mask)
        cv2.imwrite(cropped_mask_addr, instance_mask)

        if args.show_plots:  # TODO: boolean string
            # print("\tGT shape:", gt_aff_mask.shape)
            # print("\tPred shape:", instance_mask.shape)
            # print("\tresize_pred shape:", instance_mask.shape

            masks = cur_detect['masks']
            # masks = gt_aff_mask
            class_ids = np.array(cur_detect['class_ids']) - 1
            ### print("class_ids:  ", class_ids)
            class_names = np.array(['grasp','screw','scoop','pound','support'])
            visualize.display_instances(image=rgb, boxes=cur_detect['rois'],masks=masks,
                                                       class_ids=class_ids, class_names=class_names,
                                                       scores=cur_detect['scores'],
                                                       title="Predictions",
                                                       show_bbox=False, show_mask=True)
            plt.tight_layout()
            # plt.show()

            ### plotting
            # cv2.imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            # cv2.imshow("depth", np.array(depth, dtype=np.uint8))
            # cv2.imshow("gt", gt_aff_mask * 25)
            # cv2.imshow("pred", instance_mask * 25)
            mask_file_path = os.getcwd() + save_to_folder + "pred.png"
            plt.savefig(mask_file_path, bbox_inches='tight')
            masked_image = cv2.imread(mask_file_path)
            cv2.imshow("masked_image", masked_image)
            cv2.waitKey(1)

        ###########################################################
# LOOKUP FROM OBJECT ID TO AFFORDANCE LABEL
###########################################################
def map_affordance_label(current_id):

    # 1
    grasp = [
        1, # 'mallet_1_grasp'
        3, # 'spatula_1_grasp'
        5, # 'wooden_spoon_1_grasp'
        7, # 'screwdriver_1_grasp'
        9, # 'garden_shovel_1_grasp'
    ]

    screw = [
        8, # 'screwdriver_2_screw'
    ]

    scoop = [
        6, # 'wooden_spoon_3_scoop'
        10, # 'garden_shovel_3_scoop'
    ]

    pound = [
        2, # 'mallet_4_pound'
    ]

    support = [
        4, # 'spatula_2_support'
    ]

    if current_id in grasp:
        return 1
    elif current_id in screw:
        return 2
    elif current_id in scoop:
        return 3
    elif current_id in pound:
        return 4
    elif current_id in support:
        return 5
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        exit(1)

###########################################################
# 
###########################################################
if __name__ == '__main__':


    class InferenceConfig(ARLElevator.ARLElevatorConfig):
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
