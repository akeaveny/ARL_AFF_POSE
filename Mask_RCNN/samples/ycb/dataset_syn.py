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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print("ROOT_DIR: ", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log

import tensorflow as tf

from skimage.color import gray2rgb

# ###########################################################
# # Dataset
# ###########################################################

class YCBConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "YCB"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 31  # Background + objects

    ##################################
    ###  GPU
    ##################################

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # 5 for 'heads', 3 for 'all'
    bs = GPU_COUNT * IMAGES_PER_GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config_ = tf.ConfigProto()
    config_.gpu_options.allow_growth = True
    sess = tf.Session(config=config_)

    ##################################
    ###  Backbone
    ##################################

    ### BACKBONE = "resnet50"

    ##################################
    ###
    ##################################

    LEARNING_RATE = 1e-03
    WEIGHT_DECAY = 0.0001
    ### TRAIN_BN = None

    ##################################
    ###  NUM OF IMAGES
    ##################################

    # tools
    STEPS_PER_EPOCH = (4000) // bs
    VALIDATION_STEPS = (1000) // bs

    ##################################
    ###  FROM DATASET STATS
    ##################################
    ''' --- run datasetstats for all params below --- '''

    # MEAN_PIXEL = np.array([113.45, 112.19, 130.92])  ### SYN RGB DR + PR
    # MEAN_PIXEL = np.array([96.78, 94.99, 103.82])  ### REAL

    # IMAGE_RESIZE_MODE = "crop"
    # IMAGE_MIN_DIM = 384
    # IMAGE_MAX_DIM = 384
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    # MASK_SHAPE = [56, 56]  # TODO: AFFORANCENET TRIED 14, 28, 56, 112, 224

    MAX_GT_INSTANCES = 20  # really only have 1 obj/image or max 3 labels/object
    DETECTION_MAX_INSTANCES = 20

    DETECTION_MIN_CONFIDENCE = 0.9

    TRAIN_ROIS_PER_IMAGE = 100  # TODO: DS bowl 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

############################################################
#  Dataset
############################################################

class YCBDataset(utils.Dataset):

    def load_YCB(self, dataset_dir, subset):
        """Load a subset of the YCB dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("YCB", 1, "master_chef_can_03_rollable")
        self.add_class("YCB", 2, "cracker_box_02_wrap_grasp")
        self.add_class("YCB", 3, "sugar_box_02_wrap_grasp")
        self.add_class("YCB", 4, "tomato_soup_can_04_rollable")
        self.add_class("YCB", 5, "mustard_bottle_02_wrap_grasp")
        self.add_class("YCB", 6, "tuna_fish_can_03_rollable")
        self.add_class("YCB", 7, "pudding_box_02_wrap_grasp")
        self.add_class("YCB", 8, "gelatin_box_02_wrap_grasp")
        self.add_class("YCB", 9, "potted_meat_can_02_wrap_grasp")
        self.add_class("YCB", 10, "banana_02_wrap_grasp")
        self.add_class("YCB", 11, "pitcher_base_01_grasp")
        self.add_class("YCB", 12, "pitcher_base_02_wrap_grasp")
        self.add_class("YCB", 13, "pitcher_base_04_pourable")
        self.add_class("YCB", 14, "bleach_cleanser_02_wrap_grasp")
        self.add_class("YCB", 15, "bowl_04_pourable")
        self.add_class("YCB", 16, "bowl_02_wrap_grasp")
        self.add_class("YCB", 17, "mug_01_grasp")
        self.add_class("YCB", 18, "mug_04_pourable")
        self.add_class("YCB", 19, "mug_02_wrap_grasp")
        self.add_class("YCB", 20, "power_drill_01_grasp")
        self.add_class("YCB", 21, "power_drill_08_drill")
        self.add_class("YCB", 22, "power_drill_09_support")
        self.add_class("YCB", 23, "wood_block_09_support")
        self.add_class("YCB", 24, "scissors_01_grasp")
        self.add_class("YCB", 25, "scissors_05_cut")
        self.add_class("YCB", 26, "large_marker_03_rollable")
        self.add_class("YCB", 27, "large_clamp_06_clamp")
        self.add_class("YCB", 28, "large_clamp_07_squeeze")
        self.add_class("YCB", 29, "extra_large_clamp_06_clamp")
        self.add_class("YCB", 30, "extra_large_clamp_07_squeeze")
        self.add_class("YCB", 31, "foam_brick_09_support")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        # '/data/Akeaveny/Datasets/YCB_Video_Dataset/via_region_data_combined.json'
        if subset == 'train':
            print("\n************************** LOADING TRAIN **************************")
            annotations = {}
            ### tools
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/bench/coco_train_8001.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/floor/coco_train_7990.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/turn_table/coco_train_7920.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/dr/coco_train_23560.json')))

        elif subset == 'val':
            print("\n************************** LOADING VAL **************************")
            annotations = {}
            ### tools
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/bench/coco_val_1008.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/floor/coco_test_1007.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/turn_table/coco_val_998.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/dr/coco_val_2976.json')))

        if subset == 'test':
            print("\n************************** LOADING TEST **************************")
            annotations = {}
            ### tools
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/bench/coco_test_1008.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/floor/coco_test_1007.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/turn_table/coco_train_7920.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/ycb_syn/json/rgb/tools/dr/coco_test_2976.json')))

        annotations = list(annotations.values())
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        # annotations = [a for a in annotations]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            print(image_path)

            self.add_image(
                "YCB",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

            ## TODO: visualize depth images
            # image_path = os.path.join(dataset_dir, a['depthfilename'])
            # print(image_path)  # TODO: print all files
            # image = skimage.io.imread(image_path)
            # height, width = image.shape[:2]
            #
            # self.add_image(
            #     "YCB",
            #     image_id=a['depthfilename'],  # use file name as a unique image id
            #     path=image_path,
            #     width=width, height=height,
            #     polygons=polygons)

    def load_image_rgb_depth(self, image_id):

        file_path = np.str(image_id).split("rgb.png")[0]

        rgb = skimage.io.imread(file_path + "rgb.png")
        depth = skimage.io.imread(file_path + "depth.png")

        # print("DEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))
        # self.NDDS_DEPTH_CONST = 10e3 / (2 ** 8 - 1)
        # depth = np.array(depth, dtype=np.int32) * self.NDDS_DEPTH_CONST
        # depth = np.array(depth, dtype=np.int32)
        # print("DEPTH:\tMin:\t{}, Max:\t{}, dtype:\t{} ".format(np.min(depth), np.max(depth), depth.dtype))

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        return rgb[..., :3], skimage.color.gray2rgb(depth)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a YCB dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "YCB":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_IDs = np.zeros([len(info["polygons"])], dtype=np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            class_IDs[i] = p['class_id']

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_IDs

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "YCB":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

# ###########################################################
# # Dataset
# ###########################################################

def color_map():
    color_map_dic = {
    0:  [0, 0, 0],
    1:  [128, 128,   0],
    2:  [  0, 128, 128],
    3:  [128,   0, 128],
    4:  [128,   0,   0],
    5:  [  0, 128,   0],
    6:  [  0,   0, 128],
    7:  [255, 255,   0],
    8:  [255,   0, 255],
    9:  [  0, 255, 255],
    10: [255,   0,   0],
    11: [  0, 255,   0],
    12: [  0,   0, 255],
    13: [ 92,  112, 92],
    14: [  0,   0,  70],
    15: [  0,  60, 100],
    16: [  0,  80, 100],
    17: [  0,   0, 230],
    18: [119,  11,  32],
    19: [  0,   0, 121]
    }
    return color_map_dic
