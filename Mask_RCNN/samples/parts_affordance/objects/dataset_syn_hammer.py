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
from skimage.color import gray2rgb
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
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ###########################################################
# # Dataset
# ###########################################################

class AffordanceConfig(Config):
    """Configuration for training on the toy  dataset.
    # Derives from the base Config class and overrides some values.
    # """
    # Give the configuration a recognizable name
    NAME = "Affordance"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 205  # Background + objects

    ##################################
    ###  GPU
    ##################################

    GPU_COUNT = 2
    IMAGES_PER_GPU = 3  # 5 for 'heads', 3 for 'all'
    bs = GPU_COUNT * IMAGES_PER_GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

    STEPS_PER_EPOCH = (1056 + 348 + 319 + 348) // bs  ### DR
    VALIDATION_STEPS = (276 + 96 + 88 + 96) // bs

    # imgaug
    STEPS_PER_EPOCH *= 10
    VALIDATION_STEPS *= 10

    ##################################
    ###  FROM DATASET STATS
    ##################################
    ''' --- run datasetstats for all params below --- '''

    MAX_GT_INSTANCES = 20  # really only have 1 obj/image or max 3 labels/object
    DETECTION_MAX_INSTANCES = 20

    DETECTION_MIN_CONFIDENCE = 0.9

    MEAN_PIXEL = np.array([113.71, 112.65, 135.77])  ### SYN RGB DR + PR

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024

    # IMAGE_RESIZE_MODE = "square"
    # IMAGE_MIN_DIM = 640
    # IMAGE_MAX_DIM = 640
    # RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  ### 1024
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  ### 1024

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)

    TRAIN_ROIS_PER_IMAGE = 100  # TODO: DS bowl 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128

    # MASK_SHAPE = [56, 56]  # TODO: AFFORANCENET TRIED 14, 28, 56, 112, 224


# ###########################################################
# # Dataset
# ###########################################################

class AffordanceDataset(utils.Dataset):

    def load_Affordance(self, dataset_dir, subset):
        """Load a subset of the Affordance dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        #############################
        # OBJECT RPE
        #############################
        if subset == 'train':
            print("\n************************** LOADING TRAIN **************************")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/dr/coco_train_1056.json'))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/bench/coco_train_348.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/floor/coco_train_319.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/turn_table/coco_train_348.json')))
        if subset == 'val':
            print("\n************************** LOADING VAL **************************")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/dr/coco_val_276.json'))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/bench/coco_val_96.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/floor/coco_val_88.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/turn_table/coco_val_96.json')))
        elif subset == 'test':
            print("\n************************** LOADING TEST **************************")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/real/json/tools/rgb/test_100_hammer.json'))
            # annotations = json.load(
            #     open('/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer/turn_table/coco_val_96.json'))

        annotations = list(annotations.values())
        # print(annotations)
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

            image_path = os.path.join(dataset_dir, a['filename'])  # TODO: utils.load_image()
            print(image_path)  # TODO: print all files
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Affordance",
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
            #     "Affordance",
            #     image_id=a['depthfilename'],  # use file name as a unique image id
            #     path=image_path,
            #     width=width, height=height,
            #     polygons=polygons)

        self.add_class("Affordance", 1, "bowl-contain")
        self.add_class("Affordance", 2, "bowl-contain")
        self.add_class("Affordance", 3, "bowl-contain")
        self.add_class("Affordance", 4, "bowl-contain")
        self.add_class("Affordance", 5, "bowl-contain")
        self.add_class("Affordance", 6, "bowl-contain")
        self.add_class("Affordance", 7, "bowl-contain")

        # cup
        self.add_class("Affordance", 8, "cup-contain")
        self.add_class("Affordance", 8 + 1, "cup-wrap_grasp")
        self.add_class("Affordance", 10, "cup-contain")
        self.add_class("Affordance", 10 + 1, "cup-wrap_grasp")
        self.add_class("Affordance", 12, "cup-contain")
        self.add_class("Affordance", 12 + 1, "cup-wrap_grasp")
        self.add_class("Affordance", 14, "cup-contain")
        self.add_class("Affordance", 14 + 1, "cup-wrap_grasp")
        self.add_class("Affordance", 16, "cup-contain")
        self.add_class("Affordance", 16 + 1, "cup-wrap_grasp")
        self.add_class("Affordance", 18, "cup-contain")
        self.add_class("Affordance", 18 + 1, "cup-wrap_grasp")

        # hammer
        self.add_class("Affordance", 20, "hammer-grasp")
        self.add_class("Affordance", 20 + 1, "hammer-pound")
        self.add_class("Affordance", 22, "hammer-grasp")
        self.add_class("Affordance", 22 + 1, "hammer-pound")
        self.add_class("Affordance", 24, "hammer-grasp")
        self.add_class("Affordance", 24 + 1, "hammer-pound")
        self.add_class("Affordance", 26, "hammer-grasp")
        self.add_class("Affordance", 26 + 1, "hammer-pound")

        # knife
        self.add_class("Affordance", 28, "knife-grasp")
        self.add_class("Affordance", 28 + 1, "knife-cut")
        self.add_class("Affordance", 30, "knife-grasp")
        self.add_class("Affordance", 30 + 1, "knife-cut")
        self.add_class("Affordance", 32, "knife-grasp")
        self.add_class("Affordance", 32 + 1, "knife-cut")
        self.add_class("Affordance", 34, "knife-grasp")
        self.add_class("Affordance", 34 + 1, "knife-cut")
        self.add_class("Affordance", 36, "knife-grasp")
        self.add_class("Affordance", 36 + 1, "knife-cut")
        self.add_class("Affordance", 38, "knife-grasp")
        self.add_class("Affordance", 38 + 1, "knife-cut")
        self.add_class("Affordance", 40, "knife-grasp")
        self.add_class("Affordance", 40 + 1, "knife-cut")
        self.add_class("Affordance", 42, "knife-grasp")
        self.add_class("Affordance", 42 + 1, "knife-cut")
        self.add_class("Affordance", 44, "knife-grasp")
        self.add_class("Affordance", 44 + 1, "knife-cut")
        self.add_class("Affordance", 46, "knife-grasp")
        self.add_class("Affordance", 46 + 1, "knife-cut")
        self.add_class("Affordance", 48, "knife-grasp")
        self.add_class("Affordance", 48 + 1, "knife-cut")
        self.add_class("Affordance", 50, "knife-grasp")
        self.add_class("Affordance", 50 + 1, "knife-cut")

        # ladle
        self.add_class("Affordance", 52, "ladle-grasp")
        self.add_class("Affordance", 52 + 1, "ladle-contain")
        self.add_class("Affordance", 54, "ladle-grasp")
        self.add_class("Affordance", 54 + 1, "ladle-contain")
        self.add_class("Affordance", 56, "ladle-grasp")
        self.add_class("Affordance", 56 + 1, "ladle-contain")

        # mallet
        self.add_class("Affordance", 58, "mallet-grasp")
        self.add_class("Affordance", 58 + 1, "mallet-pound")
        self.add_class("Affordance", 60, "mallet-grasp")
        self.add_class("Affordance", 60 + 1, "mallet-pound")
        self.add_class("Affordance", 62, "mallet-grasp")
        self.add_class("Affordance", 62 + 1, "mallet-pound")
        self.add_class("Affordance", 64, "mallet-grasp")
        self.add_class("Affordance", 64 + 1, "mallet-pound")

        self.add_class("Affordance", 66, "mug-grasp")
        self.add_class("Affordance", 66 + 1, "mug-contain")
        self.add_class("Affordance", 66 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 69, "mug-grasp")
        self.add_class("Affordance", 69 + 1, "mug-contain")
        self.add_class("Affordance", 69 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 72, "mug-grasp")
        self.add_class("Affordance", 72 + 1, "mug-contain")
        self.add_class("Affordance", 72 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 75, "mug-grasp")
        self.add_class("Affordance", 75 + 1, "mug-contain")
        self.add_class("Affordance", 75 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 78, "mug-grasp")
        self.add_class("Affordance", 78 + 1, "mug-contain")
        self.add_class("Affordance", 78 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 81, "mug-grasp")
        self.add_class("Affordance", 81 + 1, "mug-contain")
        self.add_class("Affordance", 81 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 84, "mug-grasp")
        self.add_class("Affordance", 84 + 1, "mug-contain")
        self.add_class("Affordance", 84 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 87, "mug-grasp")
        self.add_class("Affordance", 87 + 1, "mug-contain")
        self.add_class("Affordance", 87 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 90, "mug-grasp")
        self.add_class("Affordance", 90 + 1, "mug-contain")
        self.add_class("Affordance", 90 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 93, "mug-grasp")
        self.add_class("Affordance", 93 + 1, "mug-contain")
        self.add_class("Affordance", 93 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 96, "mug-grasp")
        self.add_class("Affordance", 96 + 1, "mug-contain")
        self.add_class("Affordance", 96 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 99, "mug-grasp")
        self.add_class("Affordance", 99 + 1, "mug-contain")
        self.add_class("Affordance", 99 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 102, "mug-grasp")
        self.add_class("Affordance", 102 + 1, "mug-contain")
        self.add_class("Affordance", 102 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 105, "mug-grasp")
        self.add_class("Affordance", 105 + 1, "mug-contain")
        self.add_class("Affordance", 105 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 108, "mug-grasp")
        self.add_class("Affordance", 108 + 1, "mug-contain")
        self.add_class("Affordance", 108 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 111, "mug-grasp")
        self.add_class("Affordance", 111 + 1, "mug-contain")
        self.add_class("Affordance", 111 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 114, "mug-grasp")
        self.add_class("Affordance", 114 + 1, "mug-contain")
        self.add_class("Affordance", 114 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 117, "mug-grasp")
        self.add_class("Affordance", 117 + 1, "mug-contain")
        self.add_class("Affordance", 117 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 120, "mug-grasp")
        self.add_class("Affordance", 120 + 1, "mug-contain")
        self.add_class("Affordance", 120 + 2, "mug-wrap_grasp")
        self.add_class("Affordance", 123, "mug-grasp")
        self.add_class("Affordance", 123 + 1, "mug-contain")
        self.add_class("Affordance", 123 + 2, "mug-wrap_grasp")

        # pot
        self.add_class("Affordance", 126, "pot-contain")
        self.add_class("Affordance", 126 + 1, "pot-wrap_grasp")
        self.add_class("Affordance", 128, "pot-contain")
        self.add_class("Affordance", 128 + 1, "pot-wrap_grasp")

        # saw
        self.add_class("Affordance", 130, "saw-grasp")
        self.add_class("Affordance", 130 + 1, "saw-cut")
        self.add_class("Affordance", 132, "saw-grasp")
        self.add_class("Affordance", 132 + 1, "saw-cut")
        self.add_class("Affordance", 134, "saw-grasp")
        self.add_class("Affordance", 134 + 1, "saw-cut")

        # scissors
        self.add_class("Affordance", 136, "scissors-grasp")
        self.add_class("Affordance", 136 + 1, "scissors-cut")
        self.add_class("Affordance", 138, "scissors-grasp")
        self.add_class("Affordance", 138 + 1, "scissors-cut")
        self.add_class("Affordance", 140, "scissors-grasp")
        self.add_class("Affordance", 140 + 1, "scissors-cut")
        self.add_class("Affordance", 142, "scissors-grasp")
        self.add_class("Affordance", 142 + 1, "scissors-cut")
        self.add_class("Affordance", 144, "scissors-grasp")
        self.add_class("Affordance", 144 + 1, "scissors-cut")
        self.add_class("Affordance", 146, "scissors-grasp")
        self.add_class("Affordance", 146 + 1, "scissors-cut")
        self.add_class("Affordance", 148, "scissors-grasp")
        self.add_class("Affordance", 148 + 1, "scissors-cut")
        self.add_class("Affordance", 150, "scissors-grasp")
        self.add_class("Affordance", 150 + 1, "scissors-cut")

        # scoop
        self.add_class("Affordance", 152, "scoop-grasp")
        self.add_class("Affordance", 152 + 1, "scoop-scoop")
        self.add_class("Affordance", 154, "scoop-grasp")
        self.add_class("Affordance", 154 + 1, "scoop-scoop")

        # shears
        self.add_class("Affordance", 156, "shears-grasp")
        self.add_class("Affordance", 156 + 1, "shears-cut")
        self.add_class("Affordance", 158, "shears-grasp")
        self.add_class("Affordance", 158 + 1, "shears-cut")

        # shovel
        self.add_class("Affordance", 160, "shovel-grasp")
        self.add_class("Affordance", 160 + 1, "shovel-scoop")
        self.add_class("Affordance", 162, "shovel-grasp")
        self.add_class("Affordance", 162 + 1, "shovel-scoop")

        # spoon
        self.add_class("Affordance", 164, "spoon-grasp")
        self.add_class("Affordance", 164 + 1, "spoon-scoop")
        self.add_class("Affordance", 166, "spoon-grasp")
        self.add_class("Affordance", 166 + 1, "spoon-scoop")
        self.add_class("Affordance", 168, "spoon-grasp")
        self.add_class("Affordance", 168 + 1, "spoon-scoop")
        self.add_class("Affordance", 170, "spoon-grasp")
        self.add_class("Affordance", 170 + 1, "spoon-scoop")
        self.add_class("Affordance", 172, "spoon-grasp")
        self.add_class("Affordance", 172 + 1, "spoon-scoop")
        self.add_class("Affordance", 174, "spoon-grasp")
        self.add_class("Affordance", 174 + 1, "spoon-scoop")
        self.add_class("Affordance", 176, "spoon-grasp")
        self.add_class("Affordance", 176 + 1, "spoon-scoop")
        self.add_class("Affordance", 178, "spoon-grasp")
        self.add_class("Affordance", 178 + 1, "spoon-scoop")
        self.add_class("Affordance", 180, "spoon-grasp")
        self.add_class("Affordance", 180 + 1, "spoon-scoop")
        self.add_class("Affordance", 182, "spoon-grasp")
        self.add_class("Affordance", 182 + 1, "spoon-scoop")

        self.add_class("Affordance", 184, "tenderizer-grasp")
        self.add_class("Affordance", 184 + 1, "tenderizer-pound")

        # trowel
        self.add_class("Affordance", 186, "trowel-grasp")
        self.add_class("Affordance", 186 + 1, "trowel-scoop")
        self.add_class("Affordance", 188, "trowel-grasp")
        self.add_class("Affordance", 188 + 1, "trowel-scoop")
        self.add_class("Affordance", 190, "trowel-grasp")
        self.add_class("Affordance", 190 + 1, "trowel-scoop")

        # turner
        self.add_class("Affordance", 192, "turner-grasp")
        self.add_class("Affordance", 192 + 1, "turner-support")
        self.add_class("Affordance", 194, "turner-grasp")
        self.add_class("Affordance", 194 + 1, "turner-support")
        self.add_class("Affordance", 196, "turner-grasp")
        self.add_class("Affordance", 196 + 1, "turner-support")
        self.add_class("Affordance", 198, "turner-grasp")
        self.add_class("Affordance", 198 + 1, "turner-support")
        self.add_class("Affordance", 200, "turner-grasp")
        self.add_class("Affordance", 200 + 1, "turner-support")
        self.add_class("Affordance", 202, "turner-grasp")
        self.add_class("Affordance", 202 + 1, "turner-support")
        self.add_class("Affordance", 204, "turner-grasp")
        self.add_class("Affordance", 204 + 1, "turner-support")

    def load_image_rgb_depth(self, image_id):

        file_path = np.str(image_id).split("rgb.jpg")[0]

        rgb = skimage.io.imread(file_path + "rgb.jpg")
        depth = skimage.io.imread(file_path + "depth.png")

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
        # If not a Affordance dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Affordance":
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
        if info["source"] == "Affordance":
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
