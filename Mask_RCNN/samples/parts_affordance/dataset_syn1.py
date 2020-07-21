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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print("ROOT_DIR: ", ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log

# ###########################################################
# # Dataset
# ###########################################################

class AffordanceConfig(Config):
    """Configuration for training on the toy  dataset.
    # Derives from the base Config class and overrides some values.
    # """
    # Give the configuration a recognizable name
    NAME = "Affordance"

    NUM_CLASSES = 1 + 7 # Number of classes (including background)

    ##################################
    ###  GPU
    ##################################

    GPU_COUNT = 2
    IMAGES_PER_GPU = 1
    bs = GPU_COUNT * IMAGES_PER_GPU

    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ### gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    ### config = tf.ConfigProto()
    ### config.gpu_options.per_process_gpu_memory_fraction = 0.9
    ### tf.keras.backend.set_session(tf.Session(config=config))

    ##################################
    ###  Backbone
    ##################################

    # BACKBONE = "resnet50" or "resnet101"
    # RESNET_ARCHITECTURE = "resnet50"

    ##################################
    ###
    ##################################

    LEARNING_RATE = 1e-03
    WEIGHT_DECAY = 0.0001

    ##################################
    ###  NUM OF IMAGES
    ##################################

    # Number of training steps per epoch
    STEPS_PER_EPOCH = (671) // bs
    VALIDATION_STEPS = (295) // bs

    # STEPS_PER_EPOCH = (5000) // bs
    # VALIDATION_STEPS = (1250) // bs

    # STEPS_PER_EPOCH = (17133 + 400) // bs
    # VALIDATION_STEPS = (7308) // bs

    # STEPS_PER_EPOCH = (17133 + 17347) // bs
    # VALIDATION_STEPS = (7308 + 7342) // bs

    ##################################
    ###  FROM DATASET STATS
    ##################################
    ''' --- run datasetstats for all params below --- '''


    MAX_GT_INSTANCES = 3 # really only have 1 obj/image
    DETECTION_MAX_INSTANCES = 3

    DETECTION_MIN_CONFIDENCE = 0.9
    # DETECTION_NMS_THRESHOLD = 0.3

    ### IMAGE_CHANNEL_COUNT = 3   # TODO: change filename and conv1 layer (https://github.com/matterport/Mask_RCNN/wiki) & load_image
    # MEAN_PIXEL = np.array([182.12, 182.12, 182.12]) ### SYN depth
    # MEAN_PIXEL = np.array([111.96, 112.30, 131.78])  ### SYN rgb
    MEAN_PIXEL = np.array([91.13, 88.92, 98.65])  ### REAL

    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56) ### AFFORANCENET: TRIED 14, 28, 56, 112, 224

    ### MASK_POOL_SIZE = 28
    ### MASK_SHAPE = [56, 56]

    TRAIN_ROIS_PER_IMAGE = 100 ### like batch size for the second stage of the model

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

# ###########################################################
# # Dataset
# ###########################################################

class AffordanceDataset(utils.Dataset):

    def load_Affordance(self, dataset_dir, subset):
        """Load a subset of the Affordance dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        #  1 - 'grasp'
        #   2 - 'cut'
        #   3 - 'scoop'
        #   4 - 'contain'
        #   5 - 'pound'
        #   6 - 'support'
        #   7 - 'wrap-grasp'
        self.add_class("Affordance", 1, "grasp")
        self.add_class("Affordance", 2, "cut")
        self.add_class("Affordance", 3, "scoop")
        self.add_class("Affordance", 4, "contain")
        self.add_class("Affordance", 5, "pound")
        self.add_class("Affordance", 6, "support")
        self.add_class("Affordance", 7, "wrap-grasp")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        if subset == 'train':
            print("------------------LOADING TRAIN!------------------")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/dr/train350.json'))
            ### annotations = {}
            ### annotations.update(json.load(
            ###     open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/missing/missing.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/bench/train117.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/floor/train117.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/turn_table/train117.json')))
        elif subset == 'val':
            print("------------------LOADING VAL!--------------------")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/dr/val150.json'))
            ## annotations = {}
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/bench/val50.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/floor/val50.json')))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb1/turn_table/val50.json')))
        elif subset == 'test':
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance_combined/real/json/tools/rgb/test_100.json'))

        annotations = list(annotations.values())
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename']) # TODO: rgb (filename) or depth (depthfilename)
            print(image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Affordance",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons)

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