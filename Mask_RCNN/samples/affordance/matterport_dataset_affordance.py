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

    # ========== GPU config ================
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    bs = GPU_COUNT * IMAGES_PER_GPU

    # Color   mean (RGB): 94.58 92.69 101.30
    MEAN_PIXEL = np.array([94.58, 92.69, 101.30])

    BACKBONE = "resnet50"
    RESNET_ARCHITECTURE = "resnet50"

    IMAGE_PADDING = True
    IMAGE_MAX_DIM = 640
    IMAGE_MIN_DIM = 480
    # IMAGE_MAX_DIM = 256
    # IMAGE_MIN_DIM = 256

    IMAGE_RESIZE_MODE = "none"

    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    #
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    # ROI_POSITIVE_RATIO = 0.33
    # RPN_BBOX_STD_DEV = [0.1, 0.1, 0.2, 0.2]

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = (15000 + 1791) // bs
    VALIDATION_STEPS = (3750 + 320) // bs
    # STEPS_PER_EPOCH = (1791) // bs
    # VALIDATION_STEPS = (320) // bs

    # TRAIN_BN = True # default is true
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_MAX_INSTANCES = 2

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

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        if subset == 'train':
            print("------------------LOADING TRAIN!------------------")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_real_train.json'))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_syn_train.json')))
        elif subset == 'val':
            print("------------------LOADING VAL!--------------------")
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_real_val.json'))
            # annotations.update(json.load(
            #     open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_syn_val.json')))
        elif subset == 'test':
            annotations = json.load(
                open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_real_test.json'))
            annotations.update(json.load(
                open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_syn_test.json')))

        annotations = list(annotations.values())
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            image_path = os.path.join(dataset_dir, a['filename'])
            # =============== print file names ===============
            print(image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Affordance",
                image_id=a['filename'],  # use file name as a unique image id
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
