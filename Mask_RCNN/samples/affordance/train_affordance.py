"""
Mask R-CNN
Train on the Warehouse dataset and implement color splash effect.

Based on the work of Waleed Abdulla (Matterport)
Modified by Dinh-Cuong Hoang

------------------------------------------------------------

python3 train.py --dataset=/Warehouse_Dataset/data --weights=coco
python3 train.py --dataset=/Warehouse_Dataset/data --weights=last

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from imgaug import augmenters as iaa

# ========= load dataset (optional) =========
import setup_dataset_affordance as Affordance

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# ================ warnings ==================
# import warnings
# warnings.filterwarnings("ignore")

# ##########################################################
# # Configurations
# ###########################################################
#
# class AffordanceConfig(Config):
#     """Configuration for training on the toy  dataset.
#     # Derives from the base Config class and overrides some values.
#     # """
#     # Give the configuration a recognizable name
#     NAME = "Affordance"
#
#     # ========== GPU config ================
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     # We use a GPU with 12GB memory, which can fit two images.
#     # Adjust down if you use a smaller GPU.
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 2
#     bs = GPU_COUNT * IMAGES_PER_GPU
#
#     # ===== combined ======
#     # Images:  /data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/combined_train_15k/*_rgb.png
#     # Loaded Images:  15891
#     # ---------stats---------------
#     # Means:
#     #  [[135.42236743]
#     #  [135.5095523 ]
#     #  [136.98013335]]
#     # STD:
#     #  [[27.17498643]
#     #  [27.91349685]
#     #  [28.16875631]]
#
#     # Images:  /data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/temp7_train_real/*_rgb.png
#     # Loaded Images:  891
#     # ---------stats---------------
#     # Means:
#     #  [[100.88058092]
#     #  [ 92.19536531]
#     #  [ 93.97393321]]
#     # STD:
#     #  [[23.66287158]
#     #  [32.95783879]
#     #  [36.65833118]]
#     MEAN_PIXEL = np.array([100.88058092, 92.19536531, 93.97393321])
#
#     BACKBONE = "resnet50"
#     RESNET_ARCHITECTURE = "resnet50"
#
#     IMAGE_MAX_DIM = 640
#     IMAGE_MIN_DIM = 480
#     IMAGE_PADDING = True
#
#     IMAGE_RESIZE_MODE = "none"
#
#     # Number of classes (including background)
#     NUM_CLASSES = 1 + 2  # Background + objects
#
#     # Number of training steps per epoch
#     # batch_size = 19773
#     # train_split = 15818 # 80 %
#     # STEPS_PER_EPOCH = (15000 + 890) // bs
#     # VALIDATION_STEPS = (3750 + 89) // bs
#     STEPS_PER_EPOCH = (890) // bs
#     VALIDATION_STEPS = (89) // bs
#
#     # TRAIN_BN = True # default is true
#     # Skip detections with < 90% confidence
#     DETECTION_MIN_CONFIDENCE = 0.7


###########################################################
# train
###########################################################

############################################################
#  Training
############################################################

def train(model):

    """Train the model."""
    # Training dataset.
    dataset_train = Affordance.AffordanceDataset()
    dataset_train.load_Affordance(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Affordance.AffordanceDataset()
    dataset_val.load_Affordance(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                # augmentation=augmentation,
                layers="heads")

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=80,
    #             layers="head",
    #             augmentation=imgaug.augmenters.OneOf([
    #                 imgaug.augmenters.Fliplr(0.5),
    #                 imgaug.augmenters.Flipud(0.5),
    #                 imgaug.augmenters.Affine(rotate=(-90, 90))
    #             ]))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Affordance.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Affordance/dataset/",
                        help='Directory of the Affordance dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    assert args.dataset, "Argument --dataset is required for training"
  
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = Affordance.AffordanceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # ========== MODEL SUMMARY =========
    model.keras_model.summary()

    # Train
    train(model)