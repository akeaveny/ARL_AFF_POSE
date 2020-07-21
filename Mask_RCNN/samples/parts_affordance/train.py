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
# import matplotlib.image as mpimg

from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

import argparse
############################################################
#  Parse command line arguments
############################################################
parser = argparse.ArgumentParser( description='Train Mask R-CNN to detect Affordance.')
parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/part-affordance_combined/ndds2/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_type', required=False, default='real',
                    type=str,
                    metavar='real or syn or syn1')
parser.add_argument('--weights', required=True,
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--logs', required=False,
                    default=DEFAULT_LOGS_DIR,
                    metavar="/path/to/logs/",
                    help='Logs and checkpoints directory (default=logs/)')
parser.add_argument('--display_keras', required=False, default=False,
                    type=str,
                    metavar='Display Keras Layers')
args = parser.parse_args()

############################################################
#  REAL OR SYN
############################################################
assert args.dataset_type == 'real' or args.dataset_type == 'syn' or args.dataset_type == 'syn1' or args.dataset_type == 'hammer'
if args.dataset_type == 'real':
    import dataset_real as Affordance
elif args.dataset_type == 'syn':
    import dataset_syn as Affordance
elif args.dataset_type == 'syn1':
    import dataset_syn1 as Affordance
elif args.dataset_type == 'hammer':
    import dataset_syn_hammer as Affordance

# ##################################
# ###  GPU
# ##################################
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
#
# import tensorflow as tf
# config_ = tf.ConfigProto()
# config_.gpu_options.allow_growth = True
# sess = tf.Session(config=config_)
# ### gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

from mrcnn.config import Config
from mrcnn import model as modellib, utils

############################################################
#  train
############################################################

def train(model, args):

    """Train the model."""
    # Training dataset.
    dataset_train = Affordance.AffordanceDataset()
    dataset_train.load_Affordance(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Affordance.AffordanceDataset()
    dataset_val.load_Affordance(args.dataset, "val")
    dataset_val.prepare()

    if args.display_keras:
        model.keras_model.summary()
    config.display()

    ##################
    #  IMMGAUG
    ##################

    augmentation = iaa.Sometimes(0.9, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # iaa.Multiply((0.8, 1.2)),
        # iaa.GaussianBlur(sigma=(0.0, 5.0)),
        # iaa.OneOf([iaa.Affine(rotate=90),
        #            iaa.Affine(rotate=180),
        #            iaa.Affine(rotate=270)]),
    ])
    # elif args.dataset_type == 'syn' or args.dataset_type == 'syn1':
    #    augmentation = None

    #############################
    #  Learning Rate Scheduler
    #############################

    # Training - Stage 1 HEADS
    # HEADS
    print("\n************* trainining HEADS *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                augmentation=augmentation,
                layers='heads')

    print("\n************* trainining HEADS *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=100,
                augmentation=augmentation,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    # print("\n************* trainining ResNET 4+ *************")
    # model.train(dataset_train, dataset_val,
    #           learning_rate=config.LEARNING_RATE/10,
    #           epochs=160,
    #           augmentation=augmentation,
    #           layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("\n************* trainining ALL *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10,
                epochs=170,
                augmentation=augmentation,
                layers='all')

    # print("\n************* trainining ALL LR:{}*************".format(config.LEARNING_RATE))
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=40,
    #             augmentation=augmentation,
    #             layers='all')
    # print("\n************* trainining ALL LR:{}*************".format(config.LEARNING_RATE/10))
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=80,
    #             augmentation=augmentation,
    #             layers="all")
    # print("\n************* trainining ALL LR:{}*************".format(config.LEARNING_RATE/100))
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 100,
    #             epochs=120,
    #             augmentation=augmentation,
    #             layers='all')

############################################################
#  Training
############################################################

if __name__ == '__main__':
  
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = Affordance.AffordanceConfig()

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
            "conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True, exclude=["conv1"])
        ### model.load_weights(weights_path, by_name=True) # TODO: Change load image

    # Train
    train(model, args)