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

parser.add_argument('--train', required=False, default='rgbd',
                    type=str,
                    metavar="Train RGB or RGB+D")

parser.add_argument('--dataset', required=False,
                    default='/data/Akeaveny/Datasets/part-affordance_combined/real/',
                    # default='/data/Akeaveny/Datasets/part-affordance_combined/ndds4/',
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

if args.dataset_type == 'real':
    import dataset_real as Affordance
elif args.dataset_type == 'syn':
    import dataset_syn as Affordance
elif args.dataset_type == 'syn1':
    import dataset_syn1 as Affordance
elif args.dataset_type == 'hammer':
    import objects.dataset_syn_hammer as Affordance
elif args.dataset_type == 'hammer1':
    import objects.dataset_syn_hammer1 as Affordance
elif args.dataset_type == 'scissors_real':
    import objects.dataset_real_scissors as Affordance
elif args.dataset_type == 'scissors':
    import objects.dataset_syn_scissors as Affordance
elif args.dataset_type == 'scissors_20k':
    import objects.dataset_syn_scissors_20k as Affordance

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
if args.train == 'rgb':
    from mrcnn import model as modellib, utils
elif args.train == 'rgbd':
    from mrcnn import modeldepth as modellib, utils
elif args.train == 'rgbd+':
    from mrcnn import modeldepthv2 as modellib, utils
else:
    print("*** No Model Selected ***")
    exit(1)

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
    augmentation = iaa.Sometimes(0.833, iaa.Sequential([
        #########################
        # COLOR & MASK
        #########################
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),
        iaa.Crop(percent=(0, 0.1)),  # random crops
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # rotate=(-25, 25),
            # shear=(-8, 8)
        ),
        #########################
        # ONLY COLOR !!!
        #########################
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.5))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.25)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
    ], random_order=True))  # apply augmenters in random order

    #############################
    #  Learning Rate Scheduler
    #############################

    # ### Training - Stage 1 HEADS
    # ### HEADS
    # print("\n************* trainining HEADS *************")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=10,
    #             augmentation=augmentation,
    #             layers='heads')
    #
    # ### Training - Stage 2a
    # ### Finetune layers from ResNet stage 4 and up
    # print("\n************* trainining ResNET 4+ *************")
    # model.train(dataset_train, dataset_val,
    #           learning_rate=config.LEARNING_RATE/10,
    #           epochs=15,
    #           augmentation=augmentation,
    #           layers='4+')
    #
    # ### Training - Stage 3
    # ### Fine tune all layers
    # print("\n************* trainining ALL *************")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/100,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='all')

    # ########################
    # # Finetuning
    # ########################
    START = 20
    ### Training - Stage 1 HEADS
    ## Training - Stage 1 HEADS
    # HEADS
    print("\n************* trainining HEADS *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=START + 40,
                augmentation=augmentation,
                layers='heads')

    ### Training - Stage 2a
    ### Finetune layers from ResNet stage 4 and up
    print("\n************* trainining ResNET 4+ *************")
    model.train(dataset_train, dataset_val,
              learning_rate=config.LEARNING_RATE/10,
              epochs=START + 50,
              augmentation=augmentation,
              layers='4+')

    ### Training - Stage 3
    ### Fine tune all layers
    print("\n************* trainining ALL *************")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/100,
                epochs=START + 60,
                augmentation=augmentation,
                layers='all')

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
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    train(model, args)
