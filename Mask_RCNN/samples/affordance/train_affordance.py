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

import imgaug

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

##########################################################
# Configurations
###########################################################

class PringlesConfig(Config):
    """Configuration for training on the toy  dataset.
    # Derives from the base Config class and overrides some values.
    # """
    # Give the configuration a recognizable name
    NAME = "Pringles"

    # ========== GPU config ================
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    bs = GPU_COUNT * IMAGES_PER_GPU

    # ===== dataset ======
    # Images:  /data/Akeaveny/Datasets/part-affordance-dataset/bowl/*.jpg
    # Loaded Images:  90
    # ---------stats---------------
    # Means:
    #  [[97.77970014]
    #  [88.01273163]
    #  [90.25397264]]
    # STD:
    #  [[24.25474554]
    #  [34.13697498]
    #  [38.37487672]]
    MEAN_PIXEL = np.array([97.77970014, 88.01273163, 90.25397264])
    RESNET_ARCHITECTURE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 88 // bs
    VALIDATION_STEPS = 88 // bs

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

###########################################################
# Dataset
###########################################################

class PringlesDataset(utils.Dataset):

    def load_Pringles(self, dataset_dir, subset):
        """Load a subset of the Pringles dataset.
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
        self.add_class("Pringles", 1, "grasp")
        self.add_class("Pringles", 2, "cut")
        self.add_class("Pringles", 3, "scoop")
        self.add_class("Pringles", 4, "contain")
        self.add_class("Pringles", 5, "pound")
        self.add_class("Pringles", 6, "support")
        self.add_class("Pringles", 7, "wrap-grasp")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        # /data/Akeaveny/Datasets/pringles/Alex/train/via_region_data.json
        if subset == 'train':
            print("------------------TRAIN------------------")
            annotations = json.load(open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_combined_train.json'))
        elif subset == 'val':
            print("------------------VAL--------------------")
            annotations = json.load(open('/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_combined_val.json'))

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
            print(image_path)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "Pringles",
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
        # If not a Pringles dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "Pringles":
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
        if info["source"] == "Pringles":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PringlesDataset()
    dataset_train.load_Pringles(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PringlesDataset()
    dataset_val.load_Pringles(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads',
                augmentation=imgaug.augmenters.OneOf([
                                imgaug.augmenters.Fliplr(0.5),
                                imgaug.augmenters.Flipud(0.5),
                                imgaug.augmenters.Affine(rotate=(-90, 90))
                                ]))

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=100,
    #             layers="all",
    #             augmentation=imgaug.augmenters.OneOf([
    #                 imgaug.augmenters.Fliplr(0.5),
    #                 imgaug.augmenters.Flipud(0.5),
    #                 imgaug.augmenters.Affine(rotate=(-90, 90))
    #             ]))

    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/100,
    #             epochs=1,
    #             # epochs=80,
    #             layers="all")

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Pringles.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/Pringles/dataset/",
                        help='Directory of the Pringles dataset')
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
    config = PringlesConfig()

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
    train(model)