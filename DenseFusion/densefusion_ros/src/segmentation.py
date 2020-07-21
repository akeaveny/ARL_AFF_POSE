#!/usr/bin/env python

import os
import sys
import time
import numpy as np

##################################
###  GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join((os.path.dirname(os.path.realpath(__file__))) ,"."))
sys.path.append(ROOT_DIR)

from mrcnn import matterport_dataset_syn as Affordance
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

import matplotlib.pyplot as plt

class MRCNNDetector(object):
    def __init__(self, model_path, num_classes, logs=None):

        class InferenceConfig(Affordance.AffordanceConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        if logs is None:
            log_dir = os.path.join(os.environ['HOME'], '.ros/logs')
            logs = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_dir)
            if not os.path.isdir(logs):
                os.mkdir(logs)

        self.__model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)
        self.__model.load_weights(model_path, by_name=True)
        self.__model.keras_model._make_predict_function()

        print("--- Successfully loaded MaskRCNN! ---\n")

    def detect_and_get_masks(self, image, visualize):

        # Detect objects
        cur_detect = self.__model.detect([image], verbose=0)[0]
        instance_masks = self.seq_get_masks(image, cur_detect)

        if visualize:
            plt.figure(1)
            plt.subplot(2, 1, 1)
            plt.title("rgb")
            plt.imshow(image)
            plt.subplot(2, 1, 2)
            plt.title("rgb")
            plt.imshow(instance_masks)
            plt.ioff()
            plt.pause(2)

        return instance_masks

    def seq_get_masks(self, image, cur_detection):

        cur_masks = cur_detection['masks']
        cur_class_ids = cur_detection['class_ids']
        cur_rois = cur_detection['rois']

        instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        print("cur_class_ids", cur_class_ids)
        if cur_class_ids.shape[-1] > 0:

            for i in range(cur_masks.shape[-1]):
                print("cur_class_ids[i]", cur_class_ids[i])
                ### instance_mask = instance_mask_one * (mask_index+1)
                instance_mask = instance_mask_one * cur_class_ids[i]
                instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

            return instance_masks
        else:
            print("--- Mask-RCNN detected no objects on the scene ---")
            return instance_masks

    def map_affordance_label(self, current_id):

        # 1
        grasp = [
            20,  # 'hammer-grasp'
            22,
            24,
            26,
            28,  # 'knife-grasp'
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,  # 'ladle-grasp'
            54,
            56,
            58,  # 'mallet-grasp'
            60,
            62,
            64,
            66,  # 'mug-grasp'
            69,
            72,
            75,
            78,
            81,
            84,
            87,
            90,
            93,
            96,
            99,
            102,
            105,
            108,
            111,
            114,
            117,
            120,
            123,
            130,  # 'saw-grasp'
            132,
            134,
            136,  # 'scissors-grasp'
            138,
            140,
            142,
            144,
            146,
            148,
            150,
            152,  # 'scoop-grasp'
            154,
            156,  # 'shears-grasp'
            158,
            160,  # 'shovel-grasp'
            162,
            164,  # 'spoon-grasp'
            166,
            168,
            170,
            172,
            174,
            176,
            178,
            180,
            182,
            184,  # 'tenderizer-grasp'
            186,  # 'trowel-grasp'
            188,
            190,
            192,  # 'turner-grasp'
            194,
            196,
            198,
            200,
            202,
            204,
        ]

        # 2
        cut = [
            28 + 1,  # "knife-cut"
            30 + 1,
            32 + 1,
            34 + 1,
            36 + 1,
            38 + 1,
            40 + 1,
            42 + 1,
            44 + 1,
            46 + 1,
            48 + 1,
            50 + 1,
            130 + 1,  # "saw-cut"
            132 + 1,
            134 + 1,
            136 + 1,  # "scissors-cut"
            138 + 1,
            140 + 1,
            142 + 1,
            144 + 1,
            146 + 1,
            148 + 1,
            150 + 1,
            156 + 1,  # "shears-cut"
            158 + 1,
        ]

        # 3
        scoop = [
            152 + 1,  # "scoop-scoop"
            154 + 1,
            160 + 1,  # "shovel-scoop"
            162 + 1,
            164 + 1,  # "spoon-scoop"
            166 + 1,
            168 + 1,
            170 + 1,
            172 + 1,
            174 + 1,
            176 + 1,
            178 + 1,
            180 + 1,
            182 + 1,
            186 + 1,  # "trowel-scoop"
            188 + 1,
            190 + 1,
        ]

        # 4
        contain = [
            1,  # "bowl-contain"
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            10,  # "cup-contain"
            12,
            14,
            16,
            18,
            52 + 1,  # "ladle-contain"
            54 + 1,
            56 + 1,
            66 + 1,  # "mug-contain"
            69 + 1,
            72 + 1,
            75 + 1,
            78 + 1,
            81 + 1,
            84 + 1,
            87 + 1,
            90 + 1,
            93 + 1,
            96 + 1,
            99 + 1,
            102 + 1,
            105 + 1,
            108 + 1,
            111 + 1,
            114 + 1,
            117 + 1,
            120 + 1,
            123 + 1,
            126,  # "pot-contain"
            128,
        ]

        # 5
        pound = [
            20 + 1,  # "hammer-pound"
            22 + 1,  # "hammer-pound"
            24 + 1,  # "hammer-pound"
            26 + 1,  # "hammer-pound"
            58 + 1,  # 'mallet-pound'
            60 + 1,  # 'mallet-pound'
            62 + 1,  # 'mallet-pound'
            64 + 1,  # 'mallet-pound'
            184 + 1,  # 'tenderizer-pound'
        ]

        # 6
        support = [
            192 + 1,  # "turner-support"
            194 + 1,
            196 + 1,
            198 + 1,
            200 + 1,
            202 + 1,
            204 + 1,
        ]

        # 7
        wrap_grasp = [
            8 + 1,  # "cup-wrap_grasp"
            10 + 1,  # "cup-wrap_grasp"
            12 + 1,  # "cup-wrap_grasp"
            14 + 1,  # "cup-wrap_grasp"
            16 + 1,  # "cup-wrap_grasp"
            18 + 1,  # "cup-wrap_grasp"
            66 + 2,  # "mug-wrap_grasp"
            69 + 2,  # "mug-wrap_grasp"
            72 + 2,  # "mug-wrap_grasp"
            75 + 2,  # "mug-wrap_grasp"
            78 + 2,  # "mug-wrap_grasp"
            81 + 2,  # "mug-wrap_grasp"
            84 + 2,  # "mug-wrap_grasp"
            87 + 2,  # "mug-wrap_grasp"
            90 + 2,  # "mug-wrap_grasp"
            93 + 2,  # "mug-wrap_grasp"
            96 + 2,  # "mug-wrap_grasp"
            99 + 2,  # "mug-wrap_grasp"
            102 + 2,  # "mug-wrap_grasp"
            105 + 2,  # "mug-wrap_grasp"
            108 + 2,  # "mug-wrap_grasp"
            111 + 2,  # "mug-wrap_grasp"
            114 + 2,  # "mug-wrap_grasp"
            117 + 2,  # "mug-wrap_grasp"
            120 + 2,  # "mug-wrap_grasp"
            123 + 2,  # "mug-wrap_grasp"
            126 + 1,  # "pot-wrap_grasp"
            128 + 1,  # "pot-wrap_grasp"
        ]

        if current_id in grasp:
            return 1
        elif current_id in cut:
            return 2
        elif current_id in scoop:
            return 3
        elif current_id in contain:
            return 4
        elif current_id in pound:
            return 5
        elif current_id in support:
            return 6
        elif current_id in wrap_grasp:
            return 7
        else:
            print(" --- Object ID does not map to Affordance Label --- ")
            print(current_id)
            exit(1)
