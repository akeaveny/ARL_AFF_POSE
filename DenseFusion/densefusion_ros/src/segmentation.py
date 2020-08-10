#!/usr/bin/env python
from __future__ import division

import os
import sys
import time
import numpy as np

import cv2 as cv

import skimage.draw

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

##################################
###  GPU
##################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# if tf.test.gpu_device_name():
#     print('************ GPU found ************')
# else:
#     print("************ No GPU found ************")
#
# config_ = tf.ConfigProto()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
# config_ = tf.ConfigProto(gpu_options=gpu_options)
# ## config_.gpu_options.allow_growth = True
# sess = tf.Session(config=config_)

ROOT_DIR = os.path.abspath(os.path.join((os.path.dirname(os.path.realpath(__file__))) ,"."))
sys.path.append(ROOT_DIR)

from mrcnn import dataset_syn as Affordance
from mrcnn.config import Config
from mrcnn import modeldepth as modellib
from mrcnn import utils as utils

class MRCNNDetector(object):
    def __init__(self, model_path, num_classes, logs=None):

        class InferenceConfig(Affordance.AffordanceConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        if logs is None:
            log_dir = os.path.join(os.environ['HOME'], '.ros/logs')
            logs = os.path.join(os.path.dirname(os.path.realpath(__file__)), log_dir)
            if not os.path.isdir(logs):
                os.mkdir(logs)

        self.__model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)
        self.__model.load_weights(model_path, by_name=True)
        self.__model.keras_model._make_predict_function()

        # with open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/densefusion_ros/mrcnndepth27.txt', 'w') as fh:
        #     # Pass the file handle in as a lambda function to make it callable
        #     self.__model.keras_model.summary(print_fn=lambda x: fh.write(x + '\n'))

        config.display()

        print("*** Successfully loaded MaskRCNN! ***")

    def detect_and_get_masks(self, image, depth, visualize):

        ##################################
        # RGB has 4th channel - alpha
        # depth to 3 channels
        ##################################
        image, depth = image[..., :3], skimage.color.gray2rgb(depth)

        # Detect objects
        # cur_detect = self.__model.detect([image], verbose=1)[0]
        cur_detect = self.__model.detectWdepth([image], [depth], verbose=0)[0]
        instance_masks = self.seq_get_masks(image, cur_detect)

        return instance_masks

    def seq_get_masks(self, image, cur_detection):

        cur_masks = cur_detection['masks']
        cur_class_ids = cur_detection['class_ids']
        cur_rois = cur_detection['rois']

        instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        ### print("cur_class_ids", cur_class_ids)
        if cur_class_ids.shape[-1] > 0:

            for i in range(cur_masks.shape[-1]):
                ### print("cur_class_ids[i]", cur_class_ids[i])
                ### instance_mask = instance_mask_one * (mask_index+1)
                instance_mask = instance_mask_one * cur_class_ids[i]
                instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

            return instance_masks
        else:
            print("--- Mask-RCNN detected no objects on the scene ---")
            return instance_masks
