#!/usr/bin/env python
#!/usr/bin/env python
'''
This ros node subscribes to two camera topics: '/camera/color/image_raw' and 
'/camera/aligned_depth_to_color/image_raw' in a synchronized way. It then runs 
semantic segmentation and pose estimation with trained models using DenseFusion
(https://github.com/j96w/DenseFusion). The whole code structure is adapted from: 
(http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)
'''

import os
import sys
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import time
import PIL
import rospy
import random
import copy
import argparse
import numpy as np
import numpy.ma as ma
import message_filters
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import scipy.io as scio

from estimator import DenseFusionEstimator
from segmentation import MRCNNDetector

# # ========== GPU config ================
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class ObjectDetector(MRCNNDetector):

    def __init__(self):

        '''--- ROS PARAM --- '''
        self.__model = rospy.get_param('~model', None)
        self.__class_labels = rospy.get_param('~class_labels', None)
        self.__prob_thresh = rospy.get_param('~detection_threshold', 0.5)

        # assert os.path.isfile(self.__model), 'Trained model file not found! {}'.format(self.__model)
        # assert os.path.isfile(self.__class_labels), 'Class labels file not found! {}'.format(self.__class_labels)

        # ! initialize the detector
        self.__detector_init = True

        # ! read the object class name and labels
        lines = [line.rstrip('\n') for line in open(self.__class_labels)]

        self.__objects_meta = {}
        self.__labels = ['_']
        for line in lines:
            object_name, object_label = line.split()
            object_label = int(object_label)
            # ! color is for visualization
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            self.__objects_meta[object_label] = [object_name, color]
            self.__labels.append(object_name)

class PoseEstimator(DenseFusionEstimator):

    def __init__(self):

        '''--- ROS PARAM --- '''
        # Densfusion
        self.__pose_model = rospy.get_param('~pose_model', None)
        self.__refine_model = rospy.get_param('~refine_model', None)

        self.__num_points = rospy.get_param('~num_points', None)
        self.__num_points_mesh = rospy.get_param('~num_points_mesh', None)
        self.__iteration = rospy.get_param('~iteration', None)
        self.__bs = rospy.get_param('~bs', None)
        self.__num_obj = rospy.get_param('~num_obj', None)

        # 3D Models
        self.__classes = rospy.get_param('~classes', None)
        self.__class_ids = rospy.get_param('~class_ids', None)

        # ZED Camera
        self.__rgb_image = rospy.get_param('~rgb_image', None)
        self.__rgb_encoding = rospy.get_param('~rgb_encoding', None)
        self.__depth_image = rospy.get_param('~depth_image', None)
        self.__depth_encoding = rospy.get_param('~depth_encoding', None)

        self.__cam_width = rospy.get_param('~cam_width', None)
        self.__cam_height = rospy.get_param('~cam_height', None)
        self.__cam_scale = rospy.get_param('~cam_scale', None)
        self.__cam_fx = rospy.get_param('~cam_fx', None)
        self.__cam_fy = rospy.get_param('~cam_fy', None)
        self.__cam_cx = rospy.get_param('~cam_cx', None)
        self.__cam_cy = rospy.get_param('~cam_cy', None)

        # TESTING
        self.__DEBUG = rospy.get_param('~debug', None)
        self.__VISUALIZE = rospy.get_param('~visualize', None)
        self.__USE_SYNTHETIC_IMAGES = rospy.get_param('~use_synthetic_images', None)
        self.__CHECK_POSE = rospy.get_param('~check_pose', None)

        """ --- Init DenseFusion --- """
        DenseFusionEstimator.__init__(self, self.__pose_model, self.__refine_model,
                                      self.__num_points, self.__num_points_mesh, self.__iteration, self.__bs, self.__num_obj,
                                      self.__classes, self.__class_ids,
                                      self.__cam_width, self.__cam_height, self.__cam_scale,
                                      self.__cam_fx, self.__cam_fy, self.__cam_cx, self.__cam_cy)

        """ --- Subsribe to Camera --- """
        # RGB + Depth Images
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber(self.__rgb_image, Image)
        self.depth_sub = message_filters.Subscriber(self.__depth_image, Image)
        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 1)
        ts.registerCallback(self.camera_callback)

        if self.__USE_SYNTHETIC_IMAGES:
            print('\n--- Using Synthetic Images! ---')
        else:
            print('--- Subscribed to rgb and depth topic in a sychronized way! ---')

        """ --- MaskRCNN --- """
        # TODO: RGB Publisher
        # self.mask_sub = message_filters.Subscriber('/object_detector/mask', Image)
        self.rgb_pub = rospy.Publisher("/pose_estimation/rgb", Image, queue_size=1)

    def camera_callback(self, rgb_msg, depth_msg):

        """ === Load Images with ROS === """
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, self.__rgb_encoding)
        rgb_cv = self.bridge.cv2_to_imgmsg(rgb_cv, self.__rgb_encoding)
        bgr = np.frombuffer(rgb_cv.data, dtype=np.uint8).reshape(rgb_cv.height, rgb_cv.width, -1)
        rgb = np.array(cv.cvtColor(bgr, cv.COLOR_BGR2RGB))

        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, self.__depth_encoding) # "16UC1" or "32FC1"
        depth_cv = np.float32(depth_cv)
        depth = depth_cv
        # depth_cv[np.isnan(depth_cv)] = 0 # TODO: In-painting
        # depth_cv[depth_cv == -np.inf] = 0
        # depth_cv[depth_cv == np.inf] = 0
        # depth = depth_cv.reshape(depth.height, depth.width, -1)

        if self.__USE_SYNTHETIC_IMAGES:

            data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/'
            # ================== NDDS =========================
            train_images_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/train_data_list.txt'
            test_images_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/test_data_list.txt'
            loaded_images_ = np.loadtxt(test_images_file, dtype=np.str)
            # for idx in range(len(loaded_images_)):

            idx = np.random.choice(len(loaded_images_), size=1, replace=False)[0]

            rgb_addr = data_path + loaded_images_[idx] + "_rgb.png"
            depth_addr = data_path + loaded_images_[idx] + "_depth.png"
            gt_addr = data_path + loaded_images_[idx] + "_label.png"

            if self.__CHECK_POSE:
                meta_addr = data_path + loaded_images_[idx] + "-meta.mat"
                meta = scio.loadmat(meta_addr)

            rgb = np.array(PIL.Image.open(rgb_addr))
            depth = np.array(PIL.Image.open(depth_addr))
            gt = np.array(PIL.Image.open(gt_addr))

            # ## ============== SYNTHETIC ===================
            rgb = np.array(rgb)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]

        """ === Mask R-CNN === """
        self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(rgb, self.__rgb_encoding))
        # self.mask_sub = rospy.Subscriber('/object_detector/mask', Image)

        if self.__USE_SYNTHETIC_IMAGES and self.__CHECK_POSE:
            rospy.Subscriber('/object_detector/mask', Image, self.segmentation_callback, (rgb, depth, gt, meta))
        else:
            rospy.Subscriber('/object_detector/mask', Image, self.segmentation_callback, (rgb, depth))

        rospy.sleep(10)

    def segmentation_callback(self, mask_msg, args):
        print("Grabbed Mask")

        if self.__USE_SYNTHETIC_IMAGES and self.__CHECK_POSE:
            rgb = args[0]
            depth = args[1]
            gt = args[2]
            meta = args[3]
        else:
            rgb = args[0]
            depth = args[1]

        mask = np.frombuffer(mask_msg.data, dtype=np.uint8).reshape(mask_msg.height, mask_msg.width)
        mask = np.array(mask)

        t_start = time.time()
        if self.__USE_SYNTHETIC_IMAGES and self.__CHECK_POSE:
            DenseFusionEstimator.get_refined_pose(self, rgb, depth, gt, meta,
                                                  self.__DEBUG, self.__VISUALIZE, self.__CHECK_POSE)
        else:
            DenseFusionEstimator.get_refined_pose(self, rgb, depth, mask, self.__DEBUG, self.__VISUALIZE)
        t_mask_rcnn = time.time() - t_start
        print('DenseFusion Prediction time: {:.2f}s\n'.format(t_mask_rcnn))
        return

def main(args):

    rospy.init_node('pose_estimation', anonymous=True)
    PoseEstimator()
    ObjectDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down ROS pose estimation module')
    cv.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)