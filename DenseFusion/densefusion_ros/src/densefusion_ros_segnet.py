#!/usr/bin/env python
'''
This ros node subscribes to two camera topics: '/camera/color/image_raw' and 
'/camera/aligned_depth_to_color/image_raw' in a synchronized way. It then runs 
semantic segmentation and pose estimation with trained models using DenseFusion
(https://github.com/j96w/DenseFusion). The whole code structure is adapted from: 
(http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber)
'''
import sys
# sys.path.insert(0,'/home/fapsros/anaconda3/lib/python3.7/site-packages') # add this line if you encounter "undefined symbol: PyCObject_Type" trigged by import cv2

import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import rospy
import copy
import argparse                                                             
import numpy as np
import numpy.ma as ma 
import message_filters
from sensor_msgs.msg import Image

import torch
import torchvision
from torchvision import transforms
from torch.autograd import Variable

from vanilla_segmentation.segnet import SegNet as segnet
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
knn = KNearestNeighbor(1)

import matplotlib
import matplotlib.pyplot as plt

# ========== GPU config ================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# FLAGS
DEBUG = True
SHOW_IMAGE = True
SHOW_BBOX = False
PRINT_RESULTS = False
np.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '/data/Akeaveny/Datasets/YCB_Video_Dataset', help='dataset root dir')
parser.add_argument('--model', type=str, default = './trained_models/ycb/pose_model_current.pth',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = './trained_models/ycb/pose_refine_model_current.pth',  help='resume PoseRefineNet model')
parser.add_argument('--seg_model', type=str, default = './vanilla_segmentation/trained_models/model_current.pth',  help='resume SegNet model')
parser.add_argument('--save_processed_image', type=bool, default=False,  help='Save image with model points')
opt = parser.parse_args()

def get_bbox(bbox):
    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax

class pose_estimation:

    def __init__(self, model_, estimator_, refiner_):
        self.bridge = CvBridge()
        self.rgb_sub = message_filters.Subscriber('/zed/zed_node/left/image_rect_color', Image)
        self.depth_sub = message_filters.Subscriber('/zed/zed_node/depth/depth_registered', Image)
        ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 15)
        ts.registerCallback(self.callback)
        self.model = model_
        self.estimator = estimator_
        self.refiner = refiner_

        self.cam_scale = 1000

        self.height = 720
        self.width = 1280
        # data: [698.2383422851562, 0.0, 619.1629638671875, 0.0, 698.2383422851562, 354.4393615722656, 0.0, 0.0, 1.0]
        self.cam_fx = 698.2383422851562
        self.cam_cx = 619.1629638671875
        self.cam_fy = 698.2383422851562
        self.cam_cy = 354.4393615722656
        self.xmap = np.array([[j for i in range(self.width)] for j in range(self.height)])
        self.ymap = np.array([[i for i in range(self.width)] for j in range(self.height)])

        if DEBUG:
            print('subscribed to rgb and depth topic in a sychronized way')

    def callback(self, rgb, depth):

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

        num_points = 1000
        num_points_mesh = 500
        iteration = 2
        bs = 1

        """===============================Loaded Images with ROS==========================="""
        # "16UC1" or "32FC1"
        depth_cv = self.bridge.imgmsg_to_cv2(depth, "32FC1")
        depth_cv = np.float32(depth_cv)
        depth_cv[np.isnan(depth_cv)] = 0
        depth_cv[depth_cv == -np.inf] = 0
        depth_cv[depth_cv == np.inf] = 0
        rgb_cv = self.bridge.imgmsg_to_cv2(rgb, "bgr8")
        rgb = self.bridge.cv2_to_imgmsg(rgb_cv, "bgr8")

        #https://answers.ros.org/question/64318/how-do-i-convert-an-ros-image-into-a-numpy-array/
        # depth = depth_cv.reshape(depth.height, depth.width, -1)
        depth = depth_cv
        bgr = np.frombuffer(rgb.data, dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        # rgb = bgr
        rgb = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        """===============================Downsampling ROS Images==========================="""
        depth = depth[50:530, 100:740]
        rgb = rgb[50:530, 100:740, :] # height x width

        print("Depth: ", depth.shape)
        print("RGB: ", rgb.shape)

        # if DEBUG:
        #     print ('received depth image of type: ' +depth.encoding)
        #     print ('received rgb image of type: ' + rgb.encoding)

        if SHOW_IMAGE:
            # print("----------depth.cv----------------")
            # print(depth_cv.shape)
            # print(depth_cv)
            # cv2.imshow("Depth window", depth_cv)
            # cv2.waitKey(1)

            plt.subplot(2, 2, 1)
            plt.imshow(depth)

        if SHOW_IMAGE:
            # print("----------rbg----------------")
            # cv2.imshow('RGB', rgb)
            cv2.imshow('RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            plt.subplot(2, 2, 2)
            # plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            plt.imshow(rgb)
            plt.ioff()
            plt.pause(0.0333)

        """============================ seg mask ====================================="""
        print("----------seg mask----------------")
        rgb1 = np.transpose(rgb, (2, 0, 1))
        rgb1 = norm(torch.from_numpy(rgb1.astype(np.float32))).cuda()
        seg_data = self.model(rgb1.unsqueeze(0))
        seg_data2 = torch.transpose(seg_data[0], 0, 2)
        seg_data2 = torch.transpose(seg_data2, 0, 1)
        seg_image = torch.argmax(seg_data2, dim=-1)
        obj_list = torch.unique(seg_image).detach().cpu().numpy()
        label_img = seg_image.detach().cpu().numpy()

        if SHOW_IMAGE:
            print("Objects: ", obj_list)

            cv2.imwrite('seg_mask.png', label_img)
            # cv2.imshow("seg_mask", label_img)
            # cv2.waitKey(1)

            plt.subplot(2, 2, 3)
            plt.imshow(label_img)

        """============================ seg obj ====================================="""
        rgb2 = np.transpose(rgb, (2, 0, 1))
        rgb2 = norm(torch.from_numpy(rgb2.astype(np.float32))).cuda()
        img_out = torch.nn.functional.softmax(self.model(rgb2.unsqueeze(0)), dim=1)
        img_out_2 = img_out.cpu().data.numpy()

        seg_obj = np.argmax(img_out_2[0, :, :, :], axis=0) == 9 # 4 = soup can and 9 = spam
        # cv2.imwrite("label_img.png", seg_obj.astype(np.uint8))

        if SHOW_IMAGE:
            cv2.imwrite("seg_obj.png", seg_obj.astype(np.uint8))
            # image = cv2.imread('seg_obj.png')
            # cv2.imshow("seg_obj", image)
            # cv2.waitKey(1)

            plt.subplot(2, 2, 4)
            plt.imshow(seg_obj.astype(np.uint8))
            plt.ioff()
            plt.pause(5)

        """ =================== BOUNDING BOX ================================"""
        # image = cv2.imread('label_img.png')
        # image_copy = image.copy()
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        #
        # ROI_number = 0
        # ROI = image[cmin:cmax, rmin:rmax]
        # # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        # cv2.rectangle(image_copy, (cmin, rmin), (rmax, cmax), (100, 36, 12), 4)
        # ROI_number += 1
        # for c in cnts:
        #     x, y, w, h = cv2.boundingRect(c)
        #     ROI = image[y:y + h, x:x + w]
        #     # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        #     cv2.rectangle(image_copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
        #     ROI_number += 1
        # cv2.imwrite("image_copy.png", image_copy.astype(np.uint8))
        # ax4[1, 0].imshow(image_copy.astype(np.uint8))

def main(args):

    num_obj = 21
    num_points = 1000
    seg_model = segnet()
    seg_model.cuda()
    seg_model.load_state_dict(torch.load('./vanilla_segmentation/trained_models/model_10_0.11537120866216719.pth'))
    seg_model.eval()

    estimator = PoseNet(num_points, num_obj)
    estimator.cuda()
    estimator.load_state_dict(torch.load('./trained_models/ycb/pose_model_22_0.012934861789403338.pth'))
    estimator.eval()

    refiner = PoseRefineNet(num_points, num_obj)
    refiner.cuda()
    refiner.load_state_dict(torch.load('./trained_models/ycb/pose_refine_model_29_0.012507753662475113.pth'))
    refiner.eval()

    pe = pose_estimation(seg_model, estimator, refiner)
    rospy.init_node('pose_estimation', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ('Shutting down ROS pose estimation module')
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

'''
how to resize the display window via trackbar:
https://answers.ros.org/question/257440/python-opencv-namedwindow-and-imshow-freeze/
'''
