#! /usr/bin/env python

import argparse
import os
import sys
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math

import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
print("ROOT_DIR", ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

from sklearn.neighbors import KDTree

#############
#
#############
def get_bbox(label, affordance_id, img_width, img_length, border_list):

    ####################
    ## affordance id
    ####################

    rows = np.any(label==affordance_id, axis=1)
    cols = np.any(label==affordance_id, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmax += 1
    cmax += 1
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
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

##################################
## GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Evaluate trained model for DenseFusion')

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/YCB_Video_Dataset/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_config', required=False, default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config',
                    type=str,
                    metavar="")
parser.add_argument('--ycb_toolbox_config', required=False, default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/YCB_Video_toolbox/results_PoseCNN_RSS2018/',
                    type=str,
                    metavar="")
parser.add_argument('--dataset_type', required=False, default='val',
                    type=str,
                    metavar='train or val')

parser.add_argument('--ADD', required=False, default=2,
                    type=int,
                    metavar='ADD distance to evaluate')

parser.add_argument('--train_file', required=False, default='train_data_list.txt',
                    metavar="/path/to/model weights")
parser.add_argument('--val_file', required=False, default='test_data_list.txt',
                    metavar="/path/to/refine weights")

parser.add_argument('--model', required=False, default=ROOT_DIR + '/trained_models/ycb_syn/ycb_syn2/pose_model_7_0.012854825005869814.pth',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--refine_model', required=False, default=ROOT_DIR + '/trained_models/ycb_syn/ycb_syn2/pose_refine_model_14_0.01016292095133344.pth',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--classes', required=False, default='classes_train.txt',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--class_ids', required=False, default='class_ids_train.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--output_result_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb_syn',
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--error_metrics_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/ycb_syn/Densefusion_error_metrics_result/',
                    type=str,
                    metavar="")
parser.add_argument('--visualize', required=False, default=True,
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--save_images_path', required=False, default='/data/Akeaveny/Datasets/ycb_syn/test_densefusion/',
                    type=str,
                    metavar="Visualize Results")

args = parser.parse_args()

num_obj = 31

num_points = 1000
num_points_mesh = 1000
iteration = 2
bs = 1

##################################
## classes
##################################

class_file = open('{}/{}'.format(args.dataset_config, args.classes))
class_id_file = open('{}/{}'.format(args.dataset_config, args.class_ids))
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("Classes: ", class_IDs)

##################################
## 3D MODELS
##################################

cld = {}
for idx, class_id in enumerate(class_IDs):
    class_input = class_file.readline()
    # print("class_id: ", class_id)
    # print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/ycb_syn/models/{0}/grasp_{0}.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    # print("class_id: ", class_id)
    # print("class_input: ", class_input.rstrip())
    # print("Num Point Clouds: {}\n".format(len(cld[class_id])))
    cld[class_id] = np.array(cld[class_id])  # TODO:
    input_file.close()

class_file = open('{}/{}'.format(args.dataset_config, args.classes))
classes_full = np.loadtxt(class_file, dtype=np.str)
# print("Classes: ", classes_full)

##################################
# DENSEFUSION
##################################

estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(args.model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(args.refine_model))
refiner.eval()

###############
# ycb dataset
###############

norm_ = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0

img_width = 480
img_length = 640

##################################
# LOAD ycb dataset
##################################

loaded_images_ = np.loadtxt('{}'.format('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/YCB_Video_toolbox/keyframe.txt'), dtype=np.str)

############
#
############
num_correct, num_total = 0, 0
fw = open('{0}/eval_result_logs.txt'.format(args.output_result_dir), 'w')

for image_idx in range(len(loaded_images_)):

    ##############
    ##############

    # print("Image Info: ", loaded_images_[idx].split('/'))
    str_num = loaded_images_[image_idx].split('/')[-1]

    rgb_addr = args.dataset + 'data/' + loaded_images_[image_idx] + "-color.png"
    depth_addr = args.dataset + 'data/' + loaded_images_[image_idx] + "-depth.png"
    gt_addr = args.dataset + 'data/' + loaded_images_[image_idx] + "-meta.mat"

    count = 1000000 + image_idx
    meta_idx = str(count)[1:]
    meta_addr = args.ycb_toolbox_config + meta_idx + '.mat'

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt_meta = scio.loadmat(gt_addr)
    posecnn_meta = scio.loadmat(meta_addr)

    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    gt_poses = np.array(posecnn_meta['poses_icp'])

    lst = np.array(posecnn_rois[:, 1:2].flatten(), dtype=int)

    gt_test_pose = np.array(gt_meta['poses']).flatten().reshape(3, 4, -1)
    gt_cls_indexes = np.array(gt_meta['cls_indexes'].flatten(), dtype=int)

    roi_idxs = []
    for value in lst:
        if value in gt_cls_indexes.tolist():
            roi_idxs.append(gt_cls_indexes.tolist().index(value))

    # if args.print_output:
    #     print("pred_cls_indexes: ", lst)
    #     print("gt_cls_indexes: ", gt_cls_indexes)
    #     print("gt_idx: ", roi_idxs)
    #     print("")

    if args.visualize:
        plt.subplot(2, 2, 1)
        plt.title("rgb")
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        plt.title("depth")
        plt.imshow(depth)
        plt.subplot(2, 2, 3)
        plt.title("gt")
        plt.imshow(label)
        plt.subplot(2, 2, 4)
        plt.title("label")
        plt.imshow(label)
        plt.show()
        plt.ioff()

