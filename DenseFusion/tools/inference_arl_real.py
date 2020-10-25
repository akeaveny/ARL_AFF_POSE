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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Evaluate trained model for DenseFusion')

parser.add_argument('--dataset', required=False, default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/',
                    type=str,
                    metavar="/path/to/Affordance/dataset/")
parser.add_argument('--dataset_config', required=False, default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config',
                    type=str,
                    metavar="")
parser.add_argument('--dataset_type', required=False, default='test',
                    type=str,
                    metavar='train or val')

parser.add_argument('--ADD', required=False, default=2,
                    type=int,
                    metavar='ADD distance to evaluate')

parser.add_argument('--test_file', required=False, default='test_data_list.txt',
                    metavar="/path/to/refine weights")

parser.add_argument('--model', required=False, default=ROOT_DIR + '/trained_models/arl_real/arl1/pose_model_9_0.012771212236549688.pth',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--refine_model', required=False, default=ROOT_DIR + '/trained_models/arl_real/arl1/pose_refine_model_19_0.011534377784132781.pth',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--classes', required=False, default='classes.txt',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--class_ids', required=False, default='class_ids.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--classes_train', required=False, default='classes_train.txt',
                    metavar="/path/to/weights.h5 or 'coco'")
parser.add_argument('--class_ids_train', required=False, default='class_ids_train.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--output_result_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/arl_real',
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--error_metrics_dir', required=False, default=ROOT_DIR + '/experiments/eval_result/arl_real/Densefusion_error_metrics_result/',
                    type=str,
                    metavar="")

parser.add_argument('--visualize', required=False, default=False,
                    type=str,
                    metavar="Visualize Results")
parser.add_argument('--save_images_path', required=False, default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/test_densefusion_real/',
                    type=str,
                    metavar="Visualize Results")

args = parser.parse_args()

num_obj = 4

num_points = 1000
num_points_mesh = 1000
iteration = 5
bs = 1

norm_ = transforms.Normalize(mean=[112.83933655/255, 108.60317768/255, 100.21422303/255],
                                         std=[59.73553193/255, 63.61142929/255, 70.06586743/255])

##################################
## classes
##################################

class_file = open('{}/{}'.format(args.dataset_config, args.classes))
class_id_file = open('{}/{}'.format(args.dataset_config, args.class_ids))
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("Classes: ", class_IDs)

class_file_train = open('{}/{}'.format(args.dataset_config, args.classes_train))
class_id_file_train = open('{}/{}'.format(args.dataset_config, args.class_ids_train))
class_IDs_train = np.loadtxt(class_id_file_train, dtype=np.int32)

##################################
## 3D MODELS
##################################

cld = {}
for idx, class_id in enumerate(class_IDs_train):
    class_input = class_file_train.readline()
    # print("class_id: ", class_id)
    # print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/arl_scanned_objects/ARL/models/{0}/{0}.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    cld[class_id] = np.array(cld[class_id])
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

####################
# camera_setting
####################
cam_scale = 1000

width, height = 672, 376
cam_cx = 326.785
cam_cy = 168.432
cam_fx = 338.546630859375
cam_fy = 338.546630859375

xmap = np.array([[j for i in range(width)] for j in range(height)])
ymap = np.array([[i for i in range(width)] for j in range(height)])

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

##################################
## LOAD IMAGES
##################################

loaded_images_ = np.loadtxt('{}/{}'.format(args.dataset_config, args.test_file), dtype=np.str)

num_correct, num_total = 0, 0
fw = open('{0}/eval_result_logs.txt'.format(args.output_result_dir), 'w')

# select random test images
np.random.seed(0)
num_test = 100
test_idx = np.random.choice(np.arange(0, int(len(loaded_images_)), 1), size=int(num_test), replace=False)
print("Chosen Files: ", len(test_idx))

for run_idx, idx in enumerate(test_idx):

    ##############
    # NDDS
    ##############

    # print("Image Info: ", loaded_images_[idx].split('/'))
    str_num = loaded_images_[idx].split('/')[-1]

    rgb_addr = args.dataset + loaded_images_[idx] + "_rgb.png"
    depth_addr = args.dataset + loaded_images_[idx] + "_depth.png"
    gt_addr = args.dataset + loaded_images_[idx] + "_label.png"
    mask_addr = gt_addr

    # gt pose
    meta_addr = args.dataset + loaded_images_[idx] + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    # rgb
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

    # if args.visualize:
    #     plt.subplot(2, 2, 1)
    #     plt.title("rgb")
    #     plt.imshow(img)
    #     plt.subplot(2, 2, 2)
    #     plt.title("depth")
    #     plt.imshow(depth)
    #     plt.subplot(2, 2, 3)
    #     plt.title("gt")
    #     plt.imshow(gt)
    #     plt.subplot(2, 2, 4)
    #     plt.title("label")
    #     plt.imshow(label)
    #     plt.show()
    #     plt.ioff()

    ####################
    # affordance_ids
    ####################

    Class_id_list, ADD_list, ADD_S_list, R_list, T_list = [], [], [], [], []
    my_result_wo_refine = []
    my_result = []

    affordance_ids = np.unique(label)
    for affordance_id in affordance_ids:
        if affordance_id in class_IDs_train:

            itemid = affordance_id
            meta_idx = str(affordance_id)

            print("Affordance ID: ", itemid)

            ##############
            # bbox
            ##############

            rmin, rmax, cmin, cmax = get_bbox(label, itemid, width, height, border_list)
            # print("\nbbox: ", rmin, rmax, cmin, cmax)

            # disp
            img_bbox = np.array(img.copy())
            img_name = args.save_images_path + 'test1.bbox.png'
            cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
            cv2.imwrite(img_name, img_bbox)
            bbox_image = cv2.imread(img_name)
            bbox_image = np.array(bbox_image, dtype=np.uint8)

            ##############
            ##############

            try:
                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
                mask = mask_label * mask_depth
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                ##############
                # cloud
                ##############

                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32))
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm_(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                ##############
                ##############

                pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

                pred_c = pred_c.view(bs, num_points)
                how_max, which_max = torch.max(pred_c, 1)
                ### print("how_max: {:.5f}".format(how_max.detach().cpu().numpy()[0]))

                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())
                ### print("my_pred w/o refinement: \n", my_pred)

                for ite in range(0, iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = refiner(new_cloud, emb, index)
                    pred_r = pred_r.view(1, 1, -1)
                    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                    my_r_2 = pred_r.view(-1).cpu().data.numpy()
                    my_t_2 = pred_t.view(-1).cpu().data.numpy()
                    my_mat_2 = quaternion_matrix(my_r_2)

                    my_mat_2[0:3, 3] = my_t_2

                    my_mat_final = np.dot(my_mat, my_mat_2)
                    my_r_final = copy.deepcopy(my_mat_final)
                    my_r_final[0:3, 3] = 0
                    my_r_final = quaternion_from_matrix(my_r_final, True)
                    my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                    my_pred = np.append(my_r_final, my_t_final)
                    my_r = my_r_final
                    my_t = my_t_final
                    ### print("my_pred w/ {} refinement: \n{}".format(ite, my_pred))

                # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

                my_result.append(my_pred.tolist())

                ############################
                # project to screen
                ############################

                cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

                ############################
                # pred
                ############################

                mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                my_t_ = my_t

                rgb_img = Image.open(rgb_addr)
                imgpts, jac = cv2.projectPoints(cld[itemid] * 1e3,
                                                mat_r,
                                                my_t_ * 1e3,
                                                cam_mat, dist)
                cv2_img = cv2.polylines(np.array(rgb_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                img_name = args.save_images_path + 'test1.cv2.cloud.png'
                cv2.imwrite(img_name, cv2_img)
                cld_image = cv2.imread(img_name)

                ############################
                # gt
                ############################

                gt_rot1 = np.array(meta['rot' + meta_idx])
                gt_trans = np.array(meta['cam_translation' + meta_idx][0])  # in [m]

                rgb_img = Image.open(rgb_addr)
                imgpts_gt, jac = cv2.projectPoints(cld[itemid] * 1e3,
                                                   gt_rot1,
                                                   gt_trans * 1e3,
                                                   cam_mat, dist)
                cv2_img_gt = cv2.polylines(np.array(rgb_img), np.int32([np.squeeze(imgpts_gt)]), True, (0, 255, 255))

                img_name_gt = args.save_images_path + 'test1.gt.cloud.png'
                cv2.imwrite(img_name_gt, cv2_img_gt)
                cld_image_gt = cv2.imread(img_name_gt)

                # print("pred_trans \n", my_t_)
                # print("gt_trans \n", gt_trans)
                # print("pred_rot \n", mat_r)
                # print("gt_rot \n", gt_rot1)

                ############################
                # reprojection error
                ############################

                # ADD = np.mean(np.linalg.norm(imgpts - imgpts_gt, axis=1)) / 10 # [cm]
                # print("ADD: {:.2f} [cm]".format(ADD))

                ############################
                # Error Metrics
                ############################

                T_pred, T_gt = my_t_, gt_trans
                R_pred, R_gt = mat_r, gt_rot1

                # ADD
                pred = np.dot(cld[itemid], R_pred)
                pred = np.add(pred,  T_pred)

                target = np.dot(cld[itemid], R_gt)
                target = np.add(target, T_gt)

                ADD = np.mean(np.linalg.norm(pred - target, axis=1))

                # ADD-S
                tree = KDTree(pred)
                dist, ind = tree.query(target)
                ADD_S = np.mean(dist)

                # translation
                T_error = np.linalg.norm(T_pred - T_gt)

                # rot
                error_cos = 0.5 * (np.trace( R_pred @ np.linalg.inv(R_gt)) - 1.0)
                error_cos = min(1.0, max(-1.0, error_cos))
                error = np.arccos(error_cos)
                R_error = 180.0 * error / np.pi

                print("ADD: {:.2f} [cm]".format(ADD * 100))  # [cm]
                print("ADD-S: {:.2f} [cm]".format(ADD_S * 100))
                print("T: {:.2f} [cm]".format(T_error * 100))  # [cm]
                print("Rot: {:.2f} [def]".format(R_error))

                Class_id_list.append(itemid)
                ADD_list.append(ADD)
                ADD_S_list.append(ADD_S)
                T_list.append(T_error)
                R_list.append(R_error)

                num_total += 1
                ADD = min(ADD, ADD_S) * 100
                if ADD < args.ADD:
                    num_correct += 1
                    print('{} Pass! Distance: {:.2f}\n'.format(classes_full[int(itemid) - 1], ADD))
                    fw.write('{} Pass! Distance: {:.2f}\n'.format(classes_full[int(itemid) - 1], ADD))
                    # print('{} NOT Pass! Distance: {:.2f}\n'.format(int(itemid) - 1, ADD))
                    # fw.write('{} NOT Pass! Distance: {:.2f}\n'.format(int(itemid) - 1, ADD))
                else:
                    print('{} NOT Pass! Distance: {:.2f}\n'.format(classes_full[int(itemid) - 1], ADD))
                    fw.write('{} NOT Pass! Distance: {:.2f}\n'.format(classes_full[int(itemid) - 1], ADD))
                    # print('{} NOT Pass! Distance: {:.2f}\n'.format(int(itemid) - 1, ADD))
                    # fw.write('{} NOT Pass! Distance: {:.2f}\n'.format(int(itemid) - 1, ADD))
                if args.visualize:
                    plt.subplot(3, 2, 1)
                    plt.title("rgb")
                    plt.imshow(img)
                    plt.subplot(3, 2, 2)
                    plt.title("depth")
                    plt.imshow(depth)
                    plt.subplot(3, 2, 3)
                    plt.title("mask")
                    plt.imshow(mask_label)
                    plt.subplot(3, 2, 4)
                    plt.title("bbox")
                    plt.imshow(bbox_image)
                    plt.subplot(3, 2, 5)
                    plt.title("gt - quarternion")
                    plt.imshow(cld_image_gt)
                    plt.subplot(3, 2, 6)
                    plt.title("pred - quarternion")
                    plt.imshow(cld_image)
                    plt.show()

            except ZeroDivisionError:
                print("DenseFusion Detector Lost {0} at No.{1} keyframe".format(itemid, idx))
                my_result_wo_refine.append([0.0 for i in range(7)])
                my_result.append([0.0 for i in range(7)])

        ### scio.savemat('{0}/{1}.mat'.format(args.result_wo_refine_dir, '%04d' % idx), {'poses': my_result_wo_refine})
        ### scio.savemat('{0}/{1}.mat'.format(args.result_refine_dir, '%04d' % idx), {'poses': my_result})
        scio.savemat('{0}/{1}.mat'.format(args.error_metrics_dir, '%04d' % idx),
                        {"Class_IDs": Class_id_list, "ADD": ADD_list, "ADD_S": ADD_S_list, "R": R_list, "T": T_list})
        print("******************* Finish No.{0} keyframe *******************".format(idx))
        print("*** Num Correct: {}/{} ***\n".format(num_correct, num_total))