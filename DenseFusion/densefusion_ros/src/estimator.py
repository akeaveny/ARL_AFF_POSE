#! /usr/bin/env python

import argparse
import os
import sys
import copy
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math

np.seterr(divide='ignore')

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

##################################
### GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# device = torch.device("cpu")

print("\n********* Torch GPU ************")
print(torch.__version__)
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print("*********************************\n")

ROOT_DIR = os.path.abspath("/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/densefusion_ros/src/")
print("ROOT_DIR: ", ROOT_DIR)

print("cwd: ", os.getcwd())

# ========== lib local to src for ros-2.7 env ================
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor

knn = KNearestNeighbor(1)


class DenseFusionEstimator():

    def __init__(self, model, refine_model,
                 num_points, num_points_mesh, iteration, bs, num_obj,
                 classes_file_, class_ids_file_,
                 cam_width, cam_height, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy):

        """ --- INIT --- """
        self.num_points = num_points
        self.num_points_mesh = num_points_mesh
        self.iteration = iteration
        self.bs = bs
        self.num_obj = num_obj

        self.estimator = PoseNet(num_points=self.num_points, num_obj=self.num_obj)
        self.estimator.cuda()
        self.estimator.load_state_dict(torch.load(model))
        self.estimator.eval()

        self.refiner = PoseRefineNet(num_points=self.num_points, num_obj=self.num_obj)
        self.refiner.cuda()
        self.refiner.load_state_dict(torch.load(refine_model))
        self.refiner.eval()

        # TODO: need norm ?
        self.norm = transforms.Normalize(mean=[0.59076867, 0.51179716, 0.47878297],
                                         std=[0.16110815, 0.16659215, 0.15830115])

        """ --- Load 3D Object Models --- """
        class_file, class_id_file = open(classes_file_), open(class_ids_file_)
        self.class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

        self.cld = {}
        for idx, class_id in enumerate(self.class_IDs):
            class_input = class_file.readline()
            if not class_input:
                break
            input_file = open(
                '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/densefusion_ros/models/hammer_01_grasp.xyz')
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

        """ --- Camera Params --- """
        self.width = cam_width
        self.height = cam_height
        self.border_list = np.arange(0, self.width + 1 if self.width > self.height else self.height + 1, 40)
        self.border_list[0] = -1
        self.cam_scale = cam_scale
        self.cam_fx = cam_fx
        self.cam_fy = cam_fy
        self.cam_cx = cam_cx
        self.cam_cy = cam_cy

        print("*** Successfully loaded DenseFusion! ***")

    def get_bbox(self, label, affordance_id, img_width, img_length, border_list):
        rows = np.any(label == affordance_id, axis=1)
        cols = np.any(label == affordance_id, axis=0)

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

    def get_refined_pose(self, rgb, depth, mask, meta=None, debug=False, visualize=False, check_pose=False):

        affordance_ids = np.unique(mask)
        for affordance_id in affordance_ids[1:-1]:  # EXCLUDE THE BACKGROUND 0
            if affordance_id in self.class_IDs:
                itemid = affordance_id

                my_result_wo_refine = []
                my_result = []
                xmap = np.array([[j for i in range(self.height)] for j in range(self.width)])
                ymap = np.array([[i for i in range(self.height)] for j in range(self.width)])

                rmin, rmax, cmin, cmax = self.get_bbox(mask, itemid, self.width, self.height, self.border_list)
                # print("\nbbox: ", rmin, rmax, cmin, cmax)
                # ================ bbox =============
                bbox_image = np.array(rgb.copy())
                cv.rectangle(bbox_image, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
                # img_name = self.test_folder + 'test1.bbox.png'
                # cv.imwrite(img_name, img_bbox)
                # bbox_image = cv.imread(img_name)

                mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                mask_label = ma.getmaskarray(ma.masked_equal(mask, itemid))
                mask = mask_label * mask_depth
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                if len(choose) > self.num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, self.num_points - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                # ============ cloud =============
                pt2 = depth_masked / self.cam_scale
                pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                cloud /= 100

                img_masked = np.array(rgb)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32) / 10)
                choose = torch.LongTensor(choose.astype(np.int32))
                # img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                img_masked = self.norm(torch.from_numpy(img_masked.astype(np.float32)))
                index = torch.LongTensor([itemid - 1])

                cloud = Variable(cloud).cuda()
                choose = Variable(choose).cuda()
                img_masked = Variable(img_masked).cuda()
                index = Variable(index).cuda()

                cloud = cloud.view(1, self.num_points, 3)
                img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

                pred_r, pred_t, pred_c, emb = self.estimator(img_masked, cloud, choose, index)
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, self.num_points, 1)

                pred_c = pred_c.view(self.bs, self.num_points)
                how_max, which_max = torch.max(pred_c, 1)
                # print("how_max: {:.5f}".format(how_max.detach().cpu().numpy()[0]))

                pred_t = pred_t.view(self.bs * self.num_points, 1, 3)
                points = cloud.view(self.bs * self.num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())
                # print("my_pred w/o refinement: \n", my_pred)

                for ite in range(0, self.iteration):
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points,
                                                                                                     1).contiguous().view(
                        1, self.num_points, 3)
                    my_mat = quaternion_matrix(my_r)
                    R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                    my_mat[0:3, 3] = my_t

                    new_cloud = torch.bmm((cloud - T), R).contiguous()
                    pred_r, pred_t = self.refiner(new_cloud, emb, index)
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

                # print("my_pred w/ {} refinement: \n{}".format(ite, my_pred))

                # ===================== SCREEN POINTS =====================
                cam_mat = np.array([[self.cam_fx, 0, self.cam_cx], [0, self.cam_fy, self.cam_cy], [0, 0, 1]])
                dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

                # ===================== PREDICTION =====================
                ''' ========= quarternion ========= '''
                mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                my_t_ = my_t

                imgpts, jac = cv.projectPoints(self.cld[itemid] * 1e3,
                                               mat_r,
                                               my_t_ * 1e3,
                                               cam_mat, dist)
                cld_img_pred = cv.polylines(np.array(rgb.copy()), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                if check_pose:
                    meta_idx = '0' + np.str(itemid)
                    ### print("meta_idx: ", meta_idx)
                    # ===================== GT =====================
                    gt_trans = np.array(meta['cam_translation' + meta_idx][0]) / 1e3 / 10
                    gt_rot1 = np.dot(np.array(meta['rot' + meta_idx]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                    gt_rot1 = np.dot(gt_rot1.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

                    imgpts_gt, jac = cv.projectPoints(self.cld[itemid] * 1e3,
                                                      gt_rot1,
                                                      gt_trans * 1e3,
                                                      cam_mat, dist)
                    cld_img_gt = cv.polylines(np.array(rgb.copy()), np.int32([np.squeeze(imgpts_gt)]), True,
                                              (0, 255, 255))

                    ADD = np.mean(np.linalg.norm(imgpts - imgpts_gt, axis=1)) / 10
                    # print("ADD: {:.2f} [cm]".format(ADD))

                    ############################
                    # TODO: ADD-S
                    ############################

                    # my_r = quaternion_matrix(my_r)[:3, :3]
                    # mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                    # pred = np.dot(cld[itemid] * 1e3, mat_r.T)
                    # pred = np.add(pred,  my_t * 1e3)

                    # target = np.dot(cld[itemid] * 1e3, gt_rot1.T)
                    # target = np.add(target, gt_trans * 1e3)

                    # if idx[0].item() in sym_list:
                    #     pred = torch.from_numpy(pred.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                    #     target = torch.from_numpy(target.astype(np.float32)).cuda().transpose(1, 0).contiguous()
                    #     inds = knn(target.unsqueeze(0), pred.unsqueeze(0))
                    #     target = torch.index_select(target, 1, inds.view(-1) - 1)
                    #     dis = torch.mean(torch.norm((pred.transpose(1, 0) - target.transpose(1, 0)), dim=1), dim=0).item()

                    # ADD_ = np.mean(np.linalg.norm(pred - target, axis=1)) / 10 # [cm]
                    # print("ADD: {:.2f} [cm]".format(ADD_))

                # if visualize:
                #     plt.figure(2)
                #     plt.subplot(2, 3, 1)
                #     plt.title("rgb")
                #     plt.imshow(rgb)
                #     plt.subplot(2, 3, 2)
                #     plt.title("depth")
                #     plt.imshow(depth)
                #     plt.subplot(2, 3, 3)
                #     plt.title("mask")
                #     plt.imshow(mask)
                #     plt.subplot(2, 3, 4)
                #     plt.title("bbox")
                #     plt.imshow(bbox_image)
                #     plt.subplot(2, 3, 5)
                #     plt.title("pred")
                #     plt.imshow(cld_img_pred)
                #     if check_pose:
                #         plt.subplot(2, 3, 6)
                #         plt.title("gt")
                #         plt.imshow(cld_img_gt)
                #     plt.ioff()
                #     plt.pause(2)

                return np.array(my_r), np.array(my_t * 1e3), cld_img_pred
            else:
                print("\n ***** Could not Detect Object Parts! ***** ")
                print("Detected Affordance IDs: ", affordance_ids)
                print("Avaliable Object IDs: {}\n".format(self.class_IDs))
                return None

