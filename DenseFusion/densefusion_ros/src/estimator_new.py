#!/usr/bin/env python

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

# ========== GPU config ================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# ========== lib local to src for ros-2.7 env ================
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from lib.knn.__init__ import KNearestNeighbor
knn = KNearestNeighbor(1)

# ========== MASK RCNN ================
from mrcnn import matterport_dataset_syn as Affordance
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

from keras.backend import clear_session

import matplotlib.pyplot as plt

class DenseFusionEstimator():

    def __init__(self, model, refine_model,
                            num_points, num_points_mesh, iteration, bs, num_obj,
                                classes_file_, class_ids_file_,
                                    cam_width, cam_height, cam_scale, cam_fx, cam_fy, cam_cx, cam_cy,
                                        model_path, num_classes, logs=None):

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

        #TODO: need norm ?
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
            input_file = open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/models/{0}/{0}_grasp.xyz'.format(class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id]) * 1e3 # [m] to [mmm]
            input_file.close()

        """ --- Camera Params --- """
        self.width = cam_width
        self.height = cam_height
        self.border_list = np.arange(0, self.width+1 if self.width > self.height else self.height+1, 40)
        self.border_list[0] = -1
        self.cam_scale = cam_scale
        self.cam_fx = cam_fx
        self.cam_fy = cam_fy
        self.cam_cx = cam_cx
        self.cam_cy = cam_cy

        print("--- Successfully loaded DenseFusion! ---\n")

        """ --- Mask_RCNN --- """
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

    def detect_and_get_masks(self, image):

        # ## ============== SYNTHETIC ===================
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Detect objects
        cur_detect = self.__model.detect([image], verbose=0)[0]
        if cur_detect['masks'].shape[-1] > 0:
            pre_detect = cur_detect  # TODO:

        # get instance_masks
        semantic_masks, instance_masks, color_masks, pre_detect, good_detect = self.seq_get_masks(image, pre_detect, cur_detect)
        if good_detect:
            return instance_masks
        else:
            print(" --- Bad Detect for Mask R-CNN --- ")
            return None

    def seq_get_masks(self, image, pre_detection, cur_detection):

        cur_masks = cur_detection['masks']
        cur_class_ids = cur_detection['class_ids']
        cur_rois = cur_detection['rois']

        pre_masks = pre_detection['masks']
        pre_class_ids = pre_detection['class_ids']
        pre_rois = pre_detection['rois']

        new_masks = pre_detection['masks']
        new_class_ids = pre_detection['class_ids']
        new_rois = pre_detection['rois']

        instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        good_detection = True
        print('cur_masks.shape: {}'.format(cur_masks.shape[-1]))

        if cur_masks.shape[-1] > 0:
            mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

            for i in range(cur_masks.shape[-1]):
                sub = np.abs(pre_rois - cur_rois[i])
                dist = sum(sub.transpose())
                # print('cur_rois[i]: {}'.format(cur_rois[i]))
                mask_index = dist.argmin()
                if dist.min() < 50:
                    if new_class_ids[mask_index] != cur_class_ids[i]:  # bad classification
                        good_detection = False
                        pass
                    elif new_class_ids[mask_index] == 7:
                        pass
                    else:
                        new_rois[mask_index, :] = cur_rois[i,
                                                  :]  # change order of current masks to follow the mask order of previous prediction
                else:
                    good_detection = False
                    pass

                # print("cur_class_ids[i] :", cur_class_ids[i])
                semantic_mask = semantic_mask_one * cur_class_ids[i]
                semantic_masks = np.where(cur_masks[:, :, i], semantic_mask, semantic_masks).astype(np.uint8)

                # instance_mask = instance_mask_one * (mask_index+1)
                instance_mask = instance_mask_one * cur_class_ids[i]
                instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

        # print('old rois: \n {}'.format(cur_rois))
        # print('new rois: \n {}'.format(new_rois))

        instance_to_color = Affordance.color_map()
        color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for key in instance_to_color.keys():
            color_masks[instance_masks == key] = instance_to_color[key]

        if good_detection:
            pre_detection['masks'] = new_masks
            pre_detection['class_ids'] = new_class_ids
            pre_detection['rois'] = new_rois
        return semantic_masks, instance_masks, color_masks, pre_detection, good_detection

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

    def get_bbox(self, label, affordance_id, img_width, img_length, border_list):
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

    def get_refined_pose(self, rgb, depth, mask=None, meta=None, debug=False, visualize=False, check_pose=False):

        mask = self.detect_and_get_masks(rgb)

        plt.subplot(2, 3, 1)
        plt.title("rgb")
        plt.imshow(rgb)
        plt.subplot(2, 3, 2)
        plt.title("depth")
        plt.imshow(depth)
        plt.subplot(2, 3, 3)
        plt.title("mask")
        plt.imshow(mask)
        plt.ioff()
        plt.pause(10)

        affordance_ids = np.unique(mask)
        for affordance_id in affordance_ids[1:-1]: # EXCLUDE THE BACKGROUND 0
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
                    T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(self.num_points, 1).contiguous().view(1, self.num_points, 3)
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
                my_t = my_t * 1000

                imgpts, jac = cv.projectPoints(self.cld[itemid],
                                                mat_r,
                                                my_t,
                                                cam_mat, dist)
                cld_img_pred = cv.polylines(np.array(rgb.copy()), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                if check_pose:
                    meta_idx = '0' + np.str(itemid)
                    # ===================== GT =====================
                    gt_trans = np.array(meta['cam_translation' + meta_idx][0]) / 10
                    gt_rot1 = np.dot(np.array(meta['rot' + meta_idx]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                    gt_rot1 = np.dot(gt_rot1.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

                    imgpts_gt, jac = cv.projectPoints(self.cld[itemid],
                                                       gt_rot1,
                                                       gt_trans,
                                                       cam_mat, dist)
                    cld_img_gt = cv.polylines(np.array(rgb.copy()), np.int32([np.squeeze(imgpts_gt)]), True,
                                               (0, 255, 255))

                    # print("--- Translation ---") # TODO: rospy logging
                    # print("Pred: \n:", my_t)
                    # print("GT \n:", gt_trans)
                    # print("--- Rotation ---")
                    # print("Pred: \n:", mat_r)
                    # print("GT \n:", gt_rot1)

                    # print("ADD: ", np.linalg.norm(imgpts - imgpts_gt) / self.num_points, "[mm]")
                    print('ADD: {:.2f}[mm]'.format(np.linalg.norm(imgpts - imgpts_gt) / self.num_points))

                if visualize:
                    plt.subplot(2, 3, 1)
                    plt.title("rgb")
                    plt.imshow(rgb)
                    plt.subplot(2, 3, 2)
                    plt.title("depth")
                    plt.imshow(depth)
                    plt.subplot(2, 3, 3)
                    plt.title("mask")
                    plt.imshow(mask)
                    plt.subplot(2, 3, 4)
                    plt.title("bbox")
                    plt.imshow(bbox_image)
                    plt.subplot(2, 3, 5)
                    plt.title("pred")
                    plt.imshow(cld_img_pred)
                    if check_pose:
                        plt.subplot(2, 3, 6)
                        plt.title("gt")
                        plt.imshow(cld_img_gt)
                    plt.ioff()
                    plt.pause(10)
            else:
                print("\n --- Could not Detect Object Parts! --- ")
                print("Detected Affordance IDs: ", affordance_ids)
                print("Avaliable Object IDs: {}\n".format(self.class_IDs))
                return False
        return True

