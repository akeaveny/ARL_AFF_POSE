import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/train_data_list.txt'
        elif mode == 'test':
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/test_data_list.txt'
        print(self.path)
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        # =================== NOISE TRANSLATION ===================
        self.noise_trans = noise_trans

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            # self.real.append(input_line)
            self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        # self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/classes_train.txt')
        class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/class_ids_train.txt')
        self.class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

        # TODO:
        self.cld = {}
        for class_id in self.class_IDs:
            class_input = class_file.readline()
            # print("class_id: ", class_id)
            # print("class_input: ", class_input)
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
            self.cld[class_id] = np.array(self.cld[class_id]) * 1e3  # TODO:
            input_file.close()

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50

        # TODO
        self.norm = transforms.Normalize(mean=[0.59076867, 0.51179716, 0.47878297],
                                         std=[0.16110815, 0.16659215, 0.15830115])

        # TODO:
        self.symmetry_obj_idx = []
        self.num_pt_mesh_small = 100 # 1000
        self.num_pt_mesh_large = 100 # 1000
        self.refine = refine
        self.front_num = 0

        print("Loaded: ", len(self.list))
        print("Real Images: ", len(self.real))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}_rgb.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}_depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}_label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        test_folder = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/test_densefusion/'

        # ========== ADD TINT ==========
        if self.add_noise:
            img = self.trancolor(img)

        # ## ============== SYNTHETIC ===================
        img = np.array(img)
        if img.shape[-1] == 4:
            image = img[..., :3]

        affordance_ids = np.array(meta['Affordance_ID'].astype(np.int32))

        # TODO:
        # remove_ids = self.class_IDs + 1
        # # print("GT: ", np.unique(label))
        # # # ============== REMOVE AFFORDANCE IDs ===================
        # affordance_id = np.delete(affordance_ids, np.where([affordance_ids == remove_id for remove_id in remove_ids]))
        # rows, columns = label.shape
        # for remove_id in remove_ids:
        #     for row in range(rows):
        #         for column in range(columns):
        #             if label[row][column] == remove_id:
        #                 label[row][column] = 0
        # # print("After: ", np.unique(label))

        for affordance_id in affordance_ids:

            if affordance_id in self.class_IDs:
                obj = affordance_id
                idx = obj
                meta_idx = '0' + np.str(obj)

                while 1:
                    # idx = np.random.randint(0, np.asarray(affordance_id))
                    # idx = np.random.choice(np.asarray(affordance_id), size=1, replace=False)[0]
                    # print("Object_ID:", obj)

                    # print("GT: ", np.unique(label))

                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                    mask_label = ma.getmaskarray(ma.masked_equal(label, obj))
                    mask = mask_label * mask_depth
                    if len(mask.nonzero()[0]) > self.minimum_num_pt:
                        break

                camera_setting = meta['camera_setting' + meta_idx][0]
                # TODO:
                width = meta['height' + meta_idx].flatten().astype(np.int32)[0]
                height = meta['width' + meta_idx].flatten().astype(np.int32)[0]

                self.xmap = np.array([[j for i in range(height)] for j in range(width)])
                self.ymap = np.array([[i for i in range(height)] for j in range(width)])

                if camera_setting == 'Kinetic':
                    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]
                elif camera_setting == 'Xtion':
                    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]
                elif camera_setting == 'ZED':
                    border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
                                   720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]

                # rmax, rmin, cmax, cmin = get_bbox(label, obj, width, height, border_list)
                # rmin, rmax, cmin, cmax = rmax, rmin, cmax, cmin
                rmin, rmax, cmin, cmax = get_bbox(label, obj, width, height, border_list)
                # print("bbox: ", rmin, rmax, cmin, cmax)
                # ================ bbox =============
                # img_bbox = np.array(img.copy())
                # img_name = test_folder + '1.bbox.png'
                # cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
                # cv2.imwrite(img_name, img_bbox)

                # # TODO:
                # rmin1, rmax1, cmin1, cmax1 = meta['bbox' + meta_idx].flatten().astype(np.int32)
                # rmin, rmax, cmin, cmax = rmax1, rmin1, cmax1, cmin1
                # print("ground truth: ", rmin1, rmax1, cmin1, cmax1)
                # # ================ bbox =============
                # img_gt = np.array(img.copy())
                # img_name = test_folder + '1.bbox.gt_test.png'
                ## cv2.rectangle(img_gt, (y1, x1), (y2, x2), (255, 0, 0), 2)
                # cv2.rectangle(img_gt, (cmin1, rmin1), (cmax1, rmax1), (255, 0, 0), 2)
                # cv2.imwrite(img_name, img_gt)

                img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
                img_masked = np.array(img)

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

                # print("Mask: ", mask.shape)
                # print("Unique ", np.unique(np.array(mask[rmin:rmax, cmin:cmax])))
                # print("Choose: ", len(choose))

                if len(choose) == 0:
                    cc = torch.LongTensor([0])
                    return(cc, cc, cc, cc, cc, cc)

                if len(choose) > self.num_pt:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:self.num_pt] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                else:
                    choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                choose = np.array([choose])

                cam_cx = meta['cx' + meta_idx][0][0]
                cam_cy = meta['cy' + meta_idx][0][0]
                cam_fx = meta['fx' + meta_idx][0][0]
                cam_fy = meta['fy' + meta_idx][0][0]

                cam_translation = np.array(meta['cam_translation' + meta_idx][0])
                cam_rotation4 = np.dot(np.array(meta['rot' + meta_idx]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                cam_rotation4 = np.dot(cam_rotation4.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

                translation_noise = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

                # print("Numm points: ", self.cld[obj].shape)
                # ============ cloud =============
                cam_scale = 1.0
                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                cloud /= 100

                if self.add_noise:
                    cloud = np.add(cloud, translation_noise)

                dellist = [j for j in range(0, len(self.cld[obj]))]
                if self.refine:
                    dellist = random.sample(dellist, len(self.cld[obj]) - self.num_pt_mesh_large)
                else:
                    dellist = random.sample(dellist, len(self.cld[obj]) - self.num_pt_mesh_small)
                model_points = np.delete(self.cld[obj], dellist, axis=0)
                # print("model_points: ", model_points*1e3)

                target = np.dot(model_points, cam_rotation4.T)
                if self.add_noise:
                    target = np.add(target, cam_translation/10 + translation_noise)
                else:
                    target = np.add(target, cam_translation/10)

                # ## ===================== SCREEN POINTS =====================
                # cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                # dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                #
                # cv2_img = Image.open('{0}/{1}_rgb.png'.format(self.root, self.list[index]))
                # imgpts, jac = cv2.projectPoints(cloud, np.eye(3), np.zeros(shape=cam_translation.shape), cam_mat, dist)
                # cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
                # temp_folder = test_folder + 'cv2.cloud.png'
                # cv2.imwrite(temp_folder, cv2_img)
                #
                # cv2_img = Image.open('{0}/{1}_rgb.png'.format(self.root, self.list[index]))
                # imgpts, jac = cv2.projectPoints(target, np.eye(3), np.zeros(shape=cam_translation.shape), cam_mat, dist)
                # cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
                # temp_folder = test_folder + 'cv2.1.target.png'
                # cv2.imwrite(temp_folder, cv2_img)
                #
                # cv2_img = Image.open('{0}/{1}_rgb.png'.format(self.root, self.list[index]))
                # imgpts, jac = cv2.projectPoints(model_points, cam_rotation4, cam_translation/10, cam_mat, dist)
                # cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
                # temp_folder = test_folder + 'cv2.model_points.png'
                # cv2.imwrite(temp_folder, cv2_img)
                #
                # cv2_img = Image.open('{0}/{1}_rgb.png'.format(self.root, self.list[index]))
                # imgpts, jac = cv2.projectPoints(self.cld[obj], cam_rotation4, cam_translation/10, cam_mat, dist)
                # cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
                # temp_folder = test_folder + 'cv2.1.self.cld.png'
                # cv2.imwrite(temp_folder, cv2_img)
                #
                # ## =========== save images ================
                # p_img = np.transpose(img_masked, (1, 2, 0))
                # temp_folder = test_folder + 'input.png'
                # scipy.misc.imsave(temp_folder, p_img)
                # temp_folder = test_folder + 'label.png'
                # scipy.misc.imsave(temp_folder, mask[rmin:rmax, cmin:cmax].astype(np.int32))

                return torch.from_numpy(cloud.astype(np.float32) / 10), \
                       torch.LongTensor(choose.astype(np.int32)), \
                       self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                       torch.from_numpy(target.astype(np.float32) / 1000), \
                       torch.from_numpy(model_points.astype(np.float32) / 1000), \
                       torch.LongTensor([int(idx) - 1])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def remove_affordance_ids(self):
        return 1

def get_bbox(label, affordance_id, img_width, img_length, border_list):

    rows = np.any(label==affordance_id, axis=1)
    cols = np.any(label==affordance_id, axis=0)

    # print("\n Rows: ", np.where(rows)[0].shape)

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