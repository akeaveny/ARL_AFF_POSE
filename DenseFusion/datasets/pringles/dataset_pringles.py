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
            # self.path = '/data/Akeaveny/Datasets/pringles/pringles_config/dataset_config/train_data_list.txt'
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/train_data_list_zed.txt'
        elif mode == 'test':
            # self.path = '/data/Akeaveny/Datasets/pringles/pringles_config/dataset_config/test_data_list.txt'
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/test_data_list_zed.txt'
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
            # TODO:
            self.real.append(input_line)
            # self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        # self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('/data/Akeaveny/Datasets/pringles/pringles_config/dataset_config/classes.txt')

        # TODO:
        self.object_id = 4 # object json file
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            input_file = open('/data/Akeaveny/Datasets/pringles/pringles_config/dataset_config/{1}/points.xyz'.format(self.root, class_input[:-1]))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()
                if not input_line:
                    break
                input_line = input_line[:-1].split(' ')
                self.cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()

            class_id += 1

        # TODO
        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50

        # Images:  /data/Akeaveny/Datasets/pringles/zed/train/000???.png
        # Loaded Images:  1000
        # ---------stats---------------
        # Means:
        #  [[0.63822823]
        #  [0.64762547]
        #  [0.65962552]]
        # STD:
        #  [[0.14185635]
        #  [0.14539438]
        #  [0.11838722]]
        self.norm = transforms.Normalize(mean=[0.63822823, 0.64762547, 0.65962552],
                                         std=[0.14185635, 0.14539438, 0.11838722])

        # TODO:
        self.symmetry_obj_idx = [1]
        self.num_pt_mesh_small = 500 # 1000
        self.num_pt_mesh_large = 500 # 1000
        self.refine = refine
        self.front_num = 0

        print("Loaded: ", len(self.list))
        print("Real Images: ", len(self.real))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}.depth.16.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}.cs.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        # print('{0}/{1}-meta.mat'.format(self.root, self.list[index]))

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, self.object_id))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        # ========== ADD TINT ==========
        if self.add_noise:
            img = self.trancolor(img)

        # ## ============== SYNTHETIC ===================
        img = np.array(img)
        if img.shape[-1] == 4:
            image = img[..., :3]

        rmin, rmax, cmin, cmax = get_bbox(mask)
        # print("\nbbox: ", rmin, rmax, cmin, cmax)
        ## ================ bbox =============
        # img_bbox = np.array(img.copy())
        # img_name = '/data/Akeaveny/Datasets/pringles/temp/1.bbox.png'
        # cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
        # cv2.imwrite(img_name, img_bbox)

        # x1, y1, x2, y2 = meta['bbox'].flatten().astype(np.int32)
        # print("ground truth: ", meta['bbox'].flatten().astype(np.int32))
        # ================ bbox =============
        # img_gt = np.array(img.copy())
        # img_name = '/data/Akeaveny/Datasets/pringles/temp/' + np.str(self.list[index]) + '.gt.png'
        # cv2.rectangle(img_gt, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.imwrite(img_name, img_gt)

        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img_masked = img

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        #
        # if len(choose) > self.num_pt:
        #     c_mask = np.zeros(len(choose), dtype=int)
        #     c_mask[:self.num_pt] = 1
        #     np.random.shuffle(c_mask)
        #     choose = choose[c_mask.nonzero()]
        # else:
        #     choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_cx = meta['cx'][0]
        cam_cy = meta['cy'][0]
        cam_fx = meta['fx'][0]
        cam_fy = meta['fy'][0]

        cam_rotation4 = np.array(meta['rot'])
        cam_translation = np.array(meta['cam_translation'][0])

        # cam_rotation = np.array(meta['obj2cam_rotation'])
        # cam_rotation1 = np.array(meta['obj2cam_rotation1'])
        # cam_rotation2 = np.array(meta['obj2world_rotation'])
        # cam_rotation3 = np.array(meta['obj2world_rotation1'])
        # cam_rotation5 = np.array(meta['rot1'])

        # cam_translation1 = np.array(meta['trans'][0], dtype="float64")

        translation_noise = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # ============ cloud =============
        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud /= 100

        if self.add_noise:
            cloud = np.add(cloud, translation_noise)

        # print(self.cld[obj[idx]].shape)
        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)
        # print("model_points: ", model_points*1e12)

        target = np.dot(model_points, cam_rotation4)
        if self.add_noise:
            target = np.add(target, cam_translation/10 + translation_noise)
        else:
            target = np.add(target, cam_translation/10)

        # fw = open('/data/Akeaveny/Datasets/pringles/temp/cld.pringles.xyz'.format(index), 'w')
        # for it in cloud:
        #     fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()
        #
        # fw = open('/data/Akeaveny/Datasets/pringles/temp/model_points.pringles.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()
        #
        # fw = open('/data/Akeaveny/Datasets/pringles/temp/tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        # ===================== SCREEN POINTS =====================
        cam_mat = np.array([[cam_fx[0], 0, cam_cx[0]], [0, cam_fy[0], cam_cy[0]], [0, 0, 1]])
        dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        cv2_img = Image.open('{0}/{1}.png'.format(self.root, self.list[index]))
        imgpts, jac = cv2.projectPoints(cloud, np.eye(3), np.zeros(shape=cam_translation.shape), cam_mat, dist)
        cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
        cv2.imwrite('/data/Akeaveny/Datasets/pringles/temp/cv2.cloud.png', cv2_img)

        # imgpts, jac = cv2.projectPoints(target, np.eye(3), np.zeros(shape=cam_translation.shape), cam_mat, dist)
        # cv2_img = Image.open('{0}/{1}.png'.format(self.root, self.list[index]))
        # cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
        # cv2.imwrite('/data/Akeaveny/Datasets/pringles/temp/cv2.target.png', cv2_img)

        imgpts, jac = cv2.projectPoints(self.cld[obj[idx]], cam_rotation4.T, cam_translation / 10, cam_mat, dist)
        cv2_img = Image.open('{0}/{1}.png'.format(self.root, self.list[index]))
        cv2_img = cv2.polylines(np.array(cv2_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))
        cv2.imwrite('/data/Akeaveny/Datasets/pringles/temp/cv2.cld[obj[idx]].png', cv2_img)

        # # =========== save images ================
        p_img = np.transpose(img_masked, (1, 2, 0))
        scipy.misc.imsave('/data/Akeaveny/Datasets/pringles/temp/input.png', p_img)
        scipy.misc.imsave('/data/Akeaveny/Datasets/pringles/temp/label.png',
                          mask[rmin:rmax, cmin:cmax].astype(np.int32))

        return torch.from_numpy(cloud.astype(np.float32)/1000), \
                   torch.LongTensor(choose.astype(np.int32)), \
                   self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                   torch.from_numpy(target.astype(np.float32)/1000), \
                   torch.from_numpy(model_points.astype(np.float32)/1000), \
                   torch.LongTensor([int(obj[idx]) - 1])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

# img_width = 512
# img_length = 512
# TODO:
img_width = 720
img_length = 1280

# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 500]
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
               720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)

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