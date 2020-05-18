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

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        if mode == 'train':
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/train_data_list.txt'
        elif mode == 'test':
            self.path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/test_data_list.txt'
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
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
            # self.real.append(input_line)
            self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        self.len_real = len(self.real)
        self.len_syn = len(self.syn)

        class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/classes.txt')
        class_id = 1
        self.cld = {}
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break

            input_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/model_Pringles_20180815_173653581.xyz')
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

        # TODO:
        self.cam_fx_1 = 697.6871337890625
        self.cam_cx_1 = 620.2407836914062
        self.cam_fy_1 = 697.6871337890625
        self.cam_cy_1 = 353.91357421875

        # TODO:
        self.xmap = np.array([[j for i in range(720)] for j in range(1280)])
        self.ymap = np.array([[i for i in range(720)] for j in range(1280)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50

        # TODO:
        #    # Means:
        #     #  [[172.18892296]
        #     #  [172.6210821 ]
        #     #  [178.03783695]]
        #     # STD:
        #     #  [[35.9405501 ]
        #     #  [36.47614378]
        #     #  [29.89619891]]
        self.norm = transforms.Normalize(mean=[172.18892296/255, 172.6210821/255, 178.03783695/255], std=[35.9405501/255, 36.47614378/255, 29.89619891/255])

        self.symmetry_obj_idx = [1]

        # TODO:
        # self.num_pt_mesh_small = 118942
        # self.num_pt_mesh_large = 118942
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600

        self.refine = refine
        self.front_num = 2

        print("Loaded: ", len(self.list))
        print("Real Images: {0} & Syn Images: {1}".format(len(self.real), len(self.syn)))

    def __getitem__(self, index):
        img = Image.open('{0}/{1}.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}.depth.16.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}.cs.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))

        cam_cx = self.cam_cx_1
        cam_cy = self.cam_cy_1
        cam_fx = self.cam_fx_1
        cam_fy = self.cam_fy_1

        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        add_front = False
        
        # ============= DATA AUGMENTATION ================
        # if len(self.syn) > 0:
        #     if self.add_noise:
        #         for k in range(5):
        #             # print('\n --------------- AUG! -----------------')
        #             seed = random.choice(self.syn)
        #             front = np.array(self.trancolor(Image.open('{0}/{1}.png'.format(self.root, seed)).convert("RGB")))
        #             front = np.transpose(front, (2, 0, 1))
        #             f_label = np.array(Image.open('{0}/{1}.cs.png'.format(self.root, seed)))
        #             front_label = np.unique(f_label).tolist()[1:]
        #             if len(front_label) < self.front_num:
        #                 continue
        #             front_label = random.sample(front_label, self.front_num)
        #             for f_i in front_label:
        #                 mk = ma.getmaskarray(ma.masked_not_equal(f_label, f_i))
        #                 if f_i == front_label[0]:
        #                     mask_front = mk
        #                 else:
        #                     mask_front = mask_front * mk
        #             t_label = label * mask_front
        #             if len(t_label.nonzero()[0]) > self.num_pt_mesh_small :
        #                 label = t_label
        #                 add_front = True
        #                 break

        obj = meta['cls_indexes'].flatten().astype(np.int32)

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            # TODO: Change Object Index
            # print(np.unique(mask_label))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        if self.add_noise:
            img = self.trancolor(img)

        rmin, rmax, cmin, cmax = get_bbox(mask)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        # if self.list[index][:8] == 'data_syn':
            # seed = random.choice(self.real)
            # back = np.array(self.trancolor(Image.open('{0}/{1}.png'.format(self.root, seed)).convert("RGB")))
            # back = np.transpose(back, (2, 0, 1))[:, rmin:rmax, cmin:cmax]
            # img_masked = back * mask_back[rmin:rmax, cmin:cmax] + img
        # else:
        img_masked = img

        # ============= DATA AUGMENTATION ================
        # if self.add_noise:
        #     # print('\n ------------------ AUGMENTING! -------------------')
        #     img_masked = img_masked * mask_front[rmin:rmax, cmin:cmax] + front[:, rmin:rmax, cmin:cmax] * ~(
        #     mask_front[rmin:rmax, cmin:cmax])
        #     img_masked = img_masked + np.random.normal(loc=0.0, scale=7.0, size=img_masked.shape)
        #     # p_img = np.transpose(img_masked, (1, 2, 0))
        #     # scipy.misc.imsave('image_augmentation/{0}_input.png'.format(index), p_img)
        #     # scipy.misc.imsave('image_augmentation/{0}_label.png'.format(index), mask[rmin:rmax, cmin:cmax].astype(np.int32))

        target_r = meta['poses'][:, 0:3]
        target_t = np.array([meta['poses'][:, 3:4].flatten()])
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        print('\n --- choose ---')
        print("Num Points: {0} & Choose: {1} ".format(self.num_pt, len(choose)))
        print('diff: ', self.num_pt - len(choose))
        print(choose.shape)

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        # print('\n --- choose ---')
        # print("Num Points: {0} & Choose: {1} ".format(self.num_pt, len(choose)))
        print(choose.shape)

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # fw = open('image_augmentation/{0}_cld.xyz'.format(index), 'w')
        # for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        dellist = [j for j in range(0, len(self.cld[obj[idx]]))]
        if self.refine:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_large)
        else:
            dellist = random.sample(dellist, len(self.cld[obj[idx]]) - self.num_pt_mesh_small)
        model_points = np.delete(self.cld[obj[idx]], dellist, axis=0)

        # fw = open('image_augmentation/{0}_model_points.xyz'.format(index), 'w')
        # for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        # fw = open('image_augmentation/{0}_tar.xyz'.format(index), 'w')
        # for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        # fw.close()

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
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

# border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
# TODO:
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
               720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
img_width = 1280
img_length = 720

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)

    # print("--------------- rows -----------------")
    # print("Rows: ", rows.shape)
    # print(np.where(rows)[0].shape)

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
