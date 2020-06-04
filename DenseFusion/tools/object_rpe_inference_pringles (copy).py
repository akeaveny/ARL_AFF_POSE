import _init_paths

import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math

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

# ============= ak =============
from datasets.pringles.dataset_pringles import PoseDataset
import cv2
import matplotlib.pyplot as plt

DEBUG = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

# ========== GPU config ================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ================== Dataset =========================
norm = transforms.Normalize(mean=[0.63822823, 0.64762547, 0.65962552], std=[0.14185635, 0.14539438, 0.11838722])

dataset_config_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/'
class_names_file_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/classes.txt'
class_ids_file_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/pringles/dataset_config/class_ids.txt'

data_path = '/data/Akeaveny/Datasets/pringles/zed/densefusion/'
folder_to_save = "val/"
json_mask_id = 4

poses_path = '/data/Akeaveny/Datasets/pringles/zed/densefusion/DenseFusion_Poses.txt'

# ================== ZED =========================
cam_scale = 1
num_obj = 1

cam_cx = 620.2407836914062
cam_cy = 353.91357421875
cam_fx = 697.6871337890625
cam_fy = 697.6871337890625

img_width = 720
img_length = 1280
xmap = np.array([[j for i in range(1280)] for j in range(720)])
ymap = np.array([[i for i in range(1280)] for j in range(720)])

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
               720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]

# ================== 3D MODELS =========================
num_points = 1000
num_points_mesh = 1000
iteration = 2
bs = 1

# ================== bbox =========================
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

# ================== setup =========================
estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('/data/Akeaveny/Datasets/pringles/pringles_config/dataset_config/pringles/points.xyz')
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

# ================= class names =========================
class_names = np.loadtxt(class_names_file_dir, dtype=np.str)

# ================= detected class IDS  =========================
detected_classIDs = np.loadtxt(class_ids_file_dir, dtype=np.int32)
correct_classIDs = np.loadtxt(class_ids_file_dir, dtype=np.int32)
correct_classIDs -= 1
print("Detected Class Ids: ", detected_classIDs)

f = open(poses_path, 'w')

for now in range(1):

    # ================== NDDS ==========================
    count = 1000000 + now
    str_num = str(count)[1:]
    f.write(str_num)
    f.write("\n")

    rgb_addr = data_path + folder_to_save + str_num + ".png"
    depth_addr = data_path + folder_to_save + str_num + ".depth.16.png"
    gt_addr = data_path + folder_to_save + str_num + ".cs.png"
    mask_addr = data_path + folder_to_save + str_num + ".mask.png"

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    mask = np.array(Image.open(mask_addr))

    # depth = depth[50:530, 100:740]
    # depth = depth[50:580, 100:840]
    # img = img[50:580, 100:840, :]  # height x width
    # masks = masks[50:580, 100:840]

    # ================ check images ==============
    # plt.subplot(2, 2, 1)
    # plt.title("rgb")
    # plt.imshow(img)
    # plt.subplot(2, 2, 2)
    # plt.title("depth")
    # plt.imshow(depth)
    # plt.subplot(2, 2, 3)
    # plt.title("gt")
    # plt.imshow(gt)
    # plt.subplot(2, 2, 4)
    # plt.title("mask")
    # plt.imshow(mask)
    # plt.show()
    # plt.ioff()

    my_result_wo_refine = []
    my_result = []

    # for idx in range(len(detected_classIDs)):
    for idx in range(1):
        # idx -= 1
        itemid = json_mask_id
        maskid = idx + 1

        if DEBUG:
            print("Class Ids: ", detected_classIDs)
            print("Itemid: ", itemid)
            print("Maskid: ", maskid)

            # mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            # mask_label = ma.getmaskarray(ma.masked_equal(mask, json_mask_id))

        try:
            mask = ma.getmaskarray(ma.masked_equal(mask, json_mask_id))
            # rmin, rmax, cmin, cmax = get_bbox(mask)

            proper_mask = np.unique(mask)
            if DEBUG:
                print("Unique Masks: ", proper_mask.sum())

            if proper_mask.sum() == True: # NEED TRUE AND FALSE
                rmin, rmax, cmin, cmax = get_bbox(mask)
            else:
                break

            print('itemid: {0}\n'.format(itemid))
            print('rmin {0}, rmax {1}, cmin {2}, cmax {3}'.format(rmin, rmax, cmin, cmax))

            """ =================== BBOX ================================"""
            bbox_image = cv2.imread(rgb_addr)
            bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(bbox_image, (cmin, rmin), (cmax, rmax), (100, 36, 12), 4)
            # cv2.imwrite("image_copy.png", bbox_image.astype(np.uint8))
            # bbox_image = cv2.imread("image_copy.png")

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(masks, maskid))
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

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())

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

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

            my_result.append(my_pred.tolist())

            f.write(str(int(itemid)))
            f.write("\n")
            np.savetxt(f, my_pred, newline=' ', fmt="%.6f")
            f.write("\n")

            print("------------------------------")
            print("Detected Classes: ", detected_classIDs)
            for class_idx in range(len(class_names[correct_classIDs])):
                print(class_names[correct_classIDs[class_idx]])
            # print("Detected Object IDs: ", class_names)
            print("----------pose----------------")
            print("Object of Interest: ", class_names[itemid-1])
            print("Translation: ", my_pred[4:7])
            print("Quaternion: ", my_pred[:4])
            print("\n")

            plt.subplot(2, 2, 1)
            plt.title("rgb")
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.title("depth")
            plt.imshow(depth)
            plt.subplot(2, 2, 3)
            plt.title("mask")
            plt.imshow(masks)
            plt.subplot(2, 2, 4)
            plt.title("bbox")
            plt.imshow(bbox_image)
            plt.show()

        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

f.close()