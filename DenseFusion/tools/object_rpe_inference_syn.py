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
import cv2
import matplotlib.pyplot as plt

# ========== GPU config ================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ================== bbox =========================
def get_bbox(label, affordance_id, img_width, img_length, border_list):

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

# ================== Trained Models =========================
norm_ = transforms.Normalize(mean=[0.59076867, 0.51179716, 0.47878297], std=[0.16110815, 0.16659215, 0.15830115])

dataset_config_dir = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/'
class_names_file_dir = dataset_config_dir + 'dataset_config/object_rpe_classes_.txt'
class_ids_file_dir = dataset_config_dir + 'dataset_config/object_rpe_class_ids_.txt'
poses_path = dataset_config_dir + 'dataset_config/DenseFusion_Poses.txt'

test_folder = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/test_densefusion/'
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/'
folder_to_save = "combined_tools_val/dr/"

trained_model_path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/trained_models/parts_affordance_syn/parts_affordance1_hammer1/'
model = trained_model_path + 'pose_model_5_0.09450025089477782.pth' # TODO:
refine_model = trained_model_path + 'pose_refine_model_247_0.04728509413062927.pth'

# ================= classes =========================
# class_names = np.loadtxt(class_names_file_dir, dtype=np.str)
# class_IDs = np.loadtxt(class_ids_file_dir, dtype=np.int32)
# print("Classes: ", class_IDs)

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/class_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("Classes: ", class_IDs)

# # ================== 3D MODELS =========================
num_points = 100
num_points_mesh = 100
iteration = 5
bs = 1

# ================== setup =========================
cld = {}
for idx, class_id in enumerate(class_IDs):
    class_input = class_file.readline()
    # print("class_id: ", class_id)
    # print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/models/{0}/{0}_grasp.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    cld[class_id] = np.array(cld[class_id]) * 1e3  # TODO:
    input_file.close()

# ================== DENSEFUSION =========================
num_obj = 205
estimator = PoseNet(num_points=num_points, num_obj=num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(refine_model))
refiner.eval()

f = open(poses_path, 'w')

# ================== Load Images =========================
" --- NDDS --- "
train_images_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/train_data_list.txt'
test_images_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/test_data_list.txt'
loaded_images_ = np.loadtxt(test_images_file, dtype=np.str)

for idx in range(len(loaded_images_)):

    # TODO: pick random idx
    idx = np.random.choice(len(loaded_images_), size=1, replace=False)[0]
    # print("Idx: ", idx)

    print("Image Info: ", loaded_images_[idx].split('/'))
    # ================== NDDS ==========================
    str_num = loaded_images_[idx].split('/')[-1]
    f.write(str_num)
    f.write("\n")

    rgb_addr = data_path + loaded_images_[idx] + "_rgb.png"
    depth_addr = data_path + loaded_images_[idx] + "_depth.png"
    gt_addr = data_path + loaded_images_[idx] + "_label.png"
    mask_addr = gt_addr

    # check pose
    meta_addr = data_path + loaded_images_[idx] + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    # ## ============== SYNTHETIC ===================
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

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
    # plt.title("label")
    # plt.imshow(label)
    # plt.show()
    # plt.ioff()

    affordance_ids = np.unique(label)
    for affordance_id in affordance_ids:

        if affordance_id in class_IDs:
            itemid = affordance_id
            meta_idx = '0' + np.str(itemid)
            camera_setting = meta['camera_setting' + meta_idx][0]
            print("Affordance ID: ", itemid)
            # print("Meta IDX: ", meta_idx)

            my_result_wo_refine = []
            my_result = []

            print("camera_setting: ", meta['camera_setting' + meta_idx][0])
            if camera_setting == 'Kinetic':
                height = 640
                width = 480
                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]
                cam_scale = 1
                cam_fx = 517.055
                cam_fy = 517.679
                cam_cx = 315.008
                cam_cy = 264.155
            elif camera_setting == 'Xtion':
                height = 640
                width = 480
                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]
                cam_scale = 1
                cam_fx = 570.3422241210938
                cam_fy = 570.3422241210938
                cam_cx = 319.5
                cam_cy = 239.5
            elif camera_setting == 'ZED':
                height = 1280
                width = 720
                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680,
                               720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280]
                cam_scale = 1
                cam_fx = 697.6871337890625
                cam_fy = 697.6871337890625
                cam_cx = 620.2407836914062
                cam_cy = 353.91357421875

            xmap = np.array([[j for i in range(height)] for j in range(width)])
            ymap = np.array([[i for i in range(height)] for j in range(width)])

            rmin, rmax, cmin, cmax = get_bbox(label, itemid, width, height, border_list)
            # print("\nbbox: ", rmin, rmax, cmin, cmax)
            # ================ bbox =============
            img_bbox = np.array(img.copy())
            img_name = test_folder + 'test1.bbox.png'
            cv2.rectangle(img_bbox, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)
            cv2.imwrite(img_name, img_bbox)
            bbox_image = cv2.imread(img_name)

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

                # ============ cloud =============
                pt2 = depth_masked / cam_scale
                pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)
                cloud /= 100

                img_masked = np.array(img)[:, :, :3]
                img_masked = np.transpose(img_masked, (2, 0, 1))
                img_masked = img_masked[:, rmin:rmax, cmin:cmax]

                cloud = torch.from_numpy(cloud.astype(np.float32) / 10)
                choose = torch.LongTensor(choose.astype(np.int32))
                img_masked = norm_(torch.from_numpy(img_masked.astype(np.float32)))
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
                print("how_max: {:.5f}".format(how_max.detach().cpu().numpy()[0]))

                pred_t = pred_t.view(bs * num_points, 1, 3)
                points = cloud.view(bs * num_points, 1, 3)

                my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                my_pred = np.append(my_r, my_t)
                my_result_wo_refine.append(my_pred.tolist())
                # print("my_pred w/o refinement: \n", my_pred)

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

                    # print("my_pred w/ {} refinement: \n{}".format(ite, my_pred))

                # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

                my_result.append(my_pred.tolist())

                f.write(str(int(itemid)))
                f.write("\n")
                np.savetxt(f, my_pred, newline=' ', fmt="%.6f") # TODO:
                f.write("\n")

                # ===================== SCREEN POINTS =====================
                cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

                # ===================== PREDICTION =====================
                ''' ========= quarternion ========= '''
                mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                my_t = my_t * 1000

                rgb_img = Image.open(rgb_addr)
                imgpts, jac = cv2.projectPoints(cld[itemid],
                                                mat_r,
                                                my_t,
                                                cam_mat, dist)
                cv2_img = cv2.polylines(np.array(rgb_img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                img_name = test_folder + 'test1.cv2.cloud.png'
                cv2.imwrite(img_name, cv2_img)
                cld_image = cv2.imread(img_name)

                # ===================== GT =====================
                gt_trans = np.array(meta['cam_translation' + meta_idx][0]) /10

                gt_rot1 = np.dot(np.array(meta['rot' + meta_idx]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                gt_rot1 = np.dot(gt_rot1.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

                rgb_img = Image.open(rgb_addr)
                imgpts_gt, jac = cv2.projectPoints(cld[itemid],
                                                   gt_rot1,
                                                   gt_trans,
                                                   cam_mat, dist)
                cv2_img_gt = cv2.polylines(np.array(rgb_img), np.int32([np.squeeze(imgpts_gt)]), True, (0, 255, 255))

                img_name_gt = test_folder + 'test1.gt.cloud.png'
                cv2.imwrite(img_name_gt, cv2_img_gt)
                cld_image_gt = cv2.imread(img_name_gt)

                # gt_quart = np.array(meta['quaterniona' + meta_idx][0])
                # ''' ========= quarternion ========= '''
                # gt_rot = quaternion_matrix(gt_quart)[0:3, 0:3]
                # gt_rot = np.dot(gt_rot, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                # gt_rot = np.dot(gt_rot.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))
                # gt_rot = np.dot(gt_rot.T, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
                # gt_rot = np.dot(gt_rot.T, np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
                #
                # imgpts_gt, jac = cv2.projectPoints(cld[itemid],
                #                                    gt_rot,
                #                                    gt_trans/10,
                #                                    cam_mat, dist)
                # cv2_img_gt = cv2.polylines(np.array(rgb_img), np.int32([np.squeeze(imgpts_gt)]), True, (0, 255, 255))
                #
                # img_name_gt = test_folder + 'test1.gt.cloud.png'
                # cv2.imwrite(img_name_gt, cv2_img_gt)
                # cld_image_gt1 = cv2.imread(img_name_gt)

                print("ADD: {:.2f}[mm]".format(1/num_points * np.linalg.norm(imgpts - imgpts_gt)))

                print("mat_t \n", my_t)
                print("gt_trans \n", gt_trans)
                # print("my_r: \n", my_r)
                # print("gt_quart: \n", gt_quart)
                print("mat_r \n", mat_r)
                print("gt_rot \n", gt_rot1)

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

f.close()