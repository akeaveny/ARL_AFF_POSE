import os
print('The current working directory:')
print(os.getcwd())

import _init_paths
import argparse
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

import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
from random import random
from vanilla_segmentation.segnet import SegNet as segnet

# Flags
SHOW_IMAGE = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
bs = 1
dataset_config_dir = '/data/Akeaveny/Datasets/YCB_Video_Dataset/ycb_config/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'

def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
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

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

seg_model = segnet()
seg_model.cuda()
seg_model.load_state_dict(torch.load('vanilla_segmentation/trained_models/model_45_0.12136433521658183.pth'))
seg_model.eval()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
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

f, (ax1, ax2, ax3) = plt.subplots(3)
f, ax4 = plt.subplots(nrows=2, ncols=2)
for now in range(0, 2949):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    true_label_img = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root, testlist[now])))

    """ =================== PREDICT SEGMENTATION ================================"""
    rgb = np.transpose(img, (2, 0, 1))
    rgb = norm(torch.from_numpy(rgb.astype(np.float32))).cuda()
    seg_data = seg_model(rgb.unsqueeze(0))
    seg_data2 = torch.transpose(seg_data[0], 0, 2)
    seg_data2 = torch.transpose(seg_data2, 0, 1)
    seg_image = torch.argmax(seg_data2, dim=-1)
    obj_list = torch.unique(seg_image).detach().cpu().numpy()
    label_img = seg_image.detach().cpu().numpy()
    print("Class Item IDs: ", np.unique(label_img))
    cv2.imwrite('label_img.png', label_img)

    if SHOW_IMAGE:
        # print("----------depth----------------")
        # depth_cv = depth
        # # print(depth_cv.shape)
        # print(depth_cv.shape)
        # # plt.subplot(1, 2, 1)
        # # plt.imshow(depth_cv)
        # print("----------rbg----------------")
        # rgb = img
        # # # # print(rgb)
        # # print(rgb.shape)
        # # plt.subplot(1, 2, 2)
        # # plt.imshow(rgb)
        # # plt.ioff()
        # # plt.pause(0.0333)
        # print("----------label----------------")
        # plt.imshow(label_img)
        # plt.ioff()
        # plt.pause(0.5)
        ax1.imshow(img)
        ax2.imshow(depth)
        ax3.imshow(true_label_img)

    posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])

    lst = posecnn_rois[:, 1:2].flatten()
    my_result_wo_refine = []
    my_result = []

    for idx in range(len(lst)):
        itemid = lst[idx]
        try:
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            # print("Choose: %d and Number of Points: %d" % (len(choose), num_points))
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

            # print("Mask: ", mask.shape) # 480 x 680
            # print("ROIS: ", rmin-rmax, cmin-cmax)
            # print("Choose: ",choose[0, 0:200]) # 1 x 1000

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

            # TODO: WHAT DOES C REALLY MEAN?
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

                """ =================== VIS ================================"""
                if SHOW_IMAGE:
                    # print("----------seg obj----------------")
                    rgb = np.transpose(img, (2, 0, 1))
                    rgb = norm(torch.from_numpy(rgb.astype(np.float32))).cuda()
                    img_out = torch.nn.functional.softmax(seg_model(rgb.unsqueeze(0)), dim=1)
                    img_out_2 = img_out.cpu().data.numpy()

                    seg_obj = np.argmax(img_out_2[0, :, :, :], axis=0) == itemid
                    cv2.imwrite("label_img.png", seg_obj.astype(np.uint8))

                    ax4[0,1].imshow(seg_obj)

                    """ =================== BOUNDING BOX ================================"""
                    image = cv2.imread('label_img.png')
                    image_copy = image.copy()
                    # cv2.imwrite("rgb_img.png", img)
                    # rgb_image = cv2.imread('rgb_img.png')
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
                    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                    ROI_number = 0
                    ROI = image[cmin:cmax, rmin:rmax]
                    # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                    cv2.rectangle(image_copy, (cmin, rmin), (rmax, cmax), (100, 36, 12), 4)
                    ROI_number += 1
                    for c in cnts:
                        x, y, w, h = cv2.boundingRect(c)
                        ROI = image[y:y + h, x:x + w]
                        # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (36, 255, 12), 2)
                        ROI_number += 1
                    cv2.imwrite("image_copy.png", image_copy.astype(np.uint8))
                    ax4[1, 0].imshow(image_copy.astype(np.uint8))
                    """ =================== BOUNDING BOX ================================"""

                    # print("----------seg mask----------------")
                    rgb = np.transpose(img, (2, 0, 1))
                    rgb = norm(torch.from_numpy(rgb.astype(np.float32))).cuda()
                    seg_data = seg_model(rgb.unsqueeze(0))
                    seg_data2 = torch.transpose(seg_data[0], 0, 2)
                    seg_data2 = torch.transpose(seg_data2, 0, 1)
                    seg_image = torch.argmax(seg_data2, dim=-1)
                    obj_list = torch.unique(seg_image).detach().cpu().numpy()
                    seg_mask = seg_image.detach().cpu().numpy()

                    # print(obj_list)
                    ax4[0,0].imshow(seg_mask)

                if SHOW_IMAGE:
                    cam_mat = np.matrix([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
                    mat_r = quaternion_matrix(my_r)[0:3, 0:3]
                    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    imgpts, jac = cv2.projectPoints(cld[itemid], mat_r, my_t, cam_mat, dist)
                    # open_cv_image = draw(open_cv_image, imgpts.get(), itemid)
                    img_show = None
                    img_show = cv2.polylines(np.array(img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

                    print("----------pose----------------")
                    print("Objects", lst)
                    print("Current Object ID: ", itemid)
                    print("Confidence: ", pred_c.shape)
                    print("Translation: ", my_t)
                    print("Quaternion: ", my_r)
                    ax4[1,1].imshow(img_show)
                    plt.ioff()
                    plt.pause(10)

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

            my_result.append(my_pred.tolist())
        # except ZeroDivisionError:
        except (ZeroDivisionError, ValueError):
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now), {'poses':my_result_wo_refine})
    scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses':my_result})
    print("Finish No.{0} keyframe".format(now))