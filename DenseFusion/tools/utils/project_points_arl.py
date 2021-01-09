import cv2
import glob
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

from skimage.color import gray2rgb

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

cld = {}
for class_id in class_IDs:
    print("class_id: ", class_id)
    class_input = class_file.readline()
    print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/arl_dataset/models/{0}.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    cld[class_id] = np.array(cld[class_id])
    input_file.close()

##################################
##################################

def get_bbox(label, affordance_id, img_width, img_length, border_list):

    ###################
    # affordance id
    ###################

    rows = np.any(label==affordance_id, axis=1)
    cols = np.any(label==affordance_id, axis=0)
    ### rows = np.any(label, axis=1)
    ### cols = np.any(label, axis=0)

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
## LOAD IMAGES
##################################
dataset = '/data/Akeaveny/Datasets/arl_dataset/'
dataset_config = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/'
images_file = 'real_train_data_list.txt'    # COMBINED

loaded_images_ = np.loadtxt('{}/{}'.format(dataset_config, images_file), dtype=np.str)

bad_choose_files = 0

# select random test images
np.random.seed(1)
num_test = len(loaded_images_)
test_idx = np.random.choice(np.arange(0, int(len(loaded_images_)), 1), size=int(num_test), replace=False)
print("Chosen Files \n", len(test_idx))

for idx in test_idx:

    ##############
    ##############

    str_num = loaded_images_[idx].split('/')[-1]

    rgb_addr = dataset + loaded_images_[idx] + "_rgb.png"
    depth_addr = dataset + loaded_images_[idx] + "_depth.png"
    gt_addr = dataset + loaded_images_[idx] + "_label.png"
    mask_addr = gt_addr

    # gt pose
    meta_addr = dataset + loaded_images_[idx] + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    # rgb
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

    affordance_ids = np.unique(np.array(label))
    for affordance_id in affordance_ids:
        if affordance_id in class_IDs:

            idx = affordance_id
            count = 1000 + idx
            meta_idx = str(count)[1:]

            ############
            # meta
            ############

            # height = meta['width' + meta_idx].flatten().astype(np.int32)[0]
            # width = meta['height' + meta_idx].flatten().astype(np.int32)[0]
            width = meta['width' + meta_idx].flatten().astype(np.int32)[0]
            height = meta['height' + meta_idx].flatten().astype(np.int32)[0]

            xmap = np.array([[j for i in range(height)] for j in range(width)])
            ymap = np.array([[i for i in range(height)] for j in range(width)])

            cam_cx = meta['cx' + meta_idx][0][0]
            cam_cy = meta['cy' + meta_idx][0][0]
            cam_fx = meta['fx' + meta_idx][0][0]
            cam_fy = meta['fy' + meta_idx][0][0]

            cam_scale = np.array(meta['camera_scale' + meta_idx])[0][0]  # 1000 for [mm] to [m]
            border_list = np.array(meta['border' + meta_idx]).flatten().astype(np.int32)

            cam_rotation4 = np.array(meta['rot' + meta_idx])
            cam_translation = np.array(meta['cam_translation' + meta_idx][0])  # in [m]

            #######################################
            #######################################

            rmin, rmax, cmin, cmax = get_bbox(label, idx, height, width, border_list)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, idx))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            # print("choose: ", len(choose))

            #######################################
            # PROJECT TO SCREEN
            #######################################

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            # print("cam_mat: \n", cam_mat)

            imgpts, jac = cv2.projectPoints(cld[affordance_id] * 1e3,
                                            cam_rotation4,
                                            cam_translation * 1e3,
                                            cam_mat, dist)

            cv2_img_rot = cv2.polylines(np.array(img), np.int32([np.squeeze(imgpts)]), True, (255, 0, 255))
            cv2_gt_rot = cv2.polylines(gray2rgb(gt) * 50, np.int32([np.squeeze(imgpts)]), True, (255, 0, 255))

            cv2.rectangle(cv2_img_rot, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

            #######################################
            # PLOT
            #######################################
            if len(choose) == 0:
                print(rgb_addr)
                bad_choose_files += 1
                ###
            # plt.subplot(2, 2, 1)
            # plt.title('og')
            # plt.imshow(img)
            # plt.subplot(2, 2, 2)
            # plt.title('depth')
            # plt.imshow(depth)
            # plt.subplot(2, 2, 3)
            # plt.title('gt')
            # plt.imshow(cv2_gt_rot)
            # plt.subplot(2, 2, 4)
            # plt.title('project points')
            # plt.imshow(cv2_img_rot)
            # plt.ioff()
            # plt.pause(0.001)
            # plt.show()

        else:
            pass
print("bad_choose_files: ", bad_choose_files)