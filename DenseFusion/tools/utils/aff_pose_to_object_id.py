import cv2
import glob
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

from skimage.color import gray2rgb

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids_train.txt')
AFF_IDs = np.loadtxt(class_id_file, dtype=np.int32)

cld = {}
print("***** Aff 3D Models *****")
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

object_id_class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl1/dataset_config/classes.txt')
object_id_class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl1/dataset_config/classes_ids.txt')
object_id_class_IDs = np.loadtxt(object_id_class_id_file, dtype=np.int32)

object_id_cld = {}
print("\n***** Object 3D Models *****")
for class_id in object_id_class_IDs:
    print("class_id: ", class_id)
    class_input = object_id_class_file.readline()
    print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/arl_dataset/models/{0}.xyz'.format(class_input[:-1]))
    object_id_cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        object_id_cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    object_id_cld[class_id] = np.array(object_id_cld[class_id])
    input_file.close()

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes.txt')
object_id_class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl1/dataset_config/classes.txt')
class_full = np.loadtxt(class_file, dtype=np.str)
object_id_full = np.loadtxt(object_id_class_file, dtype=np.str)

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
##################################

def map_affordance_label(current_id):

    # 1
    mallet = [
        1, # 'mallet_1_grasp'
        2, # 'mallet_4_pound'
    ]

    spatula = [
        3,  # 'spatula_1_grasp'
        4,  # 'spatula_2_support'
    ]

    wooden_spoon = [
        5,  # 'wooden_spoon_1_grasp'
        6,  # 'wooden_spoon_3_scoop'
    ]

    screwdriver = [
        7,  # 'screwdriver_1_grasp'
        8,  # 'screwdriver_2_screw'
    ]

    garden_shovel = [
        9,  # 'garden_shovel_1_grasp'
        10,  # 'garden_shovel_3_scoop'
    ]

    if current_id in mallet:
        return 1
    elif current_id in spatula:
        return 2
    elif current_id in wooden_spoon:
        return 3
    elif current_id in screwdriver:
        return 4
    elif current_id in garden_shovel:
        return 5
    else:
        print(" --- Object ID does not map to Affordance Label --- ")
        exit(1)

##################################
## LOAD IMAGES
##################################
dataset = '/data/Akeaveny/Datasets/arl_dataset/'
dataset_config = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/combined/'
### real
# images_file = 'real_train_data_list.txt'
# images_file = 'real_val_data_list.txt'
images_file = 'real_test_data_list.txt'
### real
# images_file = 'syn_train_data_list.txt'
# images_file = 'syn_val_data_list.txt'
### combined
# images_file = 'combined_train_data_list.txt'
# images_file = 'combined_val_data_list.txt'

# loaded_images_ = np.loadtxt('{}/{}'.format(dataset_config, images_file), dtype=np.str)
text_file = open('{}/{}'.format(dataset_config, images_file), "r")
loaded_images_ = text_file.readlines()

bad_choose_files = 0

# select random test images
np.random.seed(0)
num_test = len(loaded_images_)
test_idx = np.random.choice(np.arange(0, int(len(loaded_images_)), 1), size=int(num_test), replace=False)
print("Loaded Files: ", len(test_idx))

for file_num, idx in enumerate(test_idx):

    print("File: {0}/{1}".format(file_num, len(test_idx)))

    ##############
    ##############

    str_num = loaded_images_[idx].split('/')[-1]

    rgb_addr = dataset + loaded_images_[idx].rstrip() + "_rgb.png"
    depth_addr = dataset + loaded_images_[idx].rstrip() + "_depth.png"
    gt_addr = dataset + loaded_images_[idx].rstrip() + "_label.png"
    mask_addr = gt_addr

    # gt pose
    meta_addr = dataset + loaded_images_[idx].rstrip() + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    meta1 = meta.copy()
    move_mat_name = depth_addr = dataset + loaded_images_[idx].rstrip() + '-meta.mat'

    cv2_img = np.array(img).copy()
    cv2_gt = gray2rgb(gt).copy()

    # rgb
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

    affordance_ids = np.unique(np.array(label))
    for affordance_id in affordance_ids:
        if affordance_id in AFF_IDs:

            ############
            # GRASP
            ############

            idx = affordance_id
            count = 1000 + idx
            meta_idx = str(count)[1:]

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

            grasp_rotation = np.array(meta['rot' + meta_idx])
            grasp_translation = np.array(meta['cam_translation' + meta_idx][0])  # in [m]

            #######################################
            #######################################

            rmin, rmax, cmin, cmax = get_bbox(label, idx, height, width, border_list)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, idx))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            #######################################
            # PROJECT TO SCREEN
            #######################################

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            # print("cam_mat: \n", cam_mat)

            imgpts1, jac = cv2.projectPoints(cld[affordance_id] * 1e3,
                                            grasp_rotation,
                                            grasp_translation * 1e3,
                                            cam_mat, dist)

            cv2_img = cv2.polylines(cv2_img, np.int32([np.squeeze(imgpts1)]), True, (255, 0, 255))
            # cv2.rectangle(cv2_img, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

            cv2_gt = cv2.polylines(cv2_gt, np.int32([np.squeeze(imgpts1)]), True, (255, 0, 255))

            ##################
            # OTHER AFF ID
            ##################

            idx = affordance_id + 1

            count = 1000 + idx
            meta_idx = str(count)[1:]

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

            aff_rotation = np.array(meta['rot' + meta_idx])
            aff_translation = np.array(meta['cam_translation' + meta_idx][0])  # in [m]

            #######################################
            # PROJECT TO SCREEN
            #######################################

            imgpts2, jac = cv2.projectPoints(cld[affordance_id+1] * 1e3,
                                            aff_rotation,
                                            aff_translation * 1e3,
                                            cam_mat, dist)

            cv2_img = cv2.polylines(cv2_img, np.int32([np.squeeze(imgpts2)]), True, (255, 0, 255))
            # cv2.rectangle(cv2_img, (cmin, rmin), (cmax, rmax), (255, 0, 0), 2)

            cv2_gt = cv2.polylines(cv2_gt, np.int32([np.squeeze(imgpts2)]), True, (255, 0, 255))

            ##################
            # OBJECT ID
            ##################

            object_id = map_affordance_label(affordance_id)
            # print("\naff idx: ", affordance_id, " object_id: ", object_id)
            # print("aff idx: ", class_full[affordance_id - 1], " object: ", object_id_full[object_id - 1])
            object_id_idx = str(1000 + object_id)[1:]

            object_id_cld_2D = np.squeeze(np.array(np.vstack((imgpts1,imgpts2)), dtype=np.float32))
            object_id_cld_3D = np.squeeze(np.array(object_id_cld[object_id], dtype=np.float32))
            # print("imgpnts: {0}, cld: {1}".format(len(imgpts1)+len(imgpts2), len(object_id_cld_3D)))
            assert(len(object_id_cld_2D) == len(object_id_cld_3D))

            grasp_rvec, _ = cv2.Rodrigues(grasp_rotation)

            _, rvec, tvec = cv2.solvePnP(objectPoints=object_id_cld_3D, imagePoints=object_id_cld_2D,
                                         cameraMatrix=cam_mat, distCoeffs=dist,
                                         rvec=grasp_rvec,
                                         tvec=grasp_translation,
                                         useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
            R, _ = cv2.Rodrigues(rvec)
            T = tvec.reshape(-1)

            # print("T\nPred: {0} [cm]\nGT: {1} [cm]".format(T_pred, T_gt))
            # print("R\nPred: {0} [cm]\nGT: {1} [cm]".format(R_pred, R_gt))

            ############################
            # Save mat!!!
            ############################

            meta1['object_id_width' + np.str(object_id_idx)] = width
            meta1['object_id_height' + np.str(object_id_idx)] = height

            meta1['object_id_cx' + np.str(object_id_idx)] = cam_cx
            meta1['object_id_cy' + np.str(object_id_idx)] = cam_cy
            meta1['object_id_fx' + np.str(object_id_idx)] = cam_fx
            meta1['object_id_fy' + np.str(object_id_idx)] = cam_fy

            meta1['object_id_camera_scale' + np.str(object_id_idx)] = cam_scale
            meta1['object_id_border' + np.str(object_id_idx)] = border_list

            meta1['object_id_rot' + np.str(object_id_idx)] = R
            meta1['object_id_cam_translation' + np.str(object_id_idx)] = T

            ############################
            # Error Metrics
            ############################

            T_pred, T_gt = grasp_translation, T
            R_pred, R_gt = grasp_rotation, R

            # translation
            T_error = np.linalg.norm(T_pred - T_gt) * 100 # im [cm]

            # rot
            error_cos = 0.5 * (np.trace(R_pred @ np.linalg.inv(R_gt)) - 1.0)
            error_cos = min(1.0, max(-1.0, error_cos))
            error = np.arccos(error_cos)
            R_error = 180.0 * error / np.pi

            # print("T: {:.2f} [cm]".format(T_error))
            # print("Rot: {:.2f} [deg]".format(R_error))

            imgpts, jac = cv2.projectPoints(object_id_cld_3D*1e3,
                                            R,
                                            tvec*1e3,
                                            cam_mat, dist)

            cv2_img = cv2.polylines(cv2_img, np.int32([np.squeeze(imgpts)]), True, (255, 255, 0))

            cv2_gt = cv2.polylines(cv2_gt, np.int32([np.squeeze(imgpts)]), True, (255, 255, 0))

            #######################################
            # PLOT
            #######################################
            # plt.subplot(2, 2, 1)
            # plt.title('og')
            # plt.imshow(img)
            # plt.subplot(2, 2, 2)
            # plt.title('depth')
            # plt.imshow(depth)
            # plt.subplot(2, 2, 3)
            # plt.title('gt')
            # plt.imshow(cv2_gt)
            # plt.subplot(2, 2, 4)
            # plt.title('project points')
            # plt.imshow(cv2_img)
            # plt.ioff()
            # plt.pause(0.001)
            # plt.show()

        else:
            pass

    # cv2.imshow("cv2_gt", cv2_gt)
    # cv2.imshow("cv2_img", cv2_img)
    # cv2.waitKey(1)

    scio.savemat(move_mat_name, meta1)

print("bad_choose_files: ", bad_choose_files)