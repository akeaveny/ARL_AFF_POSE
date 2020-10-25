import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

NDDS_DEPTH_CONST = 10e3 / (2 ** 8 - 1)

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/class_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

cld = {}
for class_id in class_IDs:
    print("class_id: ", class_id)
    class_input = class_file.readline()
    print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/ycb_syn/models/{0}/grasp_{0}.xyz'.format(class_input[:-1]))
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
## LOAD SYN IMAGES
##################################
dataset = '/data/Akeaveny/Datasets/ycb_syn/'
dataset_config = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/'
images_file = 'train_data_list.txt' ### or 'train_data_list.txt'

loaded_images_ = np.loadtxt('{}/{}'.format(dataset_config, images_file), dtype=np.str)

# select random test images
np.random.seed(0)
num_test = 100
test_idx = np.random.choice(np.arange(0, int(len(loaded_images_)), 1), size=int(num_test), replace=False)
print("Chosen Files \n", test_idx)

for idx in test_idx:

    ##############
    # NDDS
    ##############

    # print("Image Info: ", loaded_images_[idx].split('/'))
    str_num = loaded_images_[idx].split('/')[-1]

    rgb_addr = dataset + loaded_images_[idx] + "_rgb.png"
    depth_addr = dataset + loaded_images_[idx] + "_depth.png"
    gt_addr = dataset + loaded_images_[idx] + "_label.png"
    mask_addr = gt_addr

    print("rgb addr: ", rgb_addr)

    # gt pose
    meta_addr = dataset + loaded_images_[idx] + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    depth_scaled = depth * NDDS_DEPTH_CONST
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    # rgb
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

##################################
# Manual
##################################
# data_path = '/data/Akeaveny/Datasets/parts_affordance/combined_tools2_train/bench'
# folder_to_save = 'combined_tools_val/'
# label_format = '_label.png'
#
# gt_label_addr = data_path + '/??????' + '_label.png'
# files = np.array(sorted(glob.glob(gt_label_addr)))
# print("Loaded files: ", len(files))
#
# num_random = 1000
# random_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_random), replace=False)
# print("Num Images: ", len(files[random_idx]))
#
# for file in files[random_idx]:
#     # count = 1000000 + image_idx
#     # img_number = str(count)[1:]
#
#     str_num = file.split(data_path + '/')[1]
#     img_number = str_num.split('_label.png')[0]
#
#     label_addr = data_path + img_number + label_format
#
#     img = Image.open('{0}/{1}_rgb.png'.format(data_path, img_number))
#     # depth = np.array(Image.open('{0}/{1}_depth.png'.format(data_path, img_number)))
#     label = np.array(Image.open('{0}/{1}_label.png'.format(data_path, img_number)))
#     meta = scio.loadmat('{0}/{1}-meta.mat'.format(data_path, img_number))
#
#     ##################################

    # affordance_ids = np.array(meta['Affordance_ID'].astype(np.int32))
    affordance_ids = np.unique(np.array(label))

    for affordance_id in affordance_ids:

        if affordance_id in class_IDs:

            idx = affordance_id
            count = 1000 + idx
            idx = str(count)[1:]

            camera_setting = meta['camera_setting' + idx][0]
            print("camera_setting: ", camera_setting)

            # if camera_setting == 'ZED':

            # print("depth min: ", np.min(np.array(depth)))
            # print("depth max: ", np.max(np.array(depth)))
            #
            # print("rgb type: ", img.dtype)
            # print("depth type: ", depth.dtype)
            #
            # print("rgb shape: ", img.shape)
            # print("depth shape: ", depth.shape)

            # ================== meta ================
            cam_rotation4 = np.dot(np.array(meta['rot' + np.str(idx)]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            cam_rotation4 = np.dot(cam_rotation4.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

            cam_translation = np.array(meta['cam_translation' + np.str(idx)][0]) / 1e3 / 10 # in cm

            cam_cx = meta['cx' + np.str(idx)][0][0]
            cam_cy = meta['cy' + np.str(idx)][0][0]
            cam_fx = meta['fx' + np.str(idx)][0][0]
            cam_fy = meta['fy' + np.str(idx)][0][0]

            # ===================== SCREEN POINTS =====================
            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            print("cam_mat: \n", cam_mat)

            imgpts, jac = cv2.projectPoints(cld[affordance_id] * 1e2,
                                            cam_rotation4,
                                            cam_translation * 1e3,
                                            cam_mat, dist)
            cv2_img_rot = cv2.polylines(np.array(img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

            # # ========== plot ============
            # plt.subplot(111)
            # plt.title('og')
            # plt.imshow(cv2_img_rot)
            # plt.ioff()
            # plt.pause(0.001)
            # plt.show()

            ##########################
            ### plot
            ##########################
            plt.figure(0)
            plt.subplot(1, 3, 1)
            plt.title('og')
            plt.imshow(cv2_img_rot)
            plt.subplot(1, 3, 2)
            plt.title("og depth")
            plt.imshow(depth)
            print("Min:\t{}, Max:\t{}".format(np.min(depth), np.max(depth)))
            plt.subplot(1, 3, 3)
            plt.title("depth scaled")
            plt.imshow(depth_scaled)
            print("Min:\t{}, Max:\t{}".format(np.min(depth_scaled), np.max(depth_scaled)))
            plt.show()
            plt.ioff()

        else:
            pass
