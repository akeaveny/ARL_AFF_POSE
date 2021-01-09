import glob
import numpy as np
import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

###################################
# ARL
###################################

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
print("class_IDs: ", class_IDs)

label_format = '_label.png'

###################################
# TOOLS
###################################

# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/real_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/real_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/real_test_data_list.txt'

# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/syn_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/syn_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/syn_test_data_list.txt'

###################################
# CLUTTER
###################################

# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_real_clutter_4_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_real_clutter_4_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_real_clutter_4_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/real_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/real_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/real_test_data_list.txt'

# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_test_data_list.txt'

###################################
# test
###################################

data_root = '/data/Akeaveny/Datasets/arl_dataset/'
data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_test_tools_3_test/'
data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_test_clutter_3_test/'
data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_test_clutter_3_test'
#
train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/test/test_tools_data_list.txt'
val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/test/test_clutter_data_list.txt'
test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/test/test_clutter_data_list.txt'

###################################
###################################

scenes = [
           '',
           # '1_bench/',
           # '2_work_bench/',
           # '3_coffee_table/',
           # '4_old_table/',
           # '5_bedside_table/',
           # '6_dr/'
          ]

################################
# TRAIN
################################
saved_files = 0
all_image_files = []
for scene in scenes:

    gt_label_addr = data_path_tools_train + scene + '*' + label_format
    files = sorted(glob.glob(gt_label_addr))
    all_image_files.append(files)

flat_image_list = []
for sublist in all_image_files:
    for item in sublist:
        flat_image_list.append(item)

f_train = open(train_file, 'w')
dataset_mean, dataset_std, dataset_count = 0, 0, 0
# ===================== train ====================
print('\n-------- TRAIN --------')
for file in flat_image_list:

    # print('file: ', file)
    f_train.write(file.split(label_format)[0].split(data_root)[1])
    f_train.write('\n')
    saved_files += 1
print("Actual files: ", saved_files)

# ################################
# # IMAGE STATS
# ################################
# dataset_mean, dataset_std, dataset_count = 0, 0, 0
#
# np.random.seed(0)
# num_random = 100
# random_idx = np.random.choice(np.arange(0, len(flat_image_list), 1), size=int(num_random), replace=False)
#
# for idx in random_idx:
#
#     file = flat_image_list[idx]
#     rgb = file.split(label_format)[0] + "_rgb.png"
#
#     image = cv2.imread(rgb)
#     mean, stddev = cv2.meanStdDev(image)
#     # print("mean: ", mean)
#     #
#     # cv2.imshow("image", image)
#     # cv2.waitKey(1)
#
#     dataset_mean += mean
#     dataset_std += stddev
#     dataset_count += 1
#     saved_files += 1
#
# dataset_mean /= dataset_count
# dataset_std /= dataset_count
# print("\n*** dataset stats (in reverse) ***")
# print("Means (RGB): ", dataset_mean[2], dataset_mean[1], dataset_mean[0])
# print("std: ", dataset_std[2], dataset_std[1], dataset_std[0])

################################
# VAL
################################
saved_files = 0
all_image_files = []
for scene in scenes:

    gt_label_addr = data_path_tools_val + scene + '*' + label_format
    files = sorted(glob.glob(gt_label_addr))
    all_image_files.append(files)

flat_image_list = []
for sublist in all_image_files:
    for item in sublist:
        flat_image_list.append(item)

f_val = open(val_file, 'w')
# ===================== val ====================
print('\n-------- VAL --------')
for file in flat_image_list:

    # print('file: ', file)
    f_val.write(file.split(label_format)[0].split(data_root)[1])
    f_val.write('\n')
    saved_files += 1
print("Actual files: ", saved_files)

################################
# TEST
################################
saved_files = 0
all_image_files = []
for scene in scenes:

    gt_label_addr = data_path_tools_test + scene + '*' + label_format
    files = sorted(glob.glob(gt_label_addr))
    all_image_files.append(files)

flat_image_list = []
for sublist in all_image_files:
    for item in sublist:
        flat_image_list.append(item)

f_test = open(test_file, 'w')
# ===================== TEST ====================
print('\n-------- TEST --------')
for file in flat_image_list:

    # print('file: ', file)
    f_test.write(file.split(label_format)[0].split(data_root)[1])
    f_test.write('\n')
    saved_files += 1
print("Actual files: ", saved_files)