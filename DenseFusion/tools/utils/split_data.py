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

# class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_train.txt')
# class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids_train.txt')
class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/classes.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/classes_ids.txt')
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
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_household_tools_1_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_household_tools_1_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_household_tools_1_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/household_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/household_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/tools/household_test_data_list.txt'

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
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_household_clutter_1_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_household_clutter_1_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_household_clutter_1_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/household_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/household_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/household_test_data_list.txt'

# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_clutter_2_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/clutter/syn_test_data_list.txt'

# ###################################
# # FINETUNE
# ###################################
#
# NUM_IMAGES = 1500
#
# data_root = '/data/Akeaveny/Datasets/arl_dataset/'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_household_clutter_1_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_household_clutter_1_val/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/finetune/household_train_1500_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/data_lists/finetune/household_val_1500_data_list.txt'

###################################
# Elevator
###################################

data_root = '/data/Akeaveny/Datasets/elevator_dataset/'
data_path_tools_train = '/data/Akeaveny/Datasets/elevator_dataset/combined_real_tools_1_train/'
data_path_tools_val = '/data/Akeaveny/Datasets/elevator_dataset/combined_real_tools_1_val/'
data_path_tools_test = '/data/Akeaveny/Datasets/elevator_dataset/combined_real_tools_1_test/'

train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/data_lists/elevator_train_data_list.txt'
val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/data_lists/elevator_val_data_list.txt'
test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/elevator/dataset_config/data_lists/elevator_test_data_list.txt'

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
print('\n-------- TRAIN --------')
saved_files = 0
all_image_files = []
for scene in scenes:

    gt_label_addr = data_path_tools_train + scene + '/??????' + label_format
    files = sorted(glob.glob(gt_label_addr))
    all_image_files.append(files)

flat_image_list = []
for sublist in all_image_files:
    for item in sublist:
        flat_image_list.append(item)

# train_idx = np.random.choice(np.arange(0, len(files), 1), size=int(NUM_IMAGES*0.8), replace=False)
# print("Chosen Files ", len(train_idx))
# flat_image_list = np.array(flat_image_list)[train_idx]

f_train = open(train_file, 'w')
# ===================== train ====================
for file in flat_image_list:

    str = file.split(label_format)[0].split(data_root)[1]
    str_num = str.split('/')[-1]
    assert(len(str_num) == 6)

    # print('file: ', file)
    f_train.write(file.split(label_format)[0].split(data_root)[1])
    f_train.write('\n')
    saved_files += 1
print("Actual files: ", saved_files)

################################
# VAL
################################
print('\n-------- VAL --------')
saved_files = 0
all_image_files = []
for scene in scenes:

    gt_label_addr = data_path_tools_val + scene + '/??????' + label_format
    files = sorted(glob.glob(gt_label_addr))
    all_image_files.append(files)

flat_image_list = []
for sublist in all_image_files:
    for item in sublist:
        flat_image_list.append(item)

# val_idx = np.random.choice(np.arange(0, len(files), 1), size=int(NUM_IMAGES*0.2), replace=False)
# print("Chosen Files ", len(val_idx))
# flat_image_list = np.array(flat_image_list)[val_idx]

f_val = open(val_file, 'w')
# ===================== val ====================
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

    gt_label_addr = data_path_tools_test + scene + '/??????' + label_format
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