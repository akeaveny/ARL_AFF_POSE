import glob
import numpy as np
import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

###################################
# UMD
###################################

# class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance/dataset_config/classes_train.txt')
# class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance/dataset_config/class_ids_train.txt')
# class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("class_IDs: ", class_IDs)
#
# # data_path_tools_train = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools2_train/'
# # data_path_tools_val = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools2_val/'
# # data_path_clutter_train = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_clutter1_train/'
# # data_path_clutter_val = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_clutter1_val/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance/dataset_config/train_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance/dataset_config/test_data_list.txt'

# ###################################
# # YCB
# ###################################

# class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/classes_train.txt')
# class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/class_ids_train.txt')
# class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("class_IDs: ", class_IDs)
#
# label_format = '_label.png'
#
# tools_data_folder = 'combined_tools3_'
# data_path_tools_train = '/data/Akeaveny/Datasets/ycb_syn/combined_tools3_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/ycb_syn/combined_tools3_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/ycb_syn/combined_tools3_test/'
#
# clutter_data_folder = 'combined_clutter3_'
# data_path_clutter_train = '/data/Akeaveny/Datasets/ycb_syn/combined_clutter3_train/'
# data_path_clutter_val = '/data/Akeaveny/Datasets/ycb_syn/combined_clutter3_val/'
# data_path_clutter_test = '/data/Akeaveny/Datasets/ycb_syn/combined_clutter3_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/train_data_list_clutter.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/val_data_list_clutter.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb_syn/dataset_config/test_data_list_clutter.txt'

# ###################################
# # ARL REAL
# ###################################
#
# class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/classes_train.txt')
# class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/class_ids_train.txt')
# class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("class_IDs: ", class_IDs)
#
# label_format = '_label.png'
#
# tools_data_folder = 'combined_real_tools1_'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_real_tools1_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_real_tools1_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_real_tools1_test/'
#
# # train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/train_data_list.txt'
# # val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/val_data_list.txt'
# # test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/test_data_list.txt'

# ############
# # COMBINED
# ############
# clutter_data_folder = 'combined_syn_tools2_'
# data_path_clutter_train = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_train/'
# data_path_clutter_val = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_val/'
# data_path_clutter_test = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/train_data_list_combined.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/val_data_list_combined.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/test_data_list_combined.txt'

###################################
# ARL SYN
###################################

# class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_syn/dataset_config/classes_train.txt')
# class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_syn/dataset_config/class_ids_train.txt')
# class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
# print("class_IDs: ", class_IDs)
#
# label_format = '_label.png'
#
# tools_data_folder = 'combined_syn_tools2_'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/combined_syn_tools2_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_syn/dataset_config/train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_syn/dataset_config/val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_syn/dataset_config/test_data_list.txt'

###################################
# ARL
###################################

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/classes_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
print("class_IDs: ", class_IDs)

label_format = '_label.png'

# tools_data_folder = 'combined_real_tools_4_'
# data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_train/'
# data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_val/'
# data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_real_tools_4_test/'
#
# train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/real_train_data_list.txt'
# val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/real_val_data_list.txt'
# test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/real_test_data_list.txt'

tools_data_folder = 'combined_syn_tools_2_'
data_path_tools_train = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_train/'
data_path_tools_val = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_val/'
data_path_tools_test = '/data/Akeaveny/Datasets/arl_dataset/combined_syn_tools_2_test/'

train_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/syn_train_data_list.txt'
val_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/syn_val_data_list.txt'
test_file = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config/syn_test_data_list.txt'

###################################
###################################

scenes = [
           # '',
           '1_bench/',
           '2_work_bench/',
           '3_coffee_table/',
           '4_old_table/',
           '5_bedside_table/',
           '6_dr/'
          ]

################################
# tools
################################
for scene in scenes:

    # ===================== train ====================
    data_path = data_path_tools_train + scene
    folder_to_save = tools_data_folder + 'train/' + scene

    saved_files = 0

    gt_label_addr = data_path + '/??????' + label_format
    files = sorted(glob.glob(gt_label_addr))

    f_train = open(train_file, 'a')
    # ===================== train ====================
    print('\n-------- TRAIN --------')
    for file in files:

        str_num = file.split(data_path)[1]
        img_number = str_num.split(label_format)[0]

        label_addr = data_path + img_number + label_format
        label = np.array(Image.open(label_addr))

        affordance_ids = np.unique(np.array(label))

        for affordance_id in affordance_ids:
            if affordance_id in class_IDs:
                ### print("label_addr: ",label_addr)
                img_index_str = label_addr.split(folder_to_save)[1]
                img_index_str = img_index_str.split(label_format)[0]
                f_train.write(folder_to_save + img_index_str)
                f_train.write('\n')
                saved_files += 1
    f_train.close

    print("Loaded files: ", len(files))
    print("Actual files: ", saved_files)

    # ===================== val ====================
    data_path = data_path_tools_val + scene
    folder_to_save = tools_data_folder + 'val/' + scene

    saved_files = 0

    gt_label_addr = data_path + '/??????' + label_format
    files = sorted(glob.glob(gt_label_addr))

    f_val = open(val_file, 'a')
    # ===================== val ====================
    print('\n-------- VAL --------')
    for file in files:

        str_num = file.split(data_path)[1]
        img_number = str_num.split(label_format)[0]

        label_addr = data_path + img_number + label_format
        label = np.array(Image.open(label_addr))

        affordance_ids = np.unique(np.array(label))

        for affordance_id in affordance_ids:
            if affordance_id in class_IDs:
                # print("label_addr: ",label_addr)
                img_index_str = label_addr.split(folder_to_save)[1]
                img_index_str = img_index_str.split(label_format)[0]
                f_val.write(folder_to_save + img_index_str)
                f_val.write('\n')
                saved_files += 1
    f_val.close

    print("Loaded files: ", len(files))
    print("Actual files: ", saved_files)

    # =========================================
    # TEST
    # =========================================
    data_path = data_path_tools_test + scene
    folder_to_save = tools_data_folder + 'test/' + scene

    saved_files = 0

    gt_label_addr = data_path + '/??????' + label_format
    files = sorted(glob.glob(gt_label_addr))

    f_test = open(test_file, 'a')

    # =========================================
    # =========================================
    print('\n-------- TEST --------')
    for file in files:

        str_num = file.split(data_path)[1]
        img_number = str_num.split(label_format)[0]

        label_addr = data_path + img_number + label_format
        label = np.array(Image.open(label_addr))

        affordance_ids = np.unique(np.array(label))

        for affordance_id in affordance_ids:
            if affordance_id in class_IDs:
                # print("label_addr: ",label_addr)
                img_index_str = label_addr.split(folder_to_save)[1]
                img_index_str = img_index_str.split(label_format)[0]
                f_test.write(folder_to_save + img_index_str)
                f_test.write('\n')
                saved_files += 1
    f_test.close

    print("Loaded files: ", len(files))
    print("Actual files: ", saved_files)

################################
################################

# scenes = [
#           'turn_table/', 'bench/', 'floor/',
#           'dr/'
#           ]
#
# ################################
# # clutter
# ################################
# for scene in scenes:
#
#     # ===================== train ====================
#     data_path = data_path_clutter_train + scene
#     folder_to_save = clutter_data_folder + 'train/' + scene
#
#     saved_files = 0
#
#     gt_label_addr = data_path + '/??????' + label_format
#     files = sorted(glob.glob(gt_label_addr))
#
#     f_train = open(train_file, 'a')
#     # ===================== train ====================
#     print('\n-------- TRAIN --------')
#     for file in files:
#
#         str_num = file.split(data_path)[1]
#         img_number = str_num.split(label_format)[0]
#
#         label_addr = data_path + img_number + label_format
#         label = np.array(Image.open(label_addr))
#
#         affordance_ids = np.unique(np.array(label))
#
#         for affordance_id in affordance_ids:
#             if affordance_id in class_IDs:
#                 ### print("label_addr: ",label_addr)
#                 img_index_str = label_addr.split(folder_to_save)[1]
#                 img_index_str = img_index_str.split(label_format)[0]
#                 f_train.write(folder_to_save + img_index_str)
#                 f_train.write('\n')
#                 saved_files += 1
#     f_train.close
#
#     print("Loaded files: ", len(files))
#     print("Actual files: ", saved_files)

    # # ===================== val ====================
    # data_path = data_path_clutter_val + scene
    # folder_to_save = clutter_data_folder + 'val/' + scene
    #
    # saved_files = 0
    #
    # gt_label_addr = data_path + '/??????' + label_format
    # files = sorted(glob.glob(gt_label_addr))
    #
    # f_val = open(val_file, 'a')
    # # ===================== val ====================
    # print('\n-------- VAL --------')
    # for file in files:
    #
    #     str_num = file.split(data_path)[1]
    #     img_number = str_num.split(label_format)[0]
    #
    #     label_addr = data_path + img_number + label_format
    #     label = np.array(Image.open(label_addr))
    #
    #     affordance_ids = np.unique(np.array(label))
    #
    #     for affordance_id in affordance_ids:
    #         if affordance_id in class_IDs:
    #             # print("label_addr: ",label_addr)
    #             img_index_str = label_addr.split(folder_to_save)[1]
    #             img_index_str = img_index_str.split(label_format)[0]
    #             f_val.write(folder_to_save + img_index_str)
    #             f_val.write('\n')
    #             saved_files += 1
    # f_val.close
    #
    # print("Loaded files: ", len(files))
    # print("Actual files: ", saved_files)
    #
    # # =========================================
    # # TEST
    # # =========================================
    # data_path = data_path_clutter_test + scene
    # folder_to_save = clutter_data_folder + 'test/' + scene
    #
    # saved_files = 0
    #
    # gt_label_addr = data_path + '/??????' + label_format
    # files = sorted(glob.glob(gt_label_addr))
    #
    # f_test = open(test_file, 'a')
    #
    # # =========================================
    # # =========================================
    # print('\n-------- TEST --------')
    # for file in files:
    #
    #     str_num = file.split(data_path)[1]
    #     img_number = str_num.split(label_format)[0]
    #
    #     label_addr = data_path + img_number + label_format
    #     label = np.array(Image.open(label_addr))
    #
    #     affordance_ids = np.unique(np.array(label))
    #
    #     for affordance_id in affordance_ids:
    #         if affordance_id in class_IDs:
    #             # print("label_addr: ",label_addr)
    #             img_index_str = label_addr.split(folder_to_save)[1]
    #             img_index_str = img_index_str.split(label_format)[0]
    #             f_test.write(folder_to_save + img_index_str)
    #             f_test.write('\n')
    #             saved_files += 1
    # f_test.close
    #
    # print("Loaded files: ", len(files))
    # print("Actual files: ", saved_files)