import glob
import numpy as np
import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/class_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

# ===================== SETUP ====================
# 2.
scenes = [
          'turn_table/', 'bench/', 'floor/',
          'dr/'
          ]

for scene in scenes:

    # ===================== train ====================
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_train/' + scene
    folder_to_save = 'combined_tools_train/' + scene
    label_format = '_label.png'

    gt_label_addr = data_path + '/??????' + '_label.png'
    files = sorted(glob.glob(gt_label_addr))
    print("Loaded files: ", len(files))

    f_train = open("train_data_list.txt", 'a')
    # ===================== train ====================
    print('-------- TRAIN --------')
    for file in files:

        str_num = file.split(data_path)[1]
        img_number = str_num.split('_label.png')[0]

        label_addr = data_path + img_number + label_format

        meta = scio.loadmat('{0}/{1}-meta.mat'.format(data_path, img_number))
        affordance_ids = np.array(meta['Affordance_ID'].astype(np.int32))

        for affordance_id in affordance_ids:
            if affordance_id in class_IDs:
                print("label_addr: ",label_addr)
                img_index_str = label_addr.split(folder_to_save)[1]
                img_index_str = img_index_str.split(label_format)[0]
                f_train.write(folder_to_save + img_index_str)
                f_train.write('\n')
    f_train.close

    # ===================== val ====================
    data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_val/' + scene
    folder_to_save = 'combined_tools_val/' + scene
    label_format = '_label.png'

    gt_label_addr = data_path + '/??????' + '_label.png'
    files = sorted(glob.glob(gt_label_addr))
    print("Loaded files: ", len(files))

    f_val = open("test_data_list.txt", 'a')
    # ===================== val ====================
    print('-------- VAL --------')
    for file in files:

        str_num = file.split(data_path)[1]
        img_number = str_num.split('_label.png')[0]

        label_addr = data_path + img_number + label_format

        meta = scio.loadmat('{0}/{1}-meta.mat'.format(data_path, img_number))
        affordance_ids = np.array(meta['Affordance_ID'].astype(np.int32))

        for affordance_id in affordance_ids:
            if affordance_id in class_IDs:
                print("label_addr: ",label_addr)
                img_index_str = label_addr.split(folder_to_save)[1]
                img_index_str = img_index_str.split(label_format)[0]
                f_val.write(folder_to_save + img_index_str)
                f_val.write('\n')
    f_val.close