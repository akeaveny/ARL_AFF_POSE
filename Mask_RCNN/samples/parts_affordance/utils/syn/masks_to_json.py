import json
import cv2
import glob
import matplotlib.pyplot as plt
import re
import numpy as np
from imantics import Polygons, Mask

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing
stop = True
data = None

def load_image(addr):
    img = cv2.imread(addr, -1)
    # if visual == True:
    #     print(np.unique(img))
    #     # cv2.imshow('img', img)
    #     # cv2.waitKey(100)
    #     plt.imshow(img)
    #     plt.show()
    return img

def is_edge_point(img, row, col):
    rows, cols = img.shape
    value = (int)(img[row, col])
    if value == 0:
        return False
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if row + i >= 0 and row + i < rows and col + j >= 0 and col + j < cols:
                value_neib = (int)(img[row + i, col + j])
                if value_neib == value:
                    count = count + 1
    if count > 2 and count < 8:
        return True
    return False


def edge_downsample(img):
    rows, cols = img.shape
    for row in range(rows):
        for col in range(cols):
            if img[row, col] > 0:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        if i == 0 and j == 0:
                            continue
                        roww = row + i
                        coll = col + j
                        if roww >= 0 and roww < rows and coll >= 0 and coll < cols:
                            if img[roww, coll] == img[row, col]:
                                img[roww, coll] = 0
    return img


def next_edge(img, obj_id, row, col):
    rows, cols = img.shape
    incre = 1
    while (incre < 10):
        for i in range(-incre, incre + 1, 2 * incre):
            for j in range(-incre, incre + 1, 1):
                roww = row + i
                coll = col + j
                if roww >= 0 and roww < rows and coll >= 0 and coll < cols:
                    value = img[roww, coll]
                    if value == obj_id:
                        return True, roww, coll
        for i in range(-incre + 1, incre, 1):
            for j in range(-incre, incre + 1, 2 * incre):
                roww = row + i
                coll = col + j
                if roww >= 0 and roww < rows and coll >= 0 and coll < cols:
                    value = img[roww, coll]
                    if value == obj_id:
                        return True, roww, coll
        incre = incre + 1
    return False, row, col


def find_region(img, classes_label, obj_id, row, col):
    region = {}
    region['region_attributes'] = {}
    region['shape_attributes'] = {}

    rows, cols = img.shape
    roww = row
    coll = col
    edges_x = []
    edges_y = []
    find_edge = True
    poly_img = np.zeros((rows, cols), np.uint8)

    while (find_edge):
        edges_x.append(coll)
        edges_y.append(roww)
        img[roww, coll] = 0
        poly_img[roww, coll] = 255
        find_edge, roww, coll = next_edge(img, obj_id, roww, coll)
        if visual == True:
            cv2.imshow('polygon', poly_img)  # there is a bug here after first image drawing
            cv2.waitKey(1)

    edges_x.append(col)
    edges_y.append(row)
    col_center = sum(edges_x) / len(edges_x)
    row_center = sum(edges_y) / len(edges_y)

    class_id = classes_label[int(row_center), int(col_center)]
    class_id = class_id.item()
    class_id = class_id
    # ======================== CLASS ID ======================
    print("class_id: ", class_id)
    have_object = True
    if class_id == 0:
        have_object = False

    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["all_points_x"] = edges_x
    region['shape_attributes']["all_points_y"] = edges_y
    region['shape_attributes']["class_id"] = class_id

    return region, img, have_object


def write_to_json(instance_img, label_img, classes, img_number, folder_to_save, dataset_name):
    # print("Shape: ", img.shape)
    rows, cols = instance_img.shape
    regions = {}
    classes_list = classes
    edge_img = np.zeros((rows, cols), np.uint8)

    # print("String Sequence: ", str_seq)
    obj_name = img_number + dataset_name
    data[obj_name] = {}
    data[obj_name]['fileref'] = ""
    data[obj_name]['size'] = instance_img.shape[1]
    data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
    data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
    data[obj_name]['base64_img_data'] = ""
    data[obj_name]['file_attributes'] = {}
    data[obj_name]['regions'] = {}

    for row in range(rows):
        for col in range(cols):
            if label_img[row, col] in classes_list:
                if is_edge_point(instance_img, row, col) == True:
                    edge_img[row, col] = instance_img[row, col]
                    # print(edge_img[row, col])

    # edge_img = edge_downsample(edge_img)

    if visual == True:
        plt.imshow(edge_img)
        plt.show()

    instance_ids = []
    # 0 is background
    instance_ids.append(0)

    count = 0
    for row in range(rows):
        for col in range(cols):
            id = edge_img[row, col]
            if id not in instance_ids:
                # print(id)
                region, edge_img, have_obj = find_region(edge_img, label_img, id, row, col)
                if have_obj == True:
                    regions[str(count)] = region
                    count = count + 1
                instance_ids.append(id)

    if count > 0:
        # print("String Sequence: ", str_seq)
        # obj_name = img_number + dataset_name
        # data[obj_name] = {}
        # data[obj_name]['fileref'] = ""
        # data[obj_name]['size'] = 1280
        # # data[obj_name]['filename'] = folder_to_save + img_number + '_fused.png'
        # # data[obj_name]['rgbfilename'] = folder_to_save + img_number + '_rgb.png'
        # data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
        # data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
        # data[obj_name]['base64_img_data'] = ""
        # data[obj_name]['file_attributes'] = {}
        data[obj_name]['regions'] = regions
    return stop
###########################################################
# Manual Config
###########################################################
np.random.seed(1)

dataset_name = 'Affordance'

data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/'

######################
# objects
######################

# json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/scissors/'
# train_path = 'objects/combined_tools_scissors2_train/'
# val_path = 'objects/combined_tools_scissors2_val/'

# json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/objects/hammer1/'
# train_path = 'objects/combined_tools_hammer2_train/'
# val_path = 'objects/combined_tools_hammer2_val/'

# image_ext = '_label.png' ### object ids '_label.png'

# class_id = np.arange(0, 205+1, 1)

######################
# object id
######################

# json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/syn/'
# train_path = 'combined_tools2_train/'
# val_path = 'combined_tools2_val/'

json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/clutter/'
train_path = 'combined_clutter1_train/'
val_path = 'combined_clutter1_val/'

image_ext = '_label.png' ### object ids '_label.png'

class_id = np.arange(0, 205+1, 1)

######################
# affordance
######################

# json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/syn_aff/'
# train_path = 'combined_tools2_train/'
# val_path = 'combined_tools2_val/'

# json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/json/rgb/clutter_aff/'
# train_path = 'combined_clutter1_train/'
# val_path = 'combined_clutter1_val/'
#
# image_ext = '_gt_affordance.png' ### affordances '_gt_affordance.png'
#
# class_id = [0, 1, 2, 3, 4, 5, 6, 7]
#
# print("Affordance IDs: \n{}\n".format(class_id))

VISUALIZE = False

use_random_idx = True
num_val = num_train = 4

# 1.
scenes = [
          # 'bench/',
          # 'floor/',
          'turn_table/',
          # 'dr/'
          ]

#=====================
# JSON FILES
#=====================


# 1.
scenes = [
          'bench/',
          'floor/',
          'turn_table/',
          # 'dr/'
          ]

for scene in scenes:
    print('\n******************** {} ********************'.format(scene))

    ###########################################################
    # VALIDATION
    ###########################################################
    print('\n ------------------ VAL ------------------')

    # =====================
    ### config
    # =====================

    folder_to_save = val_path + scene
    labels = data_path + folder_to_save + '??????' + image_ext

    files = np.array(sorted(glob.glob(labels)))
    print("Loaded files: ", len(files))

    if use_random_idx:
        val_idx = np.random.choice(np.arange(0, len(files)+1, 1), size=int(num_val), replace=False)
        print("Chosen Files \n", val_idx)
        files = files[val_idx]
    else:
        num_val = len(files)

    data = {}
    iteration = 0

    ##################
    ###
    ##################

    json_addr = json_path + scene + 'val_' + np.str(num_val) + '.json'
    for file in files:

        str_num = file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(image_ext)[0]
        label_addr = file

        print("label_addr: ", label_addr)
        print('Image: {}/{}'.format(iteration, len(files)))

        label_img = load_image(label_addr)
        print("GT Affordances:", np.unique(np.array(label_img)))

        if label_img.size == 0:
            print('\n ------------------ Pass! --------------------')
            pass
        else:
            write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)

    ###########################################################
    # TRAIN
    ###########################################################
    print('\n ------------------ TRAIN ------------------')

    # =====================
    ### config
    # =====================

    folder_to_save = train_path + scene
    labels = data_path + folder_to_save + '??????' + image_ext

    files = np.array(sorted(glob.glob(labels)))
    print("Loaded files: ", len(files))

    if use_random_idx:
        train_idx = np.random.choice(np.arange(0, len(files) + 1, 1), size=int(num_train), replace=False)
        print("Chosen Files \n", train_idx)
        files = files[train_idx]
    else:
        num_train = len(files)

    data = {}
    iteration = 0

    ##################
    ###
    ##################

    json_addr = json_path + scene + 'train_' + np.str(num_train) + '.json'
    for file in files:

        str_num = file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(image_ext)[0]
        label_addr = file

        ### print("label_addr: ", label_addr)
        print('Image: {}/{}'.format(iteration, len(files)))

        label_img = load_image(label_addr)
        print("GT Affordances:", np.unique(np.array(label_img)))

        if label_img.size == 0:
            print('\n ------------------ Pass! --------------------')
            pass
        else:
            write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)

    ###########################################################
    # TEST
    ###########################################################
    # print('\n ------------------ TEST ------------------')
    #
    # # =====================
    # ### config
    # # =====================
    #
    # folder_to_save = val_path + scene
    # labels = data_path + folder_to_save + '??????' + image_ext
    #
    # files = np.array(sorted(glob.glob(labels)))
    # print("Loaded files: ", len(files))
    #
    # if use_random_idx:
    #     test_idx = np.random.choice(np.arange(0, len(files) + 1, 1), size=int(num_test), replace=False)
    #     print("Chosen Files \n", test_idx)
    #     files = files[test_idx]
    # else:
    #     num_test = len(files)
    #
    # iteration = 0
    # data = {}
    #
    # # =====================
    # ###
    # # =====================
    #
    # json_addr = json_path + scene + 'test' + np.str(num_test) + '.json'
    # for file in files:
    #
    #     str_num = file.split(data_path + folder_to_save)[1]
    #     img_number = str_num.split(image_ext)[0]
    #     label_addr = file
    #
    #     ### print("label_addr: ", label_addr)
    #     print('Image: {}/{}'.format(iteration, len(files)))
    #
    #     label_img = load_image(label_addr)
    #     print("GT Affordances:", np.unique(np.array(label_img)))
    #
    #     if label_img.size == 0:
    #         print('\n ------------------ Pass! --------------------')
    #         pass
    #     else:
    #         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
    #     iteration += 1
    #
    # with open(json_addr, 'w') as outfile:
    #     json.dump(data, outfile, sort_keys=True)

###########################################################
# MISSING
###########################################################
# print('\n ------------------ MISSING ------------------')

# ===================== MISSING ====================
# folder_to_save = 'combined_missing/'
# labels = data_path + folder_to_save + '??????' + '_label.png'
# max_img = len(sorted(glob.glob(labels)))
#
# data = {}
# iteration = 0
# # ===================== MISSING ====================
# print('-------- TEST --------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/missing/missing.json'
# for i in range(0, max_img):
#     # print("\nIteration: ", iteration)
#     print('Image: {}/{}'.format(iteration, max_img))
#     count = 1000000 + i
#     img_number = str(count)[1:]
#     label_addr = data_path + folder_to_save + img_number + '_label.png'
#
#     # print("img_number: ", img_number)
#     print("label_addr: ", label_addr)
#
#     label_img = load_image(label_addr)
#     print("Classes: ", np.unique(label_img))
#     # plt.imshow(label_img)
#     # plt.show()
#
#     if label_img.size == 0:
#         print('\n ------------------ Pass! --------------------')
#         pass
#     else:
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
#     iteration += 1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)
