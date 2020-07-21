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


def write_to_json(instance_img, label_img, classes, img_number, folder_to_save, dataset_name, save_rgb):
    # print("Shape: ", img.shape)
    rows, cols = instance_img.shape
    regions = {}
    classes_list = classes
    edge_img = np.zeros((rows, cols), np.uint8)

    # print("String Sequence: ", str_seq)
    obj_name = img_number + dataset_name
    data[obj_name] = {}
    data[obj_name]['fileref'] = ""
    data[obj_name]['size'] = 1280
    data[obj_name]['filename'] = folder_to_save + img_number + save_rgb
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

data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/'
dataset_name = 'Affordance'
save_rgb = '_rgb.png'
# save_rgb = '_fused.png'

class_id = np.arange(0, 205+1, 1)
# print("Affordance IDs: \n{}\n".format(class_id))

# ===================== val ====================
folder_to_save = 'combined_tools_val/turn_table/'
labels = data_path + folder_to_save + '??????' + '_label.png'

files = np.array(sorted(glob.glob(labels)))
print("Loaded files: ", len(files))

num_files = 425
val_idx = np.random.choice(np.arange(0, len(files)+1, 1), size=int(num_files), replace=False)
print("Chosen Files \n", val_idx)
files = files[val_idx]

data = {}
iteration = 0
# ===================== val ====================
print('-------- VAL --------')
json_addr = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb/turn_table/val_425.json'
for file in files:

    str_num = file.split(data_path + folder_to_save)[1]
    img_number = str_num.split('_label.png')[0]
    label_addr = file

    # print("\nIteration: ", iteration)
    print('Image: {}/{}'.format(iteration, len(files)))

    # count = 1000000 + i
    # img_number = str(count)[1:]
    # label_addr = data_path + folder_to_save + img_number + '_label.png'

    # print("img_number: ", img_number)
    print("label_addr: ", label_addr)

    label_img = load_image(label_addr)
    # print("Classes: ", np.unique(label_img))
    # plt.imshow(label_img)
    # plt.show()

    if label_img.size == 0:
        print('\n ------------------ Pass! --------------------')
        pass
    else:
        write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name, save_rgb)
    iteration += 1

with open(json_addr, 'w') as outfile:
    json.dump(data, outfile, sort_keys=True)

# ===================== train ====================
folder_to_save = 'combined_tools_train/turn_table/'
labels = data_path + folder_to_save + '??????' + '_label.png'

files = np.array(sorted(glob.glob(labels)))
print("Loaded files: ", len(files))

num_files = 1000
train_idx = np.random.choice(np.arange(0, len(files)+1, 1), size=int(num_files), replace=False)
print("Chosen Files \n", train_idx)
files = files[train_idx]

data = {}
iteration = 0
# # ===================== train ====================
print('-------- TRAIN --------')
json_addr = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb/turn_table/train_1000.json'
for file in files:

    str_num = file.split(data_path + folder_to_save)[1]
    img_number = str_num.split('_label.png')[0]
    label_addr = file

    # print("\nIteration: ", iteration)
    print('Image: {}/{}'.format(iteration, len(files)))

    # count = 1000000 + i
    # img_number = str(count)[1:]
    # label_addr = data_path + folder_to_save + img_number + '_label.png'

    # print("img_number: ", img_number)
    print("label_addr: ", label_addr)

    label_img = load_image(label_addr)
    # print("Classes: ", np.unique(label_img))
    # plt.imshow(label_img)
    # plt.show()

    if label_img.size == 0:
        print('\n ------------------ Pass! --------------------')
        pass
    else:
        write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name, save_rgb)
    iteration += 1

with open(json_addr, 'w') as outfile:
    json.dump(data, outfile, sort_keys=True)

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
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name, save_rgb)
#     iteration += 1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)

##  ===================== TEST ====================
# folder_to_save = 'combined_tools_test/dr/'
# labels = data_path + folder_to_save+ '??????' + '_label.png'
# dataset_name = 'Affordance'
# save_rgb = '_rgb.png'
# # save_rgb = '_fused.png'
#
# class_id = [0, 1, 2, 3, 4, 5, 6, 7]
#
# max_img = len(sorted(glob.glob(labels)))
# max_img = 10
#
# data = {}
# iteration = 0
# # ===================== TEST ====================
# print('-------- TEST --------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb/dr_test_10.json'
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
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name, save_rgb)
#     iteration += 1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)