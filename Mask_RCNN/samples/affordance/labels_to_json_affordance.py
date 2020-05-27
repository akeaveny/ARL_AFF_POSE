import json
import cv2
import glob
import matplotlib.pyplot as plt
import re
import numpy as np
from imantics import Polygons, Mask

visual = True  # only use True with 1 image for testing because there is a bug in openCV drawing
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

    for row in range(rows):
        for col in range(cols):
            if label_img[row, col] in classes_list:
                if is_edge_point(instance_img, row, col) == True:
                    edge_img[row, col] = instance_img[row, col]
                    # print(edge_img[row, col])

    edge_img = edge_downsample(edge_img)

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
        obj_name = img_number + dataset_name
        data[obj_name] = {}
        data[obj_name]['fileref'] = ""
        data[obj_name]['size'] = 640
        data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
        data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
        data[obj_name]['base64_img_data'] = ""
        data[obj_name]['file_attributes'] = {}
        data[obj_name]['regions'] = regions
    return stop

# # ===================== train  ====================
# data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/formatted_syn_train/'
# folder_to_save = 'ndds_and_real/formatted_syn_train/'
# dataset_name = 'Affordance'
#
# if data_path[len(data_path) - 1] != '/':
#     print(data_path)
#     print('The data path should have / in the end')
#     exit()
#
# class_id = [0, 1, 2]
#
# min_img = 1791
# max_img = 1791 + 100
# img_list = np.arange(1791, 1791+100+1, 10)
#
# data = {}
# count = 0
# # ===================== json ====================
# print('-------- TRAIN --------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_syn_test.json'
# # for i in range(min_img, max_img + 1):
# for i in img_list:
#     print('\nImage: {}/{}'.format(i, max_img))
#     count = 100000 + i
#     img_number = str(count)[1:]
#     label_addr = data_path + img_number + '_label.png'
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
#     count += 1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)
#
# # ===================== val  ====================
# data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/formatted_real_val/'
# folder_to_save = 'ndds_and_real/formatted_real_val/'
# dataset_name = 'Affordance'
#
# if data_path[len(data_path) - 1] != '/':
#     print(data_path)
#     print('The data path should have / in the end')
#     exit()
#
# class_id = [0, 1, 2]
#
# min_img = 0
# max_img = 319
# img_list = np.arange(0, 319+1, 10)
# print(img_list)
#
# data = {}
# count = 0
# # ===================== json ====================
# print('-------- VAL --------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_real_test.json'
# # for i in range(min_img, max_img + 1):
# for i in img_list:
#     print('\nImage: {}/{}'.format(i, max_img))
#     count = 100000 + i
#     img_number = str(count)[1:]
#     label_addr = data_path + img_number + '_label.png'
#
#     # print("img_number: ", img_number)
#     print("label_addr: ", label_addr)
#
#     label_img = load_image(label_addr)
#     # print("Classes: ", np.unique(label_img))
#     # plt.imshow(label_img)
#     # plt.show()
#
#     if label_img.size == 0:
#         print('\n ------------------ Pass! --------------------')
#         pass
#     else:
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
#     count += 1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)

# # ===================== TEST ====================
# data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/formatted_syn_train/'
# folder_to_save = 'ndds_and_real/formatted_real_val/'
# dataset_name = 'Affordance'
#
# if data_path[len(data_path) - 1] != '/':
#     print(data_path)
#     print('The data path should have / in the end')
#     exit()
#
# class_id = [0, 1, 2]
#
# img_list = np.arange(2000, 2000+200+1, 10)
#
# data = {}
# count = 0
# # ===================== json ====================
# print('-------- VAL --------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/test.json'
# # for i in range(min_img, max_img):
# for i in img_list:
#     print('\nImage: {}/{}'.format(i, img_list[-1]))
#     count = 100000 + i
#     img_number = str(count)[1:]
#     label_addr = data_path + img_number + '_label.png'
#
#     # # print("img_number: ", img_number)
#     print("label_addr: ", label_addr)
#
#     label_img = load_image(label_addr)
#     # print("Classes: ", np.unique(label_img))
#     # plt.imshow(label_img)
#     # plt.show()
#
#     ''' ============ imantics ============ '''
#     # print(np.unique(label_img))
#     # polygons = Mask(label_img).polygons()
#     # # print(polygons.points)
#     # # print(polygons.segmentation)
#     # x = polygons.mask()
#     # plt.imshow(x)
#
#     if label_img.size == 0:
#         print('\n ------------------ Pass! --------------------')
#         pass
#     else:
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
#     count += 1
#
#     # ================
#     # contours = find_contours(label_img, 0)
#     # # print(contours)
#     #
#     # fig, ax = plt.subplots()
#     # ax.imshow(label_img, cmap=plt.cm.gray)
#     #
#     # for n, contour in enumerate(contours):
#     #     print('\n', contour[:, 1], contour[:, 0])
#     #     ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
#     # plt.show()
#
#     ''' ============ CV2 ============ '''
#     # height, width = label_img.shape
#     # contours, _ = cv2.findContours(label_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # masks = []
#     # for contour in contours:
#     #     print(contour)
#     #     img = np.zeros((height, width))
#     #     cv2.fillPoly(img, pts=[contour], color=(255, 255, 255))
#     #     masks.append(img)
#     #     plt.imshow(np.squeeze(img))
#     #     plt.show()
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)