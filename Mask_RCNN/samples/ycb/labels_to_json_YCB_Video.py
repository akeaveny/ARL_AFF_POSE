import json
import cv2
import glob
import matplotlib.pyplot as plt
import re
import numpy as np

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing

data = {}
stop = True

debug = False

def load_image(addr):
    img = cv2.imread(addr, -1)
    if visual == True:
        print(np.unique(img))
        # cv2.imshow('img', img)
        # cv2.waitKey(100)
        plt.imshow(img)
        plt.show()
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
            cv2.waitKey(3)

    edges_x.append(col)
    edges_y.append(row)
    col_center = sum(edges_x) / len(edges_x)
    row_center = sum(edges_y) / len(edges_y)

    if debug:
        print(col_center)
        print(row_center)

    class_id = 0
    class_id = classes_label[int(row_center), int(col_center)]
    class_id = class_id.item()
    have_object = True
    if class_id == 0:
        have_object = False

    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["all_points_x"] = edges_x
    region['shape_attributes']["all_points_y"] = edges_y
    region['shape_attributes']["class_id"] = class_id

    if debug:
        print("class_id: ", class_id)
        print("classes_label: ", classes_label.shape)

    return region, img, have_object


def write_to_json(instance_img, label_img, img_number, folder_to_save):
    rows, cols = instance_img.shape
    regions = {}
    classes_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
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
        obj_name = img_number + 'ycb'
        data[obj_name] = {}
        data[obj_name]['fileref'] = ""
        data[obj_name]['size'] = 1024
        data[obj_name]['filename'] = folder_to_save + img_number + '-color' + '.png'
        data[obj_name]['depthfilename'] = folder_to_save + img_number + '-depth' + '.png'
        data[obj_name]['base64_img_data'] = ""
        data[obj_name]['file_attributes'] = {}
        data[obj_name]['regions'] = regions
    return stop


# # ===================== Real data ====================
#
# # data_path = '/data/Akeaveny/Datasets/YCB_Video_Dataset/data/'
#
# data_path = '/data/Akeaveny/Datasets/YCB_Video_Dataset/data_combined/'
# json_addr = '/data/Akeaveny/Datasets/YCB_Video_Dataset/via_region_data_combined.json'
#
# if data_path[len(data_path)-1] != '/':
#     print(data_path)
#     print('The data path should have / in the end')
#     exit()
#
# min_seq = 0
# max_seq = 9
# for k in range(min_seq, max_seq):
#     data.clear()
#     seq = 10000 + k
#     str_seq = str(seq)[1:]
#     label_path = data_path + str_seq + '/'
#     label_addrs = glob.glob(label_path + '*-label.png')
#
#     if debug:
#         print(str_seq)
#         print(data_path)
#         # print(label_addrs)
#
#     # for i in range(len(label_addrs)):
#     for i in range(3):
#         print('Seq, Image: {}/{} {}/{}'.format(k, max_seq, i, len(label_addrs)))
#         pos1 = len(label_path)
#         pos2 = len(label_path) + 6
#         img_number = label_addrs[i][pos1:pos2]
#
#         if debug:
#             print(img_number)
#
#         label_img = load_image(label_addrs[i])
#         write_to_json(label_img, label_img, img_number, str_seq)
#
#     # json_addr = '/data/Akeaveny/Datasets/YCB_Video_Dataset/via_region_data.json'
#     with open(json_addr, 'w') as outfile:
#         json.dump(data, outfile, sort_keys=True)

# # ===================== Real data ====================

data_path = '/data/Akeaveny/Datasets/YCB_Video_Dataset/train_posecnn/syn/'
json_addr = '/data/Akeaveny/Datasets/YCB_Video_Dataset/via_region_data_posecnn.json'
folder_to_save = 'train_posecnn/syn/'

if data_path[len(data_path) - 1] != '/':
    print(data_path)
    print('The data path should have / in the end')
    exit()

# =============== TRAIN =============
# MIN: 11-color
# MAX: 609-label

# min_img = 10
# max_img = 16
#
# for i in range(min_img, max_img):
#     if np.str(i).endswith('0'):
#         pass
#     else:
#         print("End index: ", i)
#         print('Image: {}/{}'.format(i, max_img))
#         img_number = str(i)
#         label_addr = data_path + img_number + '-label.png'
#
#         if debug:
#             print("img_number: ", img_number)
#             print("label_addr: ", label_addr)
#
#         label_img = load_image(label_addr)
#         write_to_json(label_img, label_img, img_number, folder_to_save)
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)

# ===================== Synthetic data ====================

min_img = 1
# max_img = 16
max_img = 79999

for i in range(min_img, max_img):
    print('Image: {}/{}'.format(i, max_img))
    count = 1000000 + i
    img_number = str(count)[1:]
    label_addr = data_path + img_number + '-label.png'

    # if debug:
    #     print("img_number: ", img_number)
    #     print("label_addr: ", label_addr)

    label_img = load_image(label_addr)
    write_to_json(label_img, label_img, img_number, folder_to_save)

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)