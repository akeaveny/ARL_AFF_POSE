import json
import cv2
import glob
import matplotlib.pyplot as plt
import re
import numpy as np

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing

data = {}
stop = True

debug = True

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

    class_id = 0
    class_id = classes_label[int(row_center), int(col_center)]
    class_id = class_id.item()
    class_id = 1
    have_object = True
    if class_id == 0:
        have_object = False

    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["all_points_x"] = edges_x
    region['shape_attributes']["all_points_y"] = edges_y
    region['shape_attributes']["class_id"] = class_id

    # print("class_id: ", class_id)
    # print("classes_label: ", classes_label.shape)

    return region, img, have_object


def write_to_json(instance_img, label_img, classes, img_number, folder_to_save):
    # print("Shape: ", img.shape)
    rows, cols = instance_img.shape
    regions = {}
    classes_list = [classes]
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
        obj_name = img_number + 'pringles'
        data[obj_name] = {}
        data[obj_name]['fileref'] = ""
        # data[obj_name]['size'] = 1024
        data[obj_name]['size'] = 2000
        data[obj_name]['filename'] = folder_to_save + img_number + '.png'
        data[obj_name]['depthfilename'] = folder_to_save + img_number + '.depth.16.png'
        data[obj_name]['base64_img_data'] = ""
        data[obj_name]['file_attributes'] = {}
        data[obj_name]['regions'] = regions
    return stop


# ===================== Synthetic data ====================

data_path = '/data/Akeaveny/Datasets/pringles/zed/train/'
json_addr = '/data/Akeaveny/Datasets/pringles/zed/via_region_data_train.json'
folder_to_save = 'train/'
class_id = 4

if data_path[len(data_path) - 1] != '/':
    print(data_path)
    print('The data path should have / in the end')
    exit()

min_img = 0
max_img = 14999

for i in range(min_img, max_img):
    print('Image: {}/{}'.format(i, max_img))
    count = 1000000 + i
    img_number = str(count)[1:]
    label_addr = data_path + img_number + '.cs.png'

    # print("img_number: ", img_number)
    # print("label_addr: ", label_addr)

    label_img = load_image(label_addr)
    write_to_json(label_img, label_img, class_id, img_number, folder_to_save)

with open(json_addr, 'w') as outfile:
    json.dump(data, outfile, sort_keys=True)