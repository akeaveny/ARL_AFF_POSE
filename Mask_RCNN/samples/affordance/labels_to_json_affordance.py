import json
import cv2
import glob
import matplotlib.pyplot as plt
import re
import numpy as np

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing
stop = True
debug = True

data = None

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
        obj_name = img_number + 'pringles'
        data[obj_name] = {}
        data[obj_name]['fileref'] = ""
        data[obj_name]['size'] = 1024
        # /data/Akeaveny/Datasets/part-affordance-dataset/train/scene_01_00000001_rgb.jpg
        data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
        data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
        data[obj_name]['base64_img_data'] = ""
        data[obj_name]['file_attributes'] = {}
        data[obj_name]['regions'] = regions
    return stop


# ===================== val ====================
data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/val/'
folder_to_save = 'train/'

if data_path[len(data_path) - 1] != '/':
    print(data_path)
    print('The data path should have / in the end')
    exit()

class_id = [0, 1, 2, 3, 4, 5, 6, 7]

min_img = 0
max_img = 289

data = {}
count = 0
# ===================== VAL ====================
print('\n-------- VAL ---------------')
json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_ndds_and_real_train.json'
for i in range(min_img, max_img+1):
    print('Image: {}/{}'.format(i, max_img))
    # count = 1000000 + i
    # img_number = str(count)[1:]
    img_number = str(i)
    label_addr = data_path + img_number + '_label.png'

    print("img_number: ", img_number)
    print("label_addr: ", label_addr)

    label_img = load_image(label_addr)
    print("Classes: ", np.unique(label_img))

    if label_img.size == 0:
        print('\n ------------------ Pass! --------------------')
        pass
    else:
        write_to_json(label_img, label_img, class_id, img_number, folder_to_save)
    count += 1

with open(json_addr, 'w') as outfile:
    json.dump(data, outfile, sort_keys=True)
#
# # ========== selecting train/val splits ==========
# # batch_size = 10
# # train_split = 8 # 80 %
# # batch_size = 0
# # train_split = 5 # 80 %
# # combined_imgs = np.arange(0,batch_size, 1)
# # train_imgs = np.random.choice(batch_size, size=int(train_split), replace=False)
# # val_imgs = np.delete(combined_imgs, train_imgs)
#
# print('\n-------- Loading Images! ---------------')
# print('Batch Size: ', combined_imgs.shape[0])
# print("Training: ", train_imgs.shape[0])
# print("Val: ", val_imgs.shape[0])
# # print("\n")
#
# data = {}
# # ===================== training ====================
# print('\n-------- Training json! ---------------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_ndds_and_real_train.json'
# count = 0
# for train_idx in train_imgs:
#     print('Image: {}/{}'.format(count, train_imgs.shape[0]))
#     img_number = str(train_idx)
#     label_addr = data_path + img_number + '_label.png'
#     # print("img_number: ", img_number)
#     # print("label_addr: ", label_addr)
#
#     label_img = load_image(label_addr)
#     # print("Classes: ", np.unique(label_img))
#
#     if label_img.size == 0:
#         print('\n ------------------ Pass! --------------------')
#         pass
#     else:
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save)
#     count +=1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)
#
# data = {}
# # ===================== val ====================
# print('\n-------- Val json! ---------------')
# json_addr = '/data/Akeaveny/Datasets/part-affordance-dataset/via_region_data_ndds_and_real_val.json'
# count = 0
# for val_idx in val_imgs:
#     print('Image: {}/{}'.format(count, val_imgs.shape[0]))
#     img_number = str(val_idx)
#     label_addr = data_path + img_number + '_label.png'
#     # print("img_number: ", img_number)
#     # print("label_addr: ", label_addr)
#
#     label_img = load_image(label_addr)
#     # print("Classes: ", np.unique(label_img))
#
#     if label_img.size == 0:
#         print('\n ------------------ Pass! --------------------')
#         pass
#     else:
#         write_to_json(label_img, label_img, class_id, img_number, folder_to_save)
#     count +=1
#
# with open(json_addr, 'w') as outfile:
#     json.dump(data, outfile, sort_keys=True)