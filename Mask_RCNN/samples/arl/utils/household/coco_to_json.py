import json
import glob
import cv2

import os

import matplotlib.pyplot as plt

from PIL import Image # (pip install Pillow)

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

###########################################################
# coco
###########################################################

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != (0, 0, 0):
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def create_sub_mask_annotation(sub_mask, class_id, label_img, rgb_img):
    #############
    # cv image
    #############

    h, w = sub_mask.size
    img = np.array(sub_mask.getdata(), dtype=np.uint8).reshape(w, h)

    # cv2.imwrite(os.getcwd() + "contours.png", img)
    # img = cv2.imread(os.getcwd() + "contours.png", 0) * 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * 255

    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an output of all zeroes that has the same shape as the input
    # image
    out = np.zeros_like(label_img)

    # On this output, draw all of the contours that we have detected
    # in white, and set the thickness to be 3 pixels
    cv2.drawContours(rgb_img, contours, -1, 255, 3)

    region = {}
    region['region_attributes'] = {}
    region['shape_attributes'] = {}
    region['shape_attributes']["name"] = "polygon"
    region['shape_attributes']["num_contours"] = len(contours)
    # region['shape_attributes']["all_points_x"] = np.array(x_list).tolist()
    # region['shape_attributes']["all_points_y"] = np.array(y_list).tolist()
    region['shape_attributes']["class_id"] = class_id

    for contour_idx, k in enumerate(contours):
        x_list = []
        y_list = []
        for i in k:
            for j in i:
                x_list.append(j[0])
                y_list.append(j[1])
        region['shape_attributes']["all_points_x" + str(contour_idx)] = np.array(x_list).tolist()
        region['shape_attributes']["all_points_y" + str(contour_idx)] = np.array(y_list).tolist()

    if VISUALIZE:
        ### cv
        cv2.imshow("out", rgb_img)
        cv2.waitKey(0)
        ### matplotlib
        # plt.imshow(label_img)
        # plt.plot(x_list, y_list, linewidth=1)
        # plt.show()
        # plt.ioff()

    return region

###########################################################
# Manual Config
###########################################################
np.random.seed(1)

dataset_name = 'ARL'

######################
# TOOLS
######################

# data_path = '/data/Akeaveny/Datasets/arl_dataset/'
# val_path = 'combined_household_tools_1_val/'
# train_path = 'combined_household_tools_1_train/'
# test_path = 'combined_household_tools_1_test/'
#
# json_path = '/data/Akeaveny/Datasets/arl_dataset/json/household/tools/'
# json_name = 'coco_household_tools_'

######################
# CLUTTER
######################

data_path = '/data/Akeaveny/Datasets/arl_dataset/'
val_path = 'combined_household_clutter_1_val/'
train_path = 'combined_household_clutter_1_train/'
test_path = 'combined_household_clutter_1_test/'

json_path = '/data/Akeaveny/Datasets/arl_dataset/json/household/clutter/'
json_name = 'coco_household_clutter_'

image_ext = '_label.png'

class_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("Affordance IDs: \n{}\n".format(class_id))

VISUALIZE = False
use_random_idx = True
###
BASE_NUM_IMAGES = int(1500)
num_val = int(BASE_NUM_IMAGES*0.2)
num_train = int(BASE_NUM_IMAGES*0.8)
num_test = 0

# 1.
scenes = ['']

#=====================
# JSON FILES
#=====================

for scene in scenes:
    print('\n******************** Scene: {} ********************'.format(scene))
    # =====================
    # VAL
    # =====================
    if num_val == 0:
        print('******************** SKIPPING VAL ********************')
        pass
    else:
        print('******************** VAL! ********************')
        folder_to_save = val_path + scene
        labels = data_path + folder_to_save + '*' + image_ext
        images = data_path + folder_to_save + '*' + "_rgb.png"

        print("labels: ", labels)
        files = np.array(sorted(glob.glob(labels)))
        rgb_files = np.array(sorted(glob.glob(images)))
        print("Loaded files: ", len(files))

        if use_random_idx:
            val_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_val), replace=False)
            print("Chosen Files ", len(val_idx))
            files = files[val_idx]
        else:
            num_val = len(files)

        data = {}
        iteration = 0

        #=====================

        json_addr = json_path + scene + json_name + 'val_' + np.str(len(files)) + '.json'
        print("json_addr: ", json_addr)
        for idx, file in enumerate(files):

            str_num = file.split(data_path + folder_to_save)[1]
            img_number = str_num.split(image_ext)[0]
            label_addr = file

            rgb_addr = data_path + folder_to_save + img_number + "_rgb.png"

            print("label_addr: ", label_addr)
            print("rgb_addr: ", rgb_addr)
            print('Image: {}/{}'.format(iteration, len(files)))

            rgb_img = np.array(Image.open(rgb_addr))
            label_img = Image.open(label_addr)
            object_ids = np.unique(np.array(label_img))
            print("GT Affordances:", object_ids)

            if label_img.size == 0:
                print('\n ------------------ Pass! --------------------')
                pass
            else:
                # ###################
                # # init
                # ###################
                obj_name = img_number + dataset_name
                data[obj_name] = {}
                data[obj_name]['fileref'] = ""
                data[obj_name]['size'] = 640
                data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
                data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
                data[obj_name]['base64_img_data'] = ""
                data[obj_name]['file_attributes'] = {}

                data[obj_name]['regions'] = {}
                regions = {}

                print("class ids: ", np.unique(label_img))
                ###################
                # sub masks
                ###################
                sub_masks = create_sub_masks(label_img)
                for idx, sub_mask in sub_masks.items():
                    if int(idx) > 0:
                        object_id = int(idx)
                        print("object_id: ", object_id)
                        region = create_sub_mask_annotation(sub_mask, object_id, np.array(label_img), rgb_img)
                        regions[np.str(object_id)] = region
                data[obj_name]['regions'] = regions
            iteration += 1

        with open(json_addr, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    # =====================
    # TRAIN
    # =====================
    if num_train == 0:
        print('******************** SKIPPING TRAIN ********************')
        pass
    else:
        print('******************** TRAIN! ********************')
        folder_to_save = train_path + scene
        labels = data_path + folder_to_save + '??????' + image_ext
        images = data_path + folder_to_save + '??????' + "_rgb.png"

        print("labels: ", labels)
        files = np.array(sorted(glob.glob(labels)))
        rgb_files = np.array(sorted(glob.glob(images)))
        print("Loaded files: ", len(files))

        if use_random_idx:
            train_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_train), replace=False)
            print("Chosen Files: ", len(train_idx))
            files = files[train_idx]
        else:
            num_train = len(files)

        data = {}
        iteration = 0

        # =====================

        json_addr = json_path + scene + json_name + 'train_' + np.str(len(files)) + '.json'
        print("json_addr: ", json_addr)
        for idx, file in enumerate(files):

            str_num = file.split(data_path + folder_to_save)[1]
            img_number = str_num.split(image_ext)[0]
            label_addr = file

            print("label_addr: ", label_addr)
            print('Image: {}/{}'.format(iteration, len(files)))

            rgb_img = np.array(Image.open(rgb_files[idx]))
            label_img = Image.open(label_addr)
            object_ids = np.unique(np.array(label_img))
            print("GT Affordances:", object_ids)

            if label_img.size == 0:
                print('\n ------------------ Pass! --------------------')
                pass
            else:
                ###################
                # init
                ###################
                obj_name = img_number + dataset_name
                data[obj_name] = {}
                data[obj_name]['fileref'] = ""
                data[obj_name]['size'] = 640
                data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
                data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
                data[obj_name]['base64_img_data'] = ""
                data[obj_name]['file_attributes'] = {}

                data[obj_name]['regions'] = {}
                regions = {}

                print("class ids: ", np.unique(label_img))
                ###################
                # sub masks
                ###################
                sub_masks = create_sub_masks(label_img)
                for idx, sub_mask in sub_masks.items():
                    if int(idx) > 0:
                        object_id = int(idx)
                        print("object_id: ", object_id)
                        region = create_sub_mask_annotation(sub_mask, object_id, np.array(label_img), rgb_img)
                        regions[np.str(object_id)] = region
                data[obj_name]['regions'] = regions
            iteration += 1

        with open(json_addr, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    # =====================
    # TEST
    # =====================
    if num_test == 0:
        print('******************** SKIPPING TEST ********************')
        pass
    else:
        print('******************** TEST! ********************')
        folder_to_save = test_path + scene
        labels = data_path + folder_to_save + '??????' + image_ext
        images = data_path + folder_to_save + '??????' + "_rgb.png"

        print("labels: ", labels)
        files = np.array(sorted(glob.glob(labels)))
        rgb_files = np.array(sorted(glob.glob(images)))
        print("Loaded files: ", len(files))

        if use_random_idx:
            test_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_test), replace=False)
            print("Chosen Files: ", len(test_idx))
            files = files[test_idx]
        else:
            num_test = len(files)

        data = {}
        iteration = 0

        # =====================

        json_addr = json_path + scene + json_name + 'test_' + np.str(len(files)) + '.json'
        print("json_addr: ", json_addr)
        for idx, file in enumerate(files):

            str_num = file.split(data_path + folder_to_save)[1]
            img_number = str_num.split(image_ext)[0]
            label_addr = file

            print("label_addr: ", label_addr)
            print('Image: {}/{}'.format(iteration, len(files)))

            rgb_img = np.array(Image.open(rgb_files[idx]))
            label_img = Image.open(label_addr)
            object_ids = np.unique(np.array(label_img))
            print("GT Affordances:", object_ids)

            if label_img.size == 0:
                print('\n ------------------ Pass! --------------------')
                pass
            else:
                ###################
                # init
                ###################
                obj_name = img_number + dataset_name
                data[obj_name] = {}
                data[obj_name]['fileref'] = ""
                data[obj_name]['size'] = 640
                data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
                data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
                data[obj_name]['base64_img_data'] = ""
                data[obj_name]['file_attributes'] = {}

                data[obj_name]['regions'] = {}
                regions = {}

                print("class ids: ", np.unique(label_img))
                ###################
                # sub masks
                ###################
                sub_masks = create_sub_masks(label_img)
                for idx, sub_mask in sub_masks.items():
                    if int(idx) > 0:
                        object_id = int(idx)
                        print("object_id: ", object_id)
                        region = create_sub_mask_annotation(sub_mask, object_id, np.array(label_img), rgb_img)
                        regions[np.str(object_id)] = region
                data[obj_name]['regions'] = regions
            iteration += 1

        with open(json_addr, 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)