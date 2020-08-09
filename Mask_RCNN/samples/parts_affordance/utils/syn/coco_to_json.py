import json
import glob
import cv2

import matplotlib.pyplot as plt

from PIL import Image # (pip install Pillow)

import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

visual = False  # only use True with 1 image for testing because there is a bug in openCV drawing
stop = True
data = None

debug = True

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

def create_sub_mask_annotation(sub_mask, class_id, label_img):

    ###################
    # contours
    ###################

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

    polygons = []
    x_list, y_list = [], []
    for idx, contour in enumerate(contours):
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        # segmentation = np.array(poly.exterior.coords, dtype=np.int).ravel().tolist()
        # segmentations.append(segmentation

        if type(poly) == Polygon:
            coords = np.array(poly.exterior.coords, dtype=np.int)

            if coords.size != 0:
                x, y = coords[:, 0], coords[:, 1]
                x_list.extend(x.tolist())
                y_list.extend(y.tolist())
        else:
            # Multipolygon
            polys = list(poly)
            for poly_ in polys:

                coords = np.array(poly_.exterior.coords, dtype=np.int)

                if coords.size != 0:
                    x, y = coords[:, 0], coords[:, 1]
                    x_list.extend(x.tolist())
                    y_list.extend(y.tolist())

    if len(x_list) > 0 and len(y_list) > 0:
        region = {}
        region['region_attributes'] = {}
        region['shape_attributes'] = {}
        region['shape_attributes']["name"] = "polygon"
        region['shape_attributes']["all_points_x"] = x_list
        region['shape_attributes']["all_points_y"] = y_list
        region['shape_attributes']["class_id"] = class_id

        data[obj_name]['regions'][np.str(class_id)] = region

###########################################################
# Manual Config
###########################################################
np.random.seed(1)

dataset_name = 'Affordance'
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/'
train_path = 'combined_tools_train/'
val_path = 'combined_tools_val/'

image_ext = '_label.png' ### object ids or affordances

class_id = np.arange(0, 21+1, 1)
### class_id = [0, 1, 2, 3, 4, 5, 6, 7]

print("Affordance IDs: \n{}\n".format(class_id))

use_random_idx = False
num_val = num_train = 4

#=====================
# JSON FILES
#=====================

# 0.
json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb/syn/'

# 1.
scenes = [
        'bench/', 'floor/', 'turn_table/',
        # 'dr/',
          ]

for scene in scenes:
    print('\n******************** {} ********************'.format(scene))

    print('******************** VAL ********************')
    # =====================
    ### config
    # =====================
    folder_to_save = val_path + scene
    labels = data_path + folder_to_save + '??????' + image_ext
    images = data_path + folder_to_save + '??????' + "_rgb.png"

    print("labels: ", labels)
    files = np.array(sorted(glob.glob(labels)))
    rgb_files = np.array(sorted(glob.glob(images)))
    print("Loaded files: ", len(files))

    if use_random_idx:
        val_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_val), replace=False)
        print("Chosen Files \n", val_idx)
        files = files[val_idx]
    else:
        num_val = len(files)

    data = {}
    iteration = 0

    #=====================
    #
    #=====================

    json_addr = json_path + scene + 'coco_val_' + np.str(len(files)) + '.json'
    print("json_addr: ", json_addr)
    for idx, file in enumerate(files):

        str_num = file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(image_ext)[0]
        label_addr = file

        ### print("label_addr: ", label_addr)
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

            print("class ids: ", np.unique(label_img))
            ###################
            # sub masks
            ###################
            sub_masks = create_sub_masks(label_img)
            for idx, sub_mask in sub_masks.items():
                if int(idx) > 0:
                    object_id = int(idx)
                    print("object_id: ", object_id)
                    create_sub_mask_annotation(sub_mask, object_id, np.array(label_img))
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)

    # =====================
    ### config
    # =====================
    print('******************** TRAIN ********************')
    folder_to_save = train_path + scene
    labels = data_path + folder_to_save + '??????' + image_ext
    images = data_path + folder_to_save + '??????' + "_rgb.png"

    print("labels: ", labels)
    files = np.array(sorted(glob.glob(labels)))
    rgb_files = np.array(sorted(glob.glob(images)))
    print("Loaded files: ", len(files))

    if use_random_idx:
        train_idx = np.random.choice(np.arange(0, len(files), 1), size=int(num_train), replace=False)
        print("Chosen Files \n", train_idx)
        files = files[train_idx]
    else:
        num_train = len(files)

    data = {}
    iteration = 0

    # =====================
    #
    # =====================

    json_addr = json_path + scene + 'coco_train_' + np.str(len(files)) + '.json'
    print("json_addr: ", json_addr)
    for idx, file in enumerate(files):

        str_num = file.split(data_path + folder_to_save)[1]
        img_number = str_num.split(image_ext)[0]
        label_addr = file

        ### print("label_addr: ", label_addr)
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

            print("class ids: ", np.unique(label_img))
            ###################
            # sub masks
            ###################
            sub_masks = create_sub_masks(label_img)
            for idx, sub_mask in sub_masks.items():
                if int(idx) > 0:
                    object_id = int(idx)
                    print("object_id: ", object_id)
                    create_sub_mask_annotation(sub_mask, object_id, np.array(label_img))
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)