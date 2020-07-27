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

def create_sub_mask_annotation(sub_mask, rgb_img, image_id, category_id, annotation_id, is_crowd,
                               label_img, class_id, img_number, folder_to_save, dataset_name):

    ###################
    # init
    ###################

    # print("String Sequence: ", str_seq)
    obj_name = img_number + dataset_name
    data[obj_name] = {}
    data[obj_name]['fileref'] = ""
    data[obj_name]['size'] = np.array(label_img).shape[1]
    data[obj_name]['filename'] = folder_to_save + img_number + '_rgb.png'
    data[obj_name]['depthfilename'] = folder_to_save + img_number + '_depth.png'
    data[obj_name]['fusedfilename'] = folder_to_save + img_number + '_fused.png'
    data[obj_name]['base64_img_data'] = ""
    data[obj_name]['file_attributes'] = {}

    data[obj_name]['regions'] = {}

    ###################
    # contours
    ###################

    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    all_edges_x = []
    all_edges_y = []
    for idx, contour in enumerate(contours):
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        edges_x = []
        edges_y = []
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
            edges_x.append(int(col-1))
            edges_y.append(int(row-1))
        all_edges_x.append(edges_x)
        all_edges_y.append(edges_y)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords, dtype=np.int).ravel().tolist()
        segmentations.append(segmentation)

        region = {}
        region['region_attributes'] = {}
        region['shape_attributes'] = {}

        if np.array(poly.exterior.coords, dtype=np.int).size != 0:
            region['shape_attributes']["name"] = "polygon"
            region['shape_attributes']["all_points_x"] = edges_x
            region['shape_attributes']["all_points_y"] = edges_y
            region['shape_attributes']["class_id"] = class_id

            data[obj_name]['regions'][np.str(idx)] = region

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    ###########################################################
    #
    ###########################################################

    # if debug:
    #     print("bbox", bbox)
    #
    #     img_bbox = np.array(rgb_img.copy(), dtype=np.uint8)
    #     img_name = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/test_densefusion/' + 'json.bbox.png'
    #     cv2.rectangle(img_bbox, (int(x), int(y)), (int(max_x), int(max_y)), (0, 255, 0), 5)
    #     cv2.imwrite(img_name, img_bbox)
    #     bbox_image = cv2.imread(img_name)
    #
    #     plt.subplot(111)
    #     plt.title("rgb")
    #     plt.imshow(bbox_image)
    #     plt.show()

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return data

###########################################################
# Manual Config
###########################################################
np.random.seed(1)

dataset_name = 'Affordance'
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/'
train_path = 'combined_tools_train/'
val_path = 'combined_tools_val/'

image_ext = '_label.png' ### object ids or affordances

class_id = np.arange(0, 205+1, 1)
### class_id = [0, 1, 2, 3, 4, 5, 6, 7]
print("Affordance IDs: \n{}\n".format(class_id))

use_random_idx = True
# num_val = int(300 / (2 * 3))
# num_train = int(700 / (2 * 3))
# num_test = 25
num_train = num_val = int(25)

#=====================
# JSON FILES
#=====================

# 0.
json_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/json/rgb/syn/'

# 1.
scenes = [
          # 'turn_table/', 'bench/', 'floor/',
          'dr/'
          ]

for scene in scenes:
    print('\n******************** {} ********************'.format(scene))

    if scene == 'dr/' and use_random_idx:
        num_val = int(16)
        num_train = int(num_train * 3)

    ###########################################################
    # VALIDATION
    ###########################################################
    print('\n ------------------ VAL ------------------')

    # =====================
    ### config
    # =====================

    folder_to_save = val_path + scene
    labels = data_path + folder_to_save + '??????' + image_ext
    images = data_path + folder_to_save + '??????' + "_rgb.png"

    files = np.array(sorted(glob.glob(labels)))
    rgb_files = np.array(sorted(glob.glob(images)))
    print("Loaded files: ", len(files))

    if use_random_idx:
        val_idx = np.random.choice(np.arange(0, len(files)+1, 1), size=int(num_val), replace=False)
        print("Chosen Files \n", val_idx)
        files = files[val_idx]
    else:
        num_val = len(files)

    data = {}
    iteration = 0

    # =====================
    ###
    # =====================
    is_crowd = 0
    # These ids will be automatically increased as we go
    annotation_id = 1
    image_id = 1
    # Create the annotations
    annotations = []

    json_addr = json_path + scene + 'val' + np.str(num_val) + 'coco_test.json'
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
            print("class ids: ", np.unique(label_img))
            sub_masks = create_sub_masks(label_img)
            for idx, sub_mask in sub_masks.items():
                if int(idx) > 0:
                    object_id = int(idx)
                    print("object_id: ", object_id)
                    category_id = {'(0, 255, 0)': 0}
                    annotation = create_sub_mask_annotation(sub_mask, rgb_img, image_id, category_id, annotation_id, is_crowd,
                                                            label_img, object_id, img_number, folder_to_save, dataset_name)
                    # write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
                    annotation_id += 1
            image_id += 1

            # write_to_json(label_img, label_img, class_id, img_number, folder_to_save, dataset_name)
        iteration += 1

    with open(json_addr, 'w') as outfile:
        json.dump(data, outfile, sort_keys=True)

