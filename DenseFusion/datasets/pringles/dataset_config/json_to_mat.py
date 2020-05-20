import yaml
import glob
import json
import os
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  # scipy>=1.3


'''
=================== TRAIN =================== 
'''

data_dir = '/data/Akeaveny/Datasets/pringles/zed/train/'
folder_to_save = 'train/'

output = {}
output['cls_indexes'] = []

# ============= load json ==================
json_files = []
json_addrs = '/data/Akeaveny/Datasets/pringles/zed/train/*.json'
images = [json_files.append(file) for file in sorted(glob.glob(json_addrs))]

# ============ output file ================
output = {}
output['cls_indexes'] = []

# ================ pose ===========
for json_file in json_files[0:10]:
        print(json_file)
        open_json_file = json.load(open(json_file))
        labels = []
        poses = np.zeros(shape=(3,4))

        labels.append(np.asarray(1))

        rot = np.asarray(open_json_file['objects'][1]['pose_transform'])
        translation = open_json_file['objects'][1]['location']

        poses[0:3, 0:3] = rot[0:3, 0:3]
        for i in range(0, 3):
                # pose[i].append(translation[i])
                poses[i, -1] = translation[i]

        labels = np.asarray(labels, dtype=np.uint8)

        # rmin, rmax, cmin, cmax = get_bbox(mask)
        rmax, cmax = np.asarray(open_json_file['objects'][1]['bounding_box']['top_left'])
        rmin, cmin = np.asarray(open_json_file['objects'][1]['bounding_box']['bottom_right'])

        # ================ mat =========================
        output['cls_indexes'] = np.reshape(labels, (len(labels),-1))
        output['poses'] = poses
        output['factor_depth'] = [np.asarray([1000], dtype=np.uint16)]
        output['bbox'] = rmin, rmax, cmin, cmax

        # /data/Akeaveny/Datasets/pringles/Alex/train/images/000000.json
        str_num = json_file.split(folder_to_save)[1]
        str_num = str_num.split(".")[0]

        saved_mat_file = data_dir + np.str(str_num) + '-meta.mat'
        sio.savemat(saved_mat_file, output)


'''
=================== VAL =================== 
'''

data_dir = '/data/Akeaveny/Datasets/pringles/zed/val/'
folder_to_save = 'val/'

output = {}
output['cls_indexes'] = []

# ============= load json ==================
json_files = []
json_addrs = '/data/Akeaveny/Datasets/pringles/zed/val/*.json'
images = [json_files.append(file) for file in sorted(glob.glob(json_addrs))]

# ============ output file ================
output = {}
output['cls_indexes'] = []

# ================ pose ===========
for json_file in json_files[0:10]:
        print(json_file)
        open_json_file = json.load(open(json_file))
        labels = []
        poses = np.zeros(shape=(3,4))

        labels.append(np.asarray(1))

        rot = np.asarray(open_json_file['objects'][1]['pose_transform'])
        translation = open_json_file['objects'][1]['location']

        poses[0:3, 0:3] = rot[0:3, 0:3]
        for i in range(0, 3):
                # pose[i].append(translation[i])
                poses[i, -1] = translation[i]

        labels = np.asarray(labels, dtype=np.uint8)

        # rmin, rmax, cmin, cmax = get_bbox(mask)
        rmax, cmax = np.asarray(open_json_file['objects'][1]['bounding_box']['top_left'])
        rmin, cmin = np.asarray(open_json_file['objects'][1]['bounding_box']['bottom_right'])

        # ================ mat =========================
        output['cls_indexes'] = np.reshape(labels, (len(labels),-1))
        output['poses'] = poses
        output['factor_depth'] = [np.asarray([1000], dtype=np.uint16)]
        output['bbox'] = rmin, rmax, cmin, cmax

        # /data/Akeaveny/Datasets/pringles/Alex/train/images/000000.json
        str_num = json_file.split(folder_to_save)[1]
        str_num = str_num.split(".")[0]

        saved_mat_file = data_dir + np.str(str_num) + '-meta.mat'
        sio.savemat(saved_mat_file, output)