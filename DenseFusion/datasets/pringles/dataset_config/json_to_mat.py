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

# ================ mat file ===========
for json_file in json_files:
        print(json_file)
        open_json_file = json.load(open(json_file))

        labels = []
        labels.append(np.asarray(1))
        labels = np.asarray(labels, dtype=np.uint8)
        # ================ prelim =========================
        output['cls_indexes'] = np.reshape(labels, (len(labels), -1))
        output['camera_scale'] = [np.asarray([1], dtype=np.uint16)]

        # ================ pose =========================

        rot = np.asarray(open_json_file['objects'][1]['pose_transform'])[0:3, 0:3]
        output['rot'] = rot
        output['rot1'] = np.dot(rot, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        # trans = np.zeros(shape=(1, 3))
        # translation = open_json_file['objects'][1]['location']
        #
        # poses[0:3, 0:3] = rot[0:3, 0:3]
        # for i in range(0, 3):
        #         # pose[i].append(translation[i])
        #         poses[i, -1] = translation[i]

        output['trans'] = np.asarray(open_json_file['camera_data']['location_worldframe']) * 10  # NDDS gives units in centimeters
        translation = np.array(open_json_file['objects'][1]['location']) * 10  # NDDS gives units in centimeters
        output['cam_translation'] = translation

        quaternion_obj2cam = R.from_quat(np.array(open_json_file['objects'][1]['quaternion_xyzw']))
        quaternion_cam2world = R.from_quat(np.array(open_json_file['camera_data']['quaternion_xyzw_worldframe']))
        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
        mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        output['obj2cam_rotation'] = quaternion_obj2cam.as_dcm()
        output['obj2cam_rotation1'] = np.dot(quaternion_obj2cam.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
        output['obj2world_rotation'] = quaternion_obj2world.as_dcm()
        output['obj2world_rotation1'] = mirrored_y_axis

        # ================ mat =========================
        # rmin, rmax, cmin, cmax = get_bbox(mask)
        rmax, cmax = np.asarray(open_json_file['objects'][1]['bounding_box']['top_left'])
        rmin, cmin = np.asarray(open_json_file['objects'][1]['bounding_box']['bottom_right'])
        output['bbox'] = rmin, rmax, cmin, cmax

        bbox = open_json_file['objects'][1]['bounding_box']
        bbox_x = round(bbox['top_left'][0])
        bbox_y = round(bbox['top_left'][1])
        bbox_w = round(bbox['bottom_right'][0]) - bbox_x
        bbox_h = round(bbox['bottom_right'][1]) - bbox_y
        output['bbox1'] = bbox_x, bbox_y, bbox_w, bbox_h

        # ================ camera =========================
        cam_json_file = json.load(open('/data/Akeaveny/Datasets/pringles/zed/_camera_settings.json'))
        output['fx'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['fx'])
        output['fy'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['fy'])
        output['cx'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['cx'])
        output['cy'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['cy'])

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

# ================ mat file ===========
for json_file in json_files:
        print(json_file)
        open_json_file = json.load(open(json_file))

        labels = []
        labels.append(np.asarray(1))
        labels = np.asarray(labels, dtype=np.uint8)
        # ================ prelim =========================
        output['cls_indexes'] = np.reshape(labels, (len(labels), -1))
        output['camera_scale'] = [np.asarray([1], dtype=np.uint16)]

        # ================ pose =========================
        # rot = np.zeros(shape=(3, 3))
        # trans = np.zeros(shape=(1, 3))

        rot = np.asarray(open_json_file['objects'][1]['pose_transform'])[0:3, 0:3]
        output['rot'] = rot
        output['rot1'] = np.dot(rot, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        # poses[0:3, 0:3] = rot[0:3, 0:3]
        # for i in range(0, 3):
        #         # pose[i].append(translation[i])
        #         poses[i, -1] = translation[i]

        output['trans'] = np.asarray(open_json_file['camera_data']['location_worldframe']) * 10  # NDDS gives units in centimeters
        translation = np.array(open_json_file['objects'][1]['location']) * 10  # NDDS gives units in centimeters
        output['cam_translation'] = translation

        quaternion_obj2cam = R.from_quat(np.array(open_json_file['objects'][1]['quaternion_xyzw']))
        quaternion_cam2world = R.from_quat(np.array(open_json_file['camera_data']['quaternion_xyzw_worldframe']))
        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
        mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))

        output['obj2cam_rotation'] = quaternion_obj2cam.as_dcm()
        output['obj2cam_rotation1'] = np.dot(quaternion_obj2cam.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
        output['obj2world_rotation'] = quaternion_obj2world.as_dcm()
        output['obj2world_rotation1'] = mirrored_y_axis

        # ================ mat =========================
        # rmin, rmax, cmin, cmax = get_bbox(mask)
        rmax, cmax = np.asarray(open_json_file['objects'][1]['bounding_box']['top_left'])
        rmin, cmin = np.asarray(open_json_file['objects'][1]['bounding_box']['bottom_right'])
        output['bbox'] = rmin, rmax, cmin, cmax

        bbox = open_json_file['objects'][1]['bounding_box']
        bbox_x = round(bbox['top_left'][0])
        bbox_y = round(bbox['top_left'][1])
        bbox_w = round(bbox['bottom_right'][0]) - bbox_x
        bbox_h = round(bbox['bottom_right'][1]) - bbox_y
        output['bbox1'] = bbox_x, bbox_y, bbox_w, bbox_h

        # ================ camera =========================
        cam_json_file = json.load(open('/data/Akeaveny/Datasets/pringles/zed/_camera_settings.json'))
        output['fx'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['fx'])
        output['fy'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['fy'])
        output['cx'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['cx'])
        output['cy'] = np.asarray(cam_json_file['camera_settings'][0]['intrinsic_settings']['cy'])

        # /data/Akeaveny/Datasets/pringles/Alex/train/images/000000.json
        str_num = json_file.split(folder_to_save)[1]
        str_num = str_num.split(".")[0]

        saved_mat_file = data_dir + np.str(str_num) + '-meta.mat'
        sio.savemat(saved_mat_file, output)
