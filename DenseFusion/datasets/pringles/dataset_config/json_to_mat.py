import yaml
import glob
import json
import os
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  # scipy>=1.3

data_dir = '/data/Akeaveny/Datasets/pringles/Alex/train/images/'
output = {}
output['cls_indexes'] = []

# ============= load json ==================
json_files = []
json_addrs = '/data/Akeaveny/Datasets/pringles/Alex/train/images/*.json'
images = [json_files.append(file) for file in glob.glob(json_addrs)]

# ============ output file ================
output = {}
output['cls_indexes'] = []

# ================ pose ===========
for json_file in json_files:
        open_json_file = json.load(open(json_file))
        labels = []
        poses = []

        labels.append(np.asarray(1))

        quaternion_xyzw = open_json_file['objects'][0]['quaternion_xyzw']
        translation = open_json_file['objects'][0]['location']

        rot = R.from_quat(quaternion_xyzw)  # x y z w
        pose = rot.as_dcm().tolist()
        for i in range(0, 3):
                pose[i].append(translation[i])

        labels = np.asarray(labels, dtype=np.uint8)
        # print(labels)

        poses = np.asarray(pose)
        # poses = np.reshape(poses, (3, 4, -1))
        # print(poses)

        # ================ mat =========================
        output['cls_indexes'] = np.reshape(labels, (len(labels),-1))
        output['poses'] = poses
        output['factor_depth'] = [np.asarray([1000], dtype=np.uint16)]

        # /data/Akeaveny/Datasets/pringles/Alex/train/images/000000.json
        str_num = json_file.split('images/')[1]
        str_num = str_num.split(".")[0]

        saved_mat_file = '/data/Akeaveny/Datasets/pringles/Alex/train/images/' + np.str(str_num) + '-meta.mat'
        sio.savemat(saved_mat_file, output)