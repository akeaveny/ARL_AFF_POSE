import yaml
import glob
import json
import os
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  # scipy>=1.3

def yaml_to_mat(yaml_addr):

        yaml_file = open(yaml_addr, 'r')
        parsed = yaml.load(yaml_file, Loader=yaml.FullLoader)

        ############################
        # get poses
        ############################

        labels = []
        poses = []
        for idx, obj in enumerate(parsed.keys()):
                label = np.asarray(parsed[obj]['label'], dtype=np.uint8)
                labels.append(label)
                # translation
                trans = parsed[obj]['pose'][0]
                # rotation
                quart = parsed[obj]['pose'][1] # x y z w
                quart.append(quart[0])
                quart.pop(0)
                rot = R.from_quat(quart)  # x y z w
                pose = rot.as_dcm().tolist()

                for i in range(0, 3):
                    pose[i].append(trans[i])

                if idx == 0:
                    for i in range(0, 3):
                        row = []
                        for k in range(0, 4):
                            ele = []
                            ele.append(pose[i][k])
                            row.append(ele)
                        poses.append(row)
                else:
                    for i in range(0, 3):
                        for k in range(0, 4):
                            poses[i][k].append(pose[i][k])

        poses = np.asarray(poses)
        poses = np.reshape(poses, (3, 4, len(parsed)))

        labels_array = np.asarray(labels, dtype=np.uint8)

        ############################
        # format output
        ############################

        output = {}
        output['Affordance_ID'] = []
        output['Actor_Tag'] = []

        for idx, affordance_id in enumerate(labels_array):
                if affordance_id in np.array([1, 4]): # TODO
                        # print("affordance_id: ", affordance_id)

                        # class idx
                        if affordance_id == 1:
                                actor_tag = "hammer_01_grasp"
                        elif affordance_id == 2:
                                actor_tag = "hammer_05_pound"
                        elif affordance_id == 3:
                                actor_tag = "spatula_06_support"
                        elif affordance_id == 4:
                                actor_tag = "spatula_01_grasp"
                        else:
                                print("*** ACTOR TAG DOESN'T EXIST ***")
                                exit(1)

                        count = 1000 + affordance_id
                        meta_idx = str(count)[1:]
                        output['Affordance_ID'].append(meta_idx)
                        output['Actor_Tag'].append(actor_tag)

                        # pose
                        output['rot' + np.str(meta_idx)] = poses[0:3, 0:3, idx]
                        output['cam_translation' + np.str(meta_idx)] = poses[0:3, -1, idx]

                        # ZED
                        output['width' + np.str(meta_idx)] = 672
                        output['height' + np.str(meta_idx)] = 376
                        output['fx'+ np.str(meta_idx)] = 339.11297607421875
                        output['fy'+ np.str(meta_idx)] = 339.11297607421875
                        output['cx'+ np.str(meta_idx)] = 343.0655822753906
                        output['cy'+ np.str(meta_idx)] = 176.86412048339844

                        output['camera_scale' + np.str(meta_idx)] = [np.asarray([1000], dtype=np.uint16)]
                        output['border' + np.str(meta_idx)] = [-1, 40, 80, 120, 160, 200, 240, 280, 320,
                                                               360, 400, 440, 480, 520, 560, 600, 640, 680]



        return output
