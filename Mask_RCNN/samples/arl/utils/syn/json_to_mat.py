import yaml
import glob
import json
import os
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  # scipy>=1.3

def json_to_mat(json_file, camera_settings):

        open_json_file = json.load(open(json_file))

        ####################
        # prelim
        ####################

        if camera_settings == "Kinect":
                width, height = 640, 480
                cam_fx = 1066.778
                cam_fy = 1067.487
                cam_cx = 312.9869
                cam_cy = 241.3109

                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]

        if camera_settings == "Xtion":
                width, height = 640, 480
                cam_fx = 537.99040688
                cam_fy = 539.09122804
                cam_cx = 321.24099379
                cam_cy = 237.11014479

                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640]

        elif camera_settings == "ZED":
                width, height = 672, 376
                cam_fx = 339.11297607421875
                cam_fy = 339.11297607421875
                cam_cx = 343.0655822753906
                cam_cy = 176.86412048339844

                border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

        ####################
        ####################
        output = {}
        output['Affordance_ID'] = []
        output['Affordance_Label'] = []
        output['Model_ID'] = []

        for idx, object in enumerate(open_json_file['objects']):

                ####################
                # aff id
                ####################
                actor_tag = object['class']

                affordance_id = actor_tag.split("_")[0]
                model_id = actor_tag.split("_")[1]
                affordance_label = actor_tag.split("_")[-1]

                if model_id == 'spatula':
                        if affordance_label == 'grasp':
                                idx = 4
                                count = 1000 + idx
                                affordance_id = str(count)[1:]
                        elif affordance_label == 'support':
                                idx = 3
                                count = 1000 + idx
                                affordance_id = str(count)[1:]

                if affordance_id in ['001', '004']:
                        # print('affordance_id: ', affordance_id)

                        output['Affordance_ID'].append(affordance_id)
                        output['Affordance_Label'].append(affordance_label)
                        output['Model_ID'].append(model_id)

                        ####################
                        # SE3
                        ####################
                        rot = np.asarray(object['pose_transform'])[0:3, 0:3]

                        # change LHS coordinates
                        cam_rotation4 = np.dot(np.array(rot), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
                        cam_rotation4 = np.dot(cam_rotation4.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]))

                        # NDDS gives units in centimeters
                        translation = np.array(object['location']) * 10 # now in [mm]
                        translation /= 1e3  # now in [m]

                        output['rot' + np.str(affordance_id)] = cam_rotation4
                        output['cam_translation' + np.str(affordance_id)] = translation

                        ####################
                        # bbox
                        ####################
                        rmax, cmax = np.asarray(object['bounding_box']['top_left'])
                        rmin, cmin = np.asarray(object['bounding_box']['bottom_right'])
                        output['bbox' + np.str(affordance_id)] = rmin, rmax, cmin, cmax

                        ####################
                        # cam
                        ####################
                        output['camera_setting' + np.str(affordance_id)] = camera_settings

                        output['width' + np.str(affordance_id)] = width
                        output['height' + np.str(affordance_id)] = height
                        output['fx' + np.str(affordance_id)] = cam_fx
                        output['fy' + np.str(affordance_id)] = cam_fy
                        output['cx' + np.str(affordance_id)] = cam_cx
                        output['cy' + np.str(affordance_id)] = cam_cy

                        output['camera_scale' + np.str(affordance_id)] = [np.asarray([1000], dtype=np.uint16)]
                        output['border' + np.str(affordance_id)] = border_list

        return output
