import yaml
import glob
import json
import os
import numpy as np
import scipy
import scipy.io as sio
from scipy.spatial.transform import Rotation as R  # scipy>=1.3

def json_to_mat(json_file, camera_settings):

        # print(json_file)
        open_json_file = json.load(open(json_file))

        if camera_settings == "Kinetic":
                width, height = 640, 480
                cam_fx = 517.055
                cam_fy = 517.679
                cam_cx = 315.008
                cam_cy = 264.155

        if camera_settings == "Xtion":
                width, height = 640, 480
                cam_fx = 570.3422241210938
                cam_fy = 570.3422241210938
                cam_cx = 319.5
                cam_cy = 239.5

        elif camera_settings == "ZED":
                width, height = 1280, 720
                cam_fx = 697.6871337890625
                cam_fy = 697.6871337890625
                cam_cx = 620.2407836914062
                cam_cy = 353.91357421875

        # ============ output file ================
        output = {}
        output['Affordance_ID'] = []
        output['Affordance_Label'] = []
        output['Model_ID'] = []

        # print(len(open_json_file['objects']))
        for idx, object in enumerate(open_json_file['objects']):
                # print(idx)

                # ================ class idx ======================
                actor_tag = object['class']
                # print("actor_tag: ", actor_tag)

                if 'hammer' in actor_tag:
                        # print("actor_tag: ", actor_tag)

                        affordance_id = actor_tag.split("_")[0]
                        model_id = actor_tag.split("_")[1]
                        affordance_label = actor_tag.split("_")[-1]

                        # print("affordance_id: ", affordance_id)
                        # print("affordance_label: ", affordance_label)
                        # print("model_id: ", model_id)

                        output['Affordance_ID'].append(affordance_id)
                        output['Affordance_Label'].append(affordance_label)
                        output['Model_ID'].append(model_id)

                        # ================ pose =========================
                        rot = np.asarray(object['pose_transform'])[0:3, 0:3]
                        output['rot' + np.str(affordance_id)] = rot

                        translation = np.array(object['location']) * 10  # NDDS gives units in centimeters
                        output['cam_translation' + np.str(affordance_id)] = translation

                        quaternion = np.asarray(object['quaternion_xyzw'])
                        output['quaterniona' + np.str(affordance_id)] = quaternion

                        quaternion_obj2cam = R.from_quat(np.array(object['quaternion_xyzw']))
                        quaternion_cam2world = R.from_quat(np.array(open_json_file['camera_data']['quaternion_xyzw_worldframe']))
                        quaternion_obj2world = quaternion_obj2cam * quaternion_cam2world
                        mirrored_y_axis = np.dot(quaternion_obj2world.as_dcm(), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]))
                        r = R.from_matrix(mirrored_y_axis)
                        output['quaternionb' + np.str(affordance_id)] = r.as_quat()

                        # ================ mat =========================
                        # rmin, rmax, cmin, cmax = get_bbox(mask)
                        rmax, cmax = np.asarray(object['bounding_box']['top_left'])
                        rmin, cmin = np.asarray(object['bounding_box']['bottom_right'])
                        output['bbox' + np.str(affordance_id)] = rmin, rmax, cmin, cmax

                        # ================ camera =========================
                        output['camera_setting' + np.str(affordance_id)] = camera_settings
                        output['camera_scale' + np.str(affordance_id)] = [np.asarray([1], dtype=np.uint16)]
                        output['depth_scale' + np.str(affordance_id)] = [np.asarray([1000], dtype=np.uint16)]
                        output['width' + np.str(affordance_id)] = width
                        output['height' + np.str(affordance_id)] = height
                        output['fx'+ np.str(affordance_id)] = cam_fx
                        output['fy'+ np.str(affordance_id)] = cam_fy
                        output['cx'+ np.str(affordance_id)] = cam_cx
                        output['cy'+ np.str(affordance_id)] = cam_cy

                else:
                        pass

        return output
