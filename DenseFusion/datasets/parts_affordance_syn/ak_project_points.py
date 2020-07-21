import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/parts_affordance_syn/dataset_config/class_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

# TODO:
cld = {}
for class_id in class_IDs:
    print("class_id: ", class_id)
    class_input = class_file.readline()
    print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/part-affordance_combined/ndds2/models/{0}/{0}_grasp.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    cld[class_id] = np.array(cld[class_id]) * 1e3  # TODO:
    input_file.close()

# =================== argparse ========================
# import argparse
#
# parser = argparse.ArgumentParser(description='Compute Image Mean and Stddev')
# parser.add_argument('--dataset', required=True,
#                     metavar="/path/to//dataset/",
#                     help='Directory of the dataset')
# args = parser.parse_args()

data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds2/combined_tools_val/dr'
folder_to_save = 'combined_tools_train/dr/'
label_format = '_label.png'

gt_label_addr = data_path + '/??????' + '_label.png'
files = sorted(glob.glob(gt_label_addr))
print("Loaded files: ", len(files))


for file in files:
    # count = 1000000 + image_idx
    # img_number = str(count)[1:]

    str_num = file.split(data_path + '/')[1]
    img_number = str_num.split('_label.png')[0]

    label_addr = data_path + img_number + label_format

    img = Image.open('{0}/{1}_rgb.png'.format(data_path, img_number))
    # depth = np.array(Image.open('{0}/{1}_depth.png'.format(data_path, img_number)))
    label = np.array(Image.open('{0}/{1}_label.png'.format(data_path, img_number)))
    meta = scio.loadmat('{0}/{1}-meta.mat'.format(data_path, img_number))

    affordance_ids = np.array(meta['Affordance_ID'].astype(np.int32))

    for affordance_id in affordance_ids:

        if affordance_id in class_IDs:

            idx = affordance_id
            idx = '0' + np.str(idx)
            print("img_number", img_number)
            print("idx: ", idx)
            # ================== meta ================
            cam_rotation4 = np.dot(np.array(meta['rot' + np.str(idx)]), np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]))
            cam_translation = np.array(meta['cam_translation' + np.str(idx)][0])

            cam_rotation = np.array(meta['rot' + np.str(idx)])

            cam_cx = meta['cx' + np.str(idx)][0][0]
            cam_cy = meta['cy' + np.str(idx)][0][0]
            cam_fx = meta['fx' + np.str(idx)][0][0]
            cam_fy = meta['fy' + np.str(idx)][0][0]

            # ===================== SCREEN POINTS =====================
            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            imgpts, jac = cv2.projectPoints(cld[affordance_id],
                                            np.dot(cam_rotation4.T, np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])),
                                            cam_translation / 10,
                                            cam_mat, dist)
            cv2_img_rot = cv2.polylines(np.array(img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

            # # ========== plot ============
            plt.subplot(3, 2, 1)
            plt.title('og')
            plt.imshow(cv2_img_rot)

            # plt.subplot(3, 2, 3)
            # plt.title('1')
            # plt.imshow(cv2_img_rot1)

            plt.ioff()
            plt.pause(0.001)
            plt.show()

        else:
            pass
