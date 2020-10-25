import cv2
import glob
import numpy as np
from PIL import Image
import scipy.io as scio
import matplotlib.pyplot as plt

class_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/classes_train.txt')
class_id_file = open('/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/class_ids_train.txt')
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)

cld = {}
for class_id in class_IDs:
    print("class_id: ", class_id)
    class_input = class_file.readline()
    print("class_input: ", class_input)
    if not class_input:
        break
    input_file = open('/data/Akeaveny/Datasets/arl_scanned_objects/ARL/models/{0}/{0}.xyz'.format(class_input[:-1]))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1].split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    cld[class_id] = np.array(cld[class_id])
    input_file.close()

##################################
## LOAD IMAGES
##################################
dataset = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/'
dataset_config = '/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl_real/dataset_config/'
# images_file = 'train_data_list.txt'             # REAL
images_file = 'train_data_list_combined.txt'    # COMBINED

loaded_images_ = np.loadtxt('{}/{}'.format(dataset_config, images_file), dtype=np.str)

# select random test images
np.random.seed(1)
num_test = 1000
test_idx = np.random.choice(np.arange(0, int(len(loaded_images_)), 1), size=int(num_test), replace=False)
print("Chosen Files \n", len(test_idx))

for idx in test_idx:

    ##############
    ##############

    str_num = loaded_images_[idx].split('/')[-1]

    rgb_addr = dataset + loaded_images_[idx] + "_rgb.png"
    depth_addr = dataset + loaded_images_[idx] + "_depth.png"
    gt_addr = dataset + loaded_images_[idx] + "_label.png"
    mask_addr = gt_addr

    print("rgb addr: ", rgb_addr)

    # gt pose
    meta_addr = dataset + loaded_images_[idx] + "-meta.mat"
    meta = scio.loadmat(meta_addr)

    img = np.array(Image.open(rgb_addr))
    depth = np.array(Image.open(depth_addr))
    gt = np.array(Image.open(gt_addr))
    label = np.array(Image.open(mask_addr))

    # rgb
    img = np.array(img)
    if img.shape[-1] == 4:
        image = img[..., :3]

    affordance_ids = np.unique(np.array(label))
    for affordance_id in affordance_ids:
        if affordance_id in class_IDs:

            idx = affordance_id
            count = 1000 + idx
            meta_idx = str(count)[1:]

            ############
            # meta
            ############

            height = meta['width' + meta_idx].flatten().astype(np.int32)[0]
            width = meta['height' + meta_idx].flatten().astype(np.int32)[0]

            xmap = np.array([[j for i in range(height)] for j in range(width)])
            ymap = np.array([[i for i in range(height)] for j in range(width)])

            cam_cx = meta['cx' + meta_idx][0][0]
            cam_cy = meta['cy' + meta_idx][0][0]
            cam_fx = meta['fx' + meta_idx][0][0]
            cam_fy = meta['fy' + meta_idx][0][0]

            cam_scale = np.array(meta['camera_scale' + meta_idx])[0][0]  # 1000 for [mm] to [m]
            border_list = np.array(meta['border' + meta_idx]).flatten().astype(np.int32)

            cam_rotation4 = np.array(meta['rot' + meta_idx])
            cam_translation = np.array(meta['cam_translation' + meta_idx][0])  # in [m]

            #######################################
            # PROJECT TO SCREEN
            #######################################

            cam_mat = np.array([[cam_fx, 0, cam_cx], [0, cam_fy, cam_cy], [0, 0, 1]])
            dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            print("cam_mat: \n", cam_mat)

            imgpts, jac = cv2.projectPoints(cld[affordance_id] * 1e3,
                                            cam_rotation4,
                                            cam_translation * 1e3,
                                            cam_mat, dist)
            cv2_img_rot = cv2.polylines(np.array(img), np.int32([np.squeeze(imgpts)]), True, (0, 255, 255))

            #######################################
            # PLOT
            #######################################

            plt.subplot(1, 2, 1)
            plt.title('og')
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.title('project points')
            plt.imshow(cv2_img_rot)
            plt.ioff()
            plt.pause(0.001)
            plt.show()

        else:
            pass
