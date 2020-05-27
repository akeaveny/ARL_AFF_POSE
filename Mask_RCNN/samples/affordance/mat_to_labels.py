import json
import glob
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
from PIL import Image
import imageio
import numpy as np
import os

from skimage import io
import skimage.draw

# # ===================== load mat ===========================
# mat_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real4/*_label.mat'
# print("Images: ", mat_path)
# mat_files = sorted(glob.glob(mat_path))
#
# mats = [scipy.io.loadmat(mat) for mat in mat_files]
# print("Loaded Mats: ", len(mats))
#
# for idx, mat in enumerate(mats):
#     data = np.asarray(mat['gt_label'])
#     # ============ save png ============
#     count = 100000 + idx
#     str_num = str(count)[1:]
#     # str_num = str_num[1:]
#     print(str_num)
#     print(mat_files[idx])
#     label_filename = mat_path.split("*")[0] + str_num + '_label.png'
#     print(label_filename)
#     im = Image.fromarray(data)
#     im.save(label_filename)

# # ===================== check labels ===========================
# data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/temp_Kitchen_Knife_val_real4/'
# label_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/temp_Kitchen_Knife_val_real4/*_label.png'
# label_files = sorted(glob.glob(label_path))
# print("label_files: ", len(label_files))
#
# offset = 0
# for idx, mat in enumerate(label_files):
#     idx += offset
#     image_idx = label_files[idx].split("temp_Kitchen_Knife_val_real4/")[1]
#     image_idx = image_idx.split("_label.png")[0]
#     # print(image_idx)
#
#     label_path = data_path + image_idx + "_label.png"
#     label_img = imageio.imread(label_path)
#
#     rgb_path = data_path + image_idx + "_rgb.png"
#     rgb_img = imageio.imread(rgb_path)
#
#     # ================ animation ================
#     plt.cla()
#     # for stopping simulation with the esc key.
#     plt.gcf().canvas.mpl_connect(
#         'key_release_event',
#         lambda event: [exit(0) if event.key == 'escape' else None])
#     ax1 = plt.subplot(1, 2, 1)
#     im1 = ax1.imshow(label_img)
#     ax2 = plt.subplot(1, 2, 2)
#     im2 = ax2.imshow(rgb_img)
#     plt.tight_layout()
#     plt.title(np.str(image_idx))
#     plt.show()

# # ===================== check labels ===========================

label_path = '/data/Akeaveny/Datasets/part-affordance-dataset/tests/00090_label.png'
rgb_path = '/data/Akeaveny/Datasets/part-affordance-dataset/tests/00090_rgb.png'

image = skimage.io.imread(rgb_path)
label = skimage.io.imread(label_path)

# ========== plot ============
plt.subplot(1, 2, 1)
plt.title('rgb')
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.title('label')
plt.imshow(label)
plt.ioff()
plt.pause(0.001)
plt.show()