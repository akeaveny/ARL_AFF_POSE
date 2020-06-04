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
#
# # # ===================== load mat ===========================
# mat_path = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-dataset/tools_combined3/*_label*'
# # mat_path = '/data/Akeaveny/Datasets/part-affordance-dataset/real/part-affordance-clutter/clutter_combined/*_label*'
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
#     label_filename = mat_path.split("*")[0] + str_num + '_label.png'
#     # print(str_num)
#     # print(mat_files[idx])
#     # print(label_filename)
#     im = Image.fromarray(data)
#     im.save(label_filename)

# # ===================== FUSE RGB + D ===========================
folder_to_load = 'formatted_syn_val/'
data_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/formatted_syn_val/'
label_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/formatted_syn_val/*_label.png'
label_files = sorted(glob.glob(label_path))
print("label_files: ", len(label_files))

offset = 0
for idx, mat in enumerate(label_files):
    idx += offset
    image_idx = label_files[idx].split(folder_to_load)[1]
    image_idx = image_idx.split("_label.png")[0]
    # print(image_idx)

    label_path = data_path + image_idx + "_label.png"
    label_img = imageio.imread(label_path)

    rgb_path = data_path + image_idx + "_rgb.png"
    rgb_img = np.array(imageio.imread(rgb_path))
    print("rgb_img: ", rgb_img.shape)

    depth_path = data_path + image_idx + "_depth.png"
    depth_img = np.array(imageio.imread(depth_path))
    # ===== normalize =====
    # depth_img /= np.amax(depth_img) # 2^16 - 1
    depth_img = depth_img / (2**16 - 1) # 2^16 - 1
    depth_img = np.array(depth_img * 255, dtype=np.uint8)
    print("depth_img: ", depth_img.shape)
    print("min: ", np.amin(depth_img))
    print("max: ", np.amax(depth_img))

    rgb_depth_img = rgb_img.copy()
    rgb_depth_img[:,:,-1] = depth_img

    fused_path = data_path + image_idx + "_fused.png"
    imageio.imwrite(fused_path, rgb_depth_img)
    rgb_depth_img_  = np.array(imageio.imread(fused_path))

    # # ================ animation ================
    # plt.cla()
    # # for stopping simulation with the esc key.
    # plt.gcf().canvas.mpl_connect(
    #     'key_release_event',
    #     lambda event: [exit(0) if event.key == 'escape' else None])
    # ax1 = plt.subplot(2, 2, 1)
    # im1 = ax1.imshow(label_img)
    # ax2 = plt.subplot(2, 2, 2)
    # im2 = ax2.imshow(depth_img)
    # ax3 = plt.subplot(2, 2, 3)
    # im3 = ax3.imshow(rgb_depth_img_)
    # ax4 = plt.subplot(2, 2, 4)
    # im4 = ax4.imshow(rgb_depth_img_[..., :3])
    # plt.tight_layout()
    # plt.title(np.str(image_idx))
    # plt.show()