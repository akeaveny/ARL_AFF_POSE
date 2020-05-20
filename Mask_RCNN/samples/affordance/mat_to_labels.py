import json
import glob
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
from PIL import Image
import imageio
import numpy as np

# ===================== load mat ===========================
# mat_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real/*_label.mat'
# print("Images: ", mat_path)
# mats = [scipy.io.loadmat(mat) for mat in sorted(glob.glob(mat_path))]
# print("Loaded Mats: ", len(mats))
# # print("First Mat \n", mats[0])
#
# for idx, mat in enumerate(mats):
#     data = np.asarray(mat['gt_label'])
#     # ============ save png ============
#     str_num = np.str(idx)
#     # print(str_num)
#     # str_num = str_num[1:]
#     label_filename = mat_path.split("*")[0] + str_num + '_label.png'
#     # print(str_to_save)
#     im = Image.fromarray(data)
#     im.save(label_filename)

# ===================== check labels ===========================
label_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real/*_label.png'
print("Images: ", label_path)
label_imgs = [imageio.imread(label) for label in sorted(glob.glob(label_path))]
print("Loaded Labels: ", len(label_imgs))

# /data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/val/289_rgb.png
rgb_path = '/data/Akeaveny/Datasets/part-affordance-dataset/ndds_and_real/Kitchen_Knife_val_real/*_rgb.png'
print("Images: ", label_path)
rgb_imgs = [imageio.imread(label) for label in sorted(glob.glob(rgb_path))]
print("Loaded RGBs: ", len(rgb_imgs))

# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
for idx, _ in enumerate(label_imgs):
    # ax1.imshow(label_imgs[idx])
    # ax2.imshow(rgb_imgs[idx])

    # ================ animation ================
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])
    plt.imshow(label_imgs[idx])
    plt.show()
