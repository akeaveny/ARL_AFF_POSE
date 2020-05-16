import json
import glob
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
from PIL import Image
import imageio
import numpy as np

# '/data/Akeaveny/Datasets/part-affordance-dataset/combined/19773_label.mat'
# ===================== load mat ===========================
mat_path = '/data/Akeaveny/Datasets/part-affordance-dataset/combined/*_label.mat'
print("Images: ", mat_path)
mats = [scipy.io.loadmat(mat) for mat in glob.glob(mat_path)]
print("Loaded Mats: ", len(mats))
# print("First Mat \n", mats[0])

for idx, mat in enumerate(mats):
    data = np.asarray(mat['gt_label'])
    # ============ save png ============
    str_num = np.str(idx)
    # print(str_num)
    # str_num = str_num[1:]
    label_filename = mat_path.split("*")[0] + str_num + '_label.png'
    # print(str_to_save)
    im = Image.fromarray(data)
    im.save(label_filename)

# ===================== check labels ===========================
label_path = '/data/Akeaveny/Datasets/part-affordance-dataset/bowl/*_label.png'
print("Images: ", label_path)
label_imgs = [imageio.imread(label) for label in glob.glob(label_path)]
print("Loaded Labels: ", len(label_imgs))

# label_img = cv2.imread(label_path)
# cv2.imshow('label', label_img)

for label_img in label_imgs:
    plt.imshow(label_img)
    plt.show()
