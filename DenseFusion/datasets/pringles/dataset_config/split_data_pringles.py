import glob
import numpy as np

f_train = open("train_data_list_zed.txt", 'w')
# ============= training ==================
label_images = []
label_addrs = '/data/Akeaveny/Datasets/pringles/zed/train/*.cs.png'
images = [label_images.append(file) for file in sorted(glob.glob(label_addrs))]
print("Loaded Images: ", len(images))

for label_image in label_images[0:10]:
    img_index_str = label_image.split("zed/")[1]
    img_index_str = img_index_str.split(".cs")[0]
    print(img_index_str)
    f_train.write(img_index_str)
    f_train.write('\n')

f_val = open("test_data_list_zed.txt", 'w')
# ============= training ==================
label_images = []
label_addrs = '/data/Akeaveny/Datasets/pringles/zed/val/*.cs.png'
images = [label_images.append(file) for file in sorted(glob.glob(label_addrs))]
print("Loaded Images: ", len(images))

for label_image in label_images[0:10]:
    img_index_str = label_image.split("zed/")[1]
    img_index_str = img_index_str.split(".cs")[0]
    print(img_index_str)
    f_val.write(img_index_str)
    f_val.write('\n')

