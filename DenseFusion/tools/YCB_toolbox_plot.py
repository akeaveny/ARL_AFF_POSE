#! /usr/bin/env python

import argparse
import os
import sys
import glob
import math
import numpy as np
import scipy.io as scio
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import auc, average_precision_score, roc_auc_score

# ROOT_DIR = os.path.abspath("/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/")
ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)
print("ROOT_DIR", ROOT_DIR)

MIN_DISTANCE = 1/10000 # for formatting plot
MAX_DISTANCE = 10/100 # 10 [cm]
THRESHOLD = 2/100 # 2 [cm]

##################################
## GPU
##################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############################################################
#  argparse
############################################################
parser = argparse.ArgumentParser(description='Evaluate trained model for DenseFusion')

parser.add_argument('--dataset_config', required=False,
                    # default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/ycb/dataset_config/',
                    # default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config',
                    default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/datasets/arl/dataset_config',
                    type=str,
                    metavar="")

parser.add_argument('--classes', required=False, default='classes.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--class_ids', required=False, default='classes_ids.txt',
                    metavar="/path/to/weights.h5 or 'coco'")

parser.add_argument('--error_metrics_dir', required=False,
                    # default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/eval_result/ycb/PoseCNN_error_metrics_result/',
                    # default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/eval_result/ycb/Densefusion_error_metrics_result/',
                    default='/home/akeaveny/catkin_ws/src/object-rpe-ak/DenseFusion/experiments/eval_result/arl/Densefusion_error_metrics_result/',
                    type=str,
                    metavar="Data Path")

parser.add_argument('--visualize', required=False, default=False,
                    type=str,
                    metavar="Visualize Results")

parser.add_argument('--save_images_path', required=False,
                    # default='/data/Akeaveny/Datasets/ycb_syn/test_densefusion/',
                    # default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/test_densefusion_real/',
                    # default='/data/Akeaveny/Datasets/arl_scanned_objects/ARL/test_densefusion_syn/',
                    default='/data/Akeaveny/Datasets/arl_dataset/test_densefusion_real/',
                    type=str,
                    metavar="Visualize Results")

args = parser.parse_args()

##################################
## classes
##################################

class_file = open('{}/{}'.format(args.dataset_config, args.classes))
class_id_file = open('{}/{}'.format(args.dataset_config, args.class_ids))
classes = np.loadtxt(class_file, dtype=np.str)
class_IDs = np.loadtxt(class_id_file, dtype=np.int32)
### print("Classes: ", class_IDs)

##################################
## aggregate meta
##################################

meta_list = sorted(glob.glob(args.error_metrics_dir + "*.mat"))

history = []
for meta_addr in meta_list:
    meta = scio.loadmat(meta_addr)
    if len(meta['Class_IDs']) == 0:
        pass
    else:
        class_ids = meta['Class_IDs'][0]
        for idx, class_id in enumerate(class_ids):
            history_ = np.array([
                class_id,
                meta['ADD'][0][idx],
                meta['ADD_S'][0][idx],
                meta['T'][0][idx],
                meta['R'][0][idx],
            ])
            history.append(history_)
history = np.asarray(history)

df = pd.DataFrame({'Class_Id': history[:, 0],
                    'ADD': history[:, 1],
                    'ADD_S': history[:, 2],
                    'T': history[:, 3],
                    'R': history[:, 4]})

##################################
# AUC Curves
##################################

detected_class_IDs = np.unique(df['Class_Id'].values)

MEAN_ADD = np.zeros(shape=len(detected_class_IDs))
MEAN_ADD_AUC = np.zeros(shape=len(detected_class_IDs))
MEAN_ADDS = np.zeros(shape=len(detected_class_IDs))
MEAN_ADDS_AUC = np.zeros(shape=len(detected_class_IDs))

for mean_idx, ycb_idx in enumerate(detected_class_IDs):
    print("*************** {} ***************".format(classes[int(ycb_idx)-1]))
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    plt.suptitle("{}".format(classes[int(ycb_idx)-1]), fontsize=14)

    # find all
    row_idx = df.index[df['Class_Id'] == ycb_idx].tolist()
    ADD = df['ADD'][row_idx].values
    ADD_S = df['ADD_S'][row_idx].values
    T = df['T'][row_idx].values
    R = df['R'][row_idx].values

    ##############
    # ADD
    ##############

    ADD[ADD > MAX_DISTANCE] = np.inf
    ADD = np.sort(ADD)

    idx_ = []
    for i in range(len(ADD)):
        if math.isfinite(ADD[i]):
            idx_.append(i)
    ADD_auc = ADD[idx_]

    c = len(ADD[ADD < THRESHOLD])

    if max(ADD_auc) < MAX_DISTANCE:
        ADD_auc = np.hstack((ADD_auc, np.array([MAX_DISTANCE])))
        ADD = np.hstack((ADD, np.array([MAX_DISTANCE])))

    n = len(ADD)
    accuracy = np.cumsum(np.ones(shape=(1, n))) / n
    accuracy_auc = np.cumsum(np.ones(shape=(1, len(ADD_auc)))) / len(ADD_auc)

    metrics = [c / n * 100, auc(ADD_auc, accuracy_auc) / (MAX_DISTANCE) * 100]
    title = 'ADD\nADD<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(metrics[0], metrics[1])
    print('ADD<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(metrics[0], metrics[1]))

    MEAN_ADD[int(mean_idx)] = metrics[0]
    MEAN_ADD_AUC[int(mean_idx)] = metrics[1]

    # ADD
    ax[0, 0].plot(np.sort(ADD), accuracy, '-x', color='blue', linewidth=2)
    ax[0, 0].set_xlim(MIN_DISTANCE, MAX_DISTANCE)
    ax[0, 0].axvline(x=0.02, c='blue', linewidth=1, linestyle='--')
    ax[0, 0].set_title(title)

    ax[0, 0].grid()

    ##############
    # ADD - S
    ##############

    ADD_S[ADD_S > MAX_DISTANCE] = np.inf
    ADD_S = np.sort(ADD_S)

    idx_ = []
    for i in range(len(ADD_S)):
        if math.isfinite(ADD_S[i]):
            idx_.append(i)
    ADD_S_auc = ADD_S[idx_]

    c = len(ADD_S[ADD_S < THRESHOLD])

    if max(ADD_S_auc) < MAX_DISTANCE:
        ADD_S_auc = np.hstack((ADD_S_auc, np.array([MAX_DISTANCE])))
        ADD_S = np.hstack((ADD_S, np.array([MAX_DISTANCE])))

    n = len(ADD_S)
    accuracy = np.cumsum(np.ones(shape=(1, n))) / n
    accuracy_auc = np.cumsum(np.ones(shape=(1, len(ADD_S_auc)))) / len(ADD_S_auc)

    metrics = [c / n * 100, auc(ADD_S_auc, accuracy_auc) / (MAX_DISTANCE) * 100]
    title = 'ADD-S\nADD-S<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(metrics[0], metrics[1])
    print('ADD-S<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(metrics[0], metrics[1]))

    MEAN_ADDS[int(mean_idx)] = metrics[0]
    MEAN_ADDS_AUC[int(mean_idx)] = metrics[1]

    # ADD-S
    ax[0, 1].plot(ADD_S, accuracy, '-x', color='magenta', linewidth=2)
    ax[0, 1].set_xlim(MIN_DISTANCE, MAX_DISTANCE)
    ax[0, 1].axvline(x=0.02, c='magenta', linewidth=1, linestyle='--')
    ax[0, 1].set_title(title)
    ax[0, 1].grid()

    ##############
    # T
    ##############

    # _T[_T > MAX_DISTANCE] = np.inf
    T = np.sort(T)
    n = len(T)
    accuracy = np.cumsum(np.ones(shape=(1, n))) / n

    ax[1, 0].plot(T, accuracy, '-x', color='yellow', linewidth=2)
    ax[1, 0].set(xlabel='T [m]', ylabel='accuracy', title='Translation Error')

    ax[1, 0].grid()

    ##############
    # R
    ##############

    # _ADD[_ADD > MAX_DISTANCE] = np.inf
    R = np.sort(R)
    n = len(R)
    accuracy = np.cumsum(np.ones(shape=(1, n))) / n

    ax[1, 1].plot(R, accuracy, '-x', color='green', linewidth=2)
    ax[1, 1].set(xlabel='R [deg]', ylabel='accuracy', title='Rotation Error')

    ax[1, 1].grid()

    # plt.xticks(np.arange(first_id-1,last_id+1,40))
    # plt.yticks(np.arange(0, max(train_dis_trend_r), step=0.01))
    plt.ticklabel_format(style='sci', axis='x')

    if args.visualize:
        plt.show()

print("\n*************** MEAN RESULTS ***************")
print('ADD<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(np.mean(MEAN_ADD), np.mean(MEAN_ADD_AUC)))
print('ADD-S<2cm = {:.2f} [%], AUC = {:.2f} [%]'.format(np.mean(MEAN_ADDS), np.mean(MEAN_ADDS_AUC)))