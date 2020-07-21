
import glob
import skimage.io

import os

import numpy as np
import matplotlib.pyplot as plt
import csv

###########################################################
# read csv
###########################################################

def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=10)
    return data

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

###########################################################
#
###########################################################
if __name__ == '__main__':

    ##############
    # dir
    ##############
    data_path = '/home/akeaveny/catkin_ws/src/object-rpe-ak/Mask_RCNN/samples/parts_affordance/tensorboard/real/'

    ##############
    # sub folders
    ##############
    # 1.
    splits = [
            'val/',
            'train/',
            ]

    # 2.
    losses = [
              'loss/',
              # 'class_loss/', 'mask_loss/', 'bbox_loss/'
              ]

    # init plot
    # ax = get_ax(rows=int(np.sqrt(limit)), cols=int(np.sqrt(limit)))

    colors = ['b', 'g', 'r', 'c']
    runs = ['test1', 'test2', 'test3', 'test4']
    limit = np.sqrt(len(runs)).astype(int)

    fig, ax = plt.subplots(figsize=(16, 16))
    ########################
    #
    ########################
    for loss_idx, loss in enumerate(losses):
        print('\n******************* {} *******************'.format(loss))
        for split_idx, split in enumerate(splits):
            print('\n******************* {} *******************'.format(split))

            csv_path = data_path + split + loss + '*.csv'
            csv_files = sorted(glob.glob(csv_path))
            for csv_idx, csv_file in enumerate(csv_files):
                print("csv_file: ", csv_file)

                ########################
                # read data
                ########################
                data = read_datafile(csv_file)

                time = data[:, 0]
                epoch = data[:, 1]
                X = data[:, 2]

                ########################
                # plot
                ########################
                # ax = ax[i // int(np.sqrt(limit)), i % int(np.sqrt(limit))]

                # plt.subplot(limit, limit, csv_idx+1)
                plt.subplot(2, 1, split_idx + 1)
                plt.grid()
                # plt.title("Run: {}".format(runs[csv_idx]))
                #plt.xlabel('epoch')
                #plt.ylabel(r'$L_{total}$: $L_{bbox}$ + $L_{class}$ + $L_{mask}$')
                plt.plot(epoch, X, c=colors[csv_idx],
                         alpha=0.4, linewidth=2,
                         linestyle='-' if split == 'train/' else '--',
                         ## marker='x' if split == 'train/' else 'o',
                         label='Run: {} [{}]'.format(runs[csv_idx], 'train' if split == 'train/' else 'val',))
                plt.grid(True, alpha=0.3)
                plt.legend()
                if csv_idx+1 == 1 or csv_idx+1 == 3:
                    plt.ylabel(r'$L_{total}$: $L_{bbox}$ + $L_{class}$ + $L_{mask}$')
                if csv_idx + 1 == 3 or csv_idx + 1 == 4:
                    plt.xlabel('epoch')

                ########################
                # min, max
                ########################
                min_idx, max_idx = np.argmin(X), np.argmax(X)
                # min, max = X[min_idx], X[max_idx]
                label_min, label_max = "Min:{:.3f}".format(X[min_idx]), "Max:{:.3f}".format(X[max_idx] )
                plt.scatter(epoch[min_idx], X[min_idx], s=40, marker='x', c=colors[csv_idx])
                plt.scatter(epoch[max_idx], X[max_idx], s=40, marker='x', c=colors[csv_idx])
                plt.annotate(label_min,  # this is the text
                               (epoch[min_idx], X[min_idx]),  # this is the point to label
                               textcoords="offset points",  # how to position the text
                               xytext=(-30, 0),  # distance from text to points (x,y)
                               ha='center')  # horizontal alignment can be left, right or center
                plt.annotate(label_max,  # this is the text
                             (epoch[max_idx], X[max_idx]),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center

    fig.tight_layout()
    plt.show()