# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger

from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from datasets.parts_affordance.dataset import PoseDataset as PoseDataset_syn
from datasets.ycb_syn.dataset import PoseDataset as PoseDataset_ycb_syn
from datasets.arl_real.dataset import PoseDataset as PoseDataset_arl_real
from datasets.arl_syn.dataset import PoseDataset as PoseDataset_arl_syn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb-syn', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default ='', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default =8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=1e-5, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--repeat_epoch', type=int, default=1, help='')
parser.add_argument('--resume_posenet', type=str, default= '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
opt = parser.parse_args()

###################
# GPU
###################
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        output_results = 'check_linemod.txt'
        opt.repeat_epoch = 20

    elif opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 1 #number of repeat times for one epoch training

    elif opt.dataset == 'ycb-syn':
        opt.num_objects = 31  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.dataset_root = '/data/Akeaveny/Datasets/ycb_syn'
        opt.outf = 'trained_models/ycb_syn/ycb_syn2'  # folder to save trained models
        opt.log_dir = 'experiments/logs/ycb_syn/ycb_syn2'  # folder to save logs
        output_results = 'check_ycb_syn.txt'

        opt.w = 0.05
        opt.refine_margin = 0.01

        # opt.start_epoch = 20
        # opt.resume_posenet = 'pose_model_17_0.009960328408820027.pth'
        ## resume_refinenet = 'pose_refine_model_247_0.04728509413062927.pth'

    elif opt.dataset == 'arl-real':
        opt.num_objects = 4  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.dataset_root = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL'
        opt.outf = 'trained_models/arl_real/arl3'  # folder to save trained models
        opt.log_dir = 'experiments/logs/arl_real/arl3'  # folder to save logs
        test_folder = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL/test_densefusion_real/'
        output_results = 'check_arl_real.txt'

        opt.w = 0.05
        opt.refine_margin = 0.01

        opt.start_epoch = 20
        opt.resume_posenet = 'pose_model_17_0.009960328408820027.pth'
        ## resume_refinenet = 'pose_refine_model_247_0.04728509413062927.pth'

    elif opt.dataset == 'arl-syn':
        opt.num_objects = 4  # number of object classes in the dataset
        opt.num_points = 1000  # number of points on the input pointcloud
        opt.dataset_root = '/data/Akeaveny/Datasets/arl_scanned_objects/ARL'
        opt.outf = 'trained_models/arl_syn/arl1'  # folder to save trained models
        opt.log_dir = 'experiments/logs/arl_syn/arl1'  # folder to save logs
        output_results = 'check_arl_syn.txt'

        opt.w = 0.05
        opt.refine_margin = 0.01

        # opt.start_epoch = 25
        # opt.resume_posenet = 'pose_model_23_0.05664153265627049.pth'
        ## resume_refinenet = 'pose_refine_model_247_0.04728509413062927.pth'

    elif opt.dataset == 'parts-affordance':
        print(opt.dataset)
        opt.num_objects = 205
        opt.num_points = 100
        opt.outf = 'trained_models/parts_affordance/hammer_umd1'
        opt.log_dir = 'experiments/logs/parts_affordance/hammer_umd1'
        output_results = 'check_parts_affordance.txt'
        opt.repeat_epoch = 1

    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'parts-affordance':
        dataset = PoseDataset_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'ycb-syn':
        dataset = PoseDataset_ycb_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'arl-real':
        dataset = PoseDataset_arl_real('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'arl-syn':
        dataset = PoseDataset_arl_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'parts-affordance':
        test_dataset = PoseDataset_syn('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'ycb-syn':
        test_dataset = PoseDataset_ycb_syn('test', opt.num_points, True, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'arl-real':
        test_dataset = PoseDataset_arl_real('test', opt.num_points,  True, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'arl-syn':
        test_dataset = PoseDataset_arl_syn('test', opt.num_points,  True, opt.dataset_root, 0.0, opt.refine_start)

    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    ######################
    ######################

    # TODO (ak): set up tensor board
    # if not os.path.exists(opt.log_dir):
    #     os.makedirs(opt.log_dir)
    #
    # writer = SummaryWriter(opt.log_dir)

    ######################
    ######################

    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        if opt.refine_start:
            estimator.eval()
            refiner.train()
        else:
            estimator.train()
        optimizer.zero_grad()

        for rep in range(opt.repeat_epoch):

            ##################
            # train
            ##################

            for i, data in enumerate(dataloader, 0):
                points, choose, img, target, model_points, idx = data

                # TODO: txt file
                # fw = open(test_folder + output_results, 'w')
                # fw.write('Points\n{0}\n\nchoose\n{1}\n\nimg\n{2}\n\ntarget\n{3}\n\nmodel_points\n{4}'.format(points, choose, img, target, model_points))
                # fw.close()

                points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                
                if opt.refine_start:
                    for ite in range(0, opt.iteration):
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        dis.backward()
                else:
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                if train_count % opt.batch_size == 0:
                    logger.info('Train time {} Epoch {} Batch {} Frame {} Avg_dis: {:.2f} [cm]'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size * 100))
                    optimizer.step()
                    optimizer.zero_grad()

                    # TODO: tensorboard
                    # if train_count != 0 and train_count % 250 == 0:
                    #     scalar_info = {'loss': loss.item(),
                    #                    'dis': train_dis_avg / opt.batch_size}
                    #     for key, val in scalar_info.items():
                    #         writer.add_scalar(key, val, train_count)

                    train_dis_avg = 0

                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

                    # TODO: tensorboard
                    # scalar_info = {'loss': loss.item(),
                    #                'dis': dis.item()}
                    # for key, val in scalar_info.items():
                    #     writer.add_scalar(key, val, train_count)

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        estimator.eval()
        refiner.eval()

        for j, data in enumerate(testdataloader, 0):
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            test_dis += dis.item()
            logger.info('Test time {} Test Frame No.{} dis: {} [cm]'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis * 100))

            test_count += 1

        test_dis = test_dis / test_count
        logger.info('Test time {} Epoch {} TEST FINISH Avg dis: {} [cm]'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis * 100))

        # TODO: tensorboard
        # scalar_info = {'test dis': test_dis}
        # for key, val in scalar_info.items():
        #     writer.add_scalar(key, val, train_count)

        if test_dis <= best_test:
            best_test = test_dis
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'parts-affordance':
                dataset = PoseDataset_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'ycb-syn':
                dataset = PoseDataset_ycb_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'arl-real':
                dataset = PoseDataset_arl_real('train', opt.num_points, True, opt.dataset_root, opt.noise_trans,opt.refine_start)
            elif opt.dataset == 'arl-syn':
                dataset = PoseDataset_arl_syn('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)

            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'parts-affordance':
                test_dataset = PoseDataset_syn('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'ycb-syn':
                test_dataset = PoseDataset_ycb_syn('test', opt.num_points, True, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'arl-real':
                test_dataset = PoseDataset_arl_real('test', opt.num_points, True, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'arl-syn':
                test_dataset = PoseDataset_arl_syn('test', opt.num_points, True, opt.dataset_root, 0.0, opt.refine_start)

            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()
