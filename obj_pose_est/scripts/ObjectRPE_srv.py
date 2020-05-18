#!/usr/bin/env python

from obj_pose_est.srv import *
import rospy
import os

def handle_ObjectRPE(req):
    mask_dir = req.data_dir + '/train_combined' + '/mask/*.png'
    mask_color_dir = req.data_dir + '/train_combined' + '/mask-color/*.png'

    # os.system('rm -r ' + mask_dir)
    # os.system('rm -r ' + mask_color_dir)

    #--------------------------Start MaskRCNN---------------------------
    print("Mask-RCNN running ...")

    executed_file = 'train_ycb.py'
    #  os.path.join(req.ObjectRPE_dir, 'Mask_RCNN/samples/ycb/test_ycb.py') 
    maskrcnn_model_dir = ' --weights=' + '/home/akeaveny/catkin_ws/src/Object-RPE/DenseFusion/trained_models/ycb/mask_rcnn_ycb_0040.h5'
    num_frames = ' --num_frames=' + str(req.num_frames)
    data_dir = ' --data=' + req.data_dir
    os.chdir('/home/akeaveny/catkin_ws/src/Object-RPE/Mask_RCNN/samples/ycb/')
    aa = os.popen('python3 ' + executed_file  + maskrcnn_model_dir + data_dir + num_frames).read()

    #--------------------------Start DenseFusion---------------------------
    print("DenseFusion running ...")

    densefusion_dir = '/home/akeaveny/catkin_ws/src/Object-RPE/DenseFusion' 
    executed_file = densefusion_dir + '/tools/inference_ycb.py'  

    dataset_root = ' --dataset_root ' + req.data_dir
    saved_root = ' --saved_root ' + req.data_dir
    
    pose_model = ' --model ' + '/home/akeaveny/catkin_ws/src/Object-RPE/DenseFusion/trained_models/ycb/pose_model_22_0.012934861789403338.pth'
    pose_refine_model = ' --refine_model ' + '/home/akeaveny/catkin_ws/src/Object-RPE/DenseFusion/trained_models/ycb/pose_refine_model_29_0.012507753662475113.pth'
    num_frames = ' --num_frames ' + str(req.num_frames)

    os.chdir(densefusion_dir)
    aa = os.popen('python3 ' + executed_file + dataset_root + saved_root + pose_model + pose_refine_model + num_frames).read()

    return 1;

def ObjectRPE_server():
    rospy.init_node('ObjectRPE_server')
    s = rospy.Service('Seg_Reconst_PoseEst', ObjectRPE, handle_ObjectRPE)
    print("Ready to run ObjectRPE.")
    rospy.spin()

if __name__ == "__main__":
    print('The current working directory:')
    print(os.getcwd())
    ObjectRPE_server()