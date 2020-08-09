# Affordance-semantic Labelling and 6-DoF PoseEstimation
This work is largely based on:
1. [Mask R-CNN](https://github.com/matterport/Mask_RCNN) in Tensofflow 1.14.0 
2. [DenseFusion](https://github.com/j96w/DenseFusion) in Torch 1.4.0

## Synthetic UMD Dataset
The synthentic dataset is avaliable [here](https://drive.google.com/file/d/1ffP3N0ZVzPAGjTGMdSS1_40JPadBOayS/view?usp=sharing).

## Pre-Trained Weights
Pre-trained Mask R-CNN and DenseFusion weights for the hammer object are avaliable [here](https://drive.google.com/file/d/1ffP3N0ZVzPAGjTGMdSS1_40JPadBOayS/view?usp=sharing).

## Env
Three conda env were used: 1. Mask R-CNN, 2. DenseFusion and 3. DenseFusion-ROS. Requirement files are included.

## Mask R-CNN
1. To inspect dataset statistics run:
   ```
   $ python3 inspect_dataset_stats.py --dataset='(file path to synthetic dataset)' --dataset_type='hammer' --dataset_split='val'
   ```
2. To inspect trained model run:
   ```
   $ python inspect_trained_model.py --dataset_type='hammer' --detect=rgbd --weights='(file path to weights)'
   ```
3. To get predicted Affordance-semantic Masks run:
   ```
   $ python test.py --dataset_type='syn' --detect=rgbd  --weights='(file path to weights)'
   ```
## DenseFusion
1. To inspect dataset statistics run:
   ```
   $ python inference_parts_affordance.py
   ```

## Running Live ROS Node
A camera (e.g. Stereolabs Zed) is needed to run a live demo.

   ```
   $ roslaunch zed_wrapper zed.launch
   $ roslaunch densefusion_ros densefusion_ros.launch
   ```

