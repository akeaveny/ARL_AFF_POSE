# AffPose: Leveraging Real and Synthetic RGB-D Datasets for Affordance Detection and 6-DoF Pose Estimation
This work is largely based on:

1. [Labelusion](https://github.com/akeaveny/LabelFusion) for generating Real Images
2. [NDDS](https://github.com/NVIDIA/Dataset_Synthesizer) for generating Synthetic Images   
3. [Mask R-CNN](https://github.com/matterport/Mask_RCNN) in Tensorflow 1.14.0 
4. [DenseFusion](https://github.com/j96w/DenseFusion) in Torch 1.0.1
5. [Object-RPE](https://github.com/hoangcuongbk80/Object-RPE) previous work that integrated Mask R-CNN with DenseFusion
6. [DenseFusionROSNode](https://github.com/akeaveny/DenseFusionROSNode) custom rospy node for running AffPose in near real time

![Alt text](Images/AffPose.png?raw=true "Title")
   
## Real UMD Dataset [Austin Myers, Ching L. Teo, Cornelia Fermüller, Yiannis Aloimonos]
The RGB-D Part Affordance Dataset dataset is avaliable [here](http://users.umiacs.umd.edu/~amyers/part-affordance-dataset/).

## Synthetic UMD Dataset
The Synthetic dataset is avaliable [here]().

## Real & Synthetic ARL Dataset
The Real dataset is avaliable [here]().
The Synthetic dataset is avaliable [here]().

## Pre-Trained Weights
Pre-trained Mask R-CNN are avaliable [here]().
Pre-trained DenseFusion are avaliable [here]().

## Requirements
1. Mask R-CNN
   ```
   $ conda env create -f environment_tensorflow114.yml --name MaskRCNN
   ```
2. DenseFusion
   ```
   $ conda env create -f environment_pytorch101.yml --name DenseFusion
   ```

## Mask R-CNN
1. To inspect dataset statistics run:
   ```
   $ python inspect_dataset_stats.py --dataset='(file path to dataset)' --dataset_type='(real or syn)' --dataset_split='val'
   ```
2. To inspect trained model run:
   ```
   $ python inspect_trained_model.py --dataset_type='(real or syn)' --detect=rgbd+ --weights='(file path to weights)'
   ```
3. To get predicted Affordance-semantic Masks run:
   ```
   $ python test.py --dataset_type='(real or syn)' --detect=rgbd+  --weights='(file path to weights)'
   ```
4. To test preformance with the weighted F-b measure run the following in MATLAB:
   ```
   $ cd '(path to project)/Mask_RCNN/matlab/'
   $ evaluate_UMD('file path to test folder')
   ```
## DenseFusion
1. To inspect dataset run:
   ```
   $ python project_points.py
   ```
2. To get predicted pose run:
   ```
   $ python inference_arl.py
   ```
3. To get evaluation metrics run:
   ```
   $ python YCB_toolbox_plot.py
   ```   


