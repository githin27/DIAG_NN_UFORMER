#!/bin/bash

#creating train dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./rough_GoPro/train/groundtruth/ \
    --patch_src_in ./rough_GoPro/train/input/ \
    --patch_tar ./datasets/rough/deblur/train \
    --ps 64 \
    --num_patches 5 \
    --num_cores 2
    

"""
creating validation dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./rough_GoPro/train/groundtruth/ \
    --patch_src_in ./rough_GoPro/train/input/ \
    --patch_tar ./datasets/rough/deblur/val \
    --ps 128 \
    --num_patches 20 \
    --num_cores 2


creating test dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./rough_GoPro/train/groundtruth/ \
    --patch_src_in ./rough_GoPro/train/input/ \
    --patch_tar ./datasets/rough/deblur \
    --ps 128 \
    --num_patches 20 \
    --num_cores 2

read -p "Press Enter to exit..."
