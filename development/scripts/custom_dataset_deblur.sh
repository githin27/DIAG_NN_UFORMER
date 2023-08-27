#!/bin/bash

#creating train dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./GoPro/train/groundtruth/ \
    --patch_src_in ./GoPro/train/input/ \
    --patch_tar ./datasets/deblur/GoPro/customized_dataset/train \
    --ps 128 \
    --num_patches 50 \
    --num_cores 8
    

"""
#creating validation dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./GoPro/train/groundtruth/ \
    --patch_src_in ./GoPro/train/input/ \
    --patch_tar ./datasets/deblur/GoPro/customized_dataset/val \
    --ps 128 \
    --num_patches 20 \
    --num_cores 8


#creating test dataset
python ./generate_patch/custom_dataset_deblur.py \
    --patch_src_gt ./GoPro/test/groundtruth/ \
    --patch_src_in ./GoPro/test/input/ \
    --patch_tar ./datasets/deblur/GoPro/customized_dataset/test \
    --ps 128 \
    --num_patches 20 \
    --num_cores 8
"""

read -p "Press Enter to exit..."
