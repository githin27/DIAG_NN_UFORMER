#!/bin/bash

#creating train dataset
python ./generate_patch/custom_dataset_denoise.py \
    --patch_src ./SIDD_Medium_Srgb/Data \
    --patch_tar ./datasets/denoise/SIDD/customized_dataset/ \
    --ps 128 \
    --num_patches 50 \
    --num_cores 8
    
read -p "Press Enter to exit..."

"""
creating validation dataset
python ./generate_patch/custom_dataset_deoise.py \
    --patch_src ./SIDD_Medium_Srgb/Data \
    --patch_tar ./datasets/denoise/SIDD/customized_dataset/ \
    --ps 128 \
    --num_patches 20 \
    --num_cores 8


creating test dataset
python ./generate_patch/custom_dataset_denoise.py \
    --patch_src ./SIDD_Medium_Srgb/Data \
    --patch_tar ./datasets/denoise/SIDD/customized_dataset/ \
    --ps 128 \
    --num_patches 20 \
    --num_cores 8
"""

