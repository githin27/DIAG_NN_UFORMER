#!/bin/bash

python ./train/train_denoise.py \
    --pretrain_weights ./models/pretrained/denoise/model_best.pth \
    --train_dir ./datasets/denoise/SIDD/customized_dataset/train \
    --val_dir ./datasets/denoise/SIDD/customized_dataset/val \
    --model_dir ./models/training/denoise \
    --log_dir ./log/denoise \
    --nepoch 2 \
    --gpu 0 
    
    
    
read -p "Press Enter to exit..."