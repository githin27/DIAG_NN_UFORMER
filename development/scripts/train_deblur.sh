#!/bin/bash

python ./train/train_deblur.py \
    --pretrain_weights ./models/pretrained/deblur/model_best.pth \
    --train_dir ./datasets/deblur/GoPro/customized_dataset/train \
    --val_dir ./datasets/deblur/GoPro/customized_dataset/val \
    --model_dir ./models/training/deblur \
    --log_dir ./log/deblur \
    --nepoch 2 \
    --gpu 0  
    
read -p "Press Enter to exit..."