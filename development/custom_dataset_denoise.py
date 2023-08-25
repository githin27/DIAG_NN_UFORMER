# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:13:59 2023

@author: githinNote:
    * Download the dataset from https://www.eecs.yorku.ca/~kamel/sidd/dataset.php
    * Save the dataset in ./datasets/denoise/SIDD
"""


from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse


#patching both files and saving
def save_files(i):
    noisy_file, clean_file = files_in[i], files_gt[i]
    noisy_img = cv2.imread(os.path.join(src_in, noisy_file))
    clean_img = cv2.imread(os.path.join(src_gt, clean_file))

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    
    # Generate multiple patches from each image
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        
        # Define paths for saving patches
        noisy_dir=noisy_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'
        clear_dir=clean_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'

        # Save the patches as image files
        cv2.imwrite(noisy_dir, noisy_patch) 
        cv2.imwrite(clear_dir, clean_patch)

# Define directories for train dataset
src_gt = "./datasets/denoise/SIDD/train/groundtruth"
src_in = "./datasets/denoise/SIDD/train/input"
tar = "./datasets/denoise/SIDD/train/custom_dataset/train"

# # for val dataset
# tar = "./datasets/denoise/SIDD/train/custom_dataset/val"

# # for test
# tar = "./datasets/denoise/SIDD/train/custom_dataset/test"

# set hyper parameters
PS = 128
NUM_PATCHES = 50 #train:50, val:20, test:20
NUM_CORES = 10

# create ground_truth and clean patch directories
noisy_patchDir = os.path.join(tar, 'input')
clean_patchDir = os.path.join(tar, 'groundtruth')
if not os.path.exists(noisy_patchDir):
    os.makedirs(noisy_patchDir)
    
if not os.path.exists(clean_patchDir):
    os.makedirs(clean_patchDir)

# Save the patches as image files
files_gt = os.listdir(src_gt)
files_in = os.listdir(src_in)

# Use parallel processing to generate and save patches from images
Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(files_gt))))
"""

