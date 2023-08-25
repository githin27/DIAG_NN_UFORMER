# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:29:54 2023

@author: githin

Note:
    * Download the dataset from https://mailustceducn-my.sharepoint.com/personal/zhendongwang_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fzhendongwang%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FUformer%2Fdatasets%2FGoPro
    * Both train and test dataset contains ground_truth and input
    * Save the dataset in ./datasets/deblur/GoPro
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
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        
        noisy_dir=noisy_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'
        clear_dir=clean_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'

        cv2.imwrite(noisy_dir, noisy_patch)
        cv2.imwrite(clear_dir, clean_patch)

# for train dataset
src_gt = "./datasets/deblur/GoPro/train/groundtruth"
src_in = "./datasets/deblur/GoPro/train/input"
tar = "./datasets/deblur/GoPro/train/custom_dataset/train"

# # for val dataset
# src_gt = "./datasets/deblur/GoPro/train/groundtruth"
# src_in = "./datasets/deblur/GoPro/train/input"
# tar = "./datasets/deblur/GoPro/train/custom_dataset/val"

# # for test
# src_gt = "./datasets/deblur/GoPro/test/groundtruth"
# src_in = "./datasets/deblur/GoPro/test/input"
# tar = "./datasets/deblur/GoPro/train/custom_dataset/test"

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

files_gt = os.listdir(src_gt)
files_in = os.listdir(src_in)

Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(files_gt))))