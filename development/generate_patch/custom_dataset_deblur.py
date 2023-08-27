# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 18:29:54 2023

@author: githin

"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../development')

from glob import glob
from tqdm import tqdm
import numpy as np
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
import options

#patching both files and saving
def save_files(i):
    noisy_files, clear_files = [], []
    noisy_file, clean_file = files_in[i], files_gt[i]
    noisy_img = cv2.imread(src_in + noisy_file)
    clean_img = cv2.imread(src_gt + clean_file)

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

if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='Deblur image patch')).parse_args()
    
    # Setting directories and hyper parameters
    src_gt = opt.patch_src_gt
    src_in = opt.patch_src_in
    tar = opt.patch_tar
    PS = opt.ps
    NUM_PATCHES = opt.num_patches
    NUM_CORES = opt.num_cores
    
    print("\nsrc_gt:", src_gt)
    print("\nsrc_in:", src_in)
    print("\ntar:", tar)

    # create ground_truth and clean patch directories
    noisy_patchDir = os.path.join(tar, 'input')
    clean_patchDir = os.path.join(tar, 'groundtruth')
    
    if not os.path.exists(noisy_patchDir):
        os.makedirs(noisy_patchDir)
        print(f"\ninput directory created at {noisy_patchDir}")
        
    if not os.path.exists(clean_patchDir):
        os.makedirs(clean_patchDir)
        print(f"\nground_truth directory created at {clean_patchDir}")
    
    files_gt = os.listdir(src_gt)
    files_in = os.listdir(src_in)
    
    print(f"\nnumber of images in the source file : {len(files_in)}\n")
    Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(files_gt)))) 
    print("\nDone!!!\n")