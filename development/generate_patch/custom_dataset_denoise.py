# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:13:59 2023

@author: githinNote:
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

# patching both files and saving
def save_files(i):
    noisy_file, clean_file = noisy_files[i], clean_files[i]
    noisy_img = cv2.imread(noisy_file)
    print(np.array(noisy_img).shape)
    clean_img = cv2.imread(clean_file)

    H = noisy_img.shape[0]
    W = noisy_img.shape[1]
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]
        
        noisy_dir=noisy_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'
        clear_dir=clean_patchDir+'/'+str(i+1)+'_'+str(j+1)+'.png'
        
        print("Clean cdir : ", clear_dir)
        print("Noisy dir : ",noisy_dir)
        cv2.imwrite(noisy_dir, noisy_patch)
        cv2.imwrite(clear_dir, clean_patch)
 
if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='Deblur image patch')).parse_args()

    src = opt.patch_src
    tar = opt.patch_tar
    PS = opt.ps   
    NUM_PATCHES = opt.num_patches 
    NUM_CORES = opt.num_cores
    
    print(f"\nSource Directory: {src}")
    print(f"\nTarget Directory: {tar}")
    
    noisy_patchDir = os.path.join(tar, 'input')
    clean_patchDir = os.path.join(tar, 'groundtruth')
    
    print(f"\nTarget input directory: {noisy_patchDir}")
    print(f"\nTarget ground_truth directory: {clean_patchDir}")
    
    files = natsorted(glob(os.path.join(src, '*', '*.PNG')))
    print(f"\nTotal number of raw images: {len(files)}")
    
    if not os.path.exists(noisy_patchDir):
        os.makedirs(noisy_patchDir)
        print(f"\ninput directory created at {noisy_patchDir}")
            
    if not os.path.exists(clean_patchDir):
        os.makedirs(clean_patchDir)
        print(f"\nground_truth directory created at {clean_patchDir}\n")
        
    noisy_files, clean_files = [], []
    
    for file_ in files:
        filename = os.path.split(file_)[-1]
        if 'GT' in filename:
            clean_files.append(file_)
        if 'NOISY' in filename:
            noisy_files.append(file_)
            
        
    Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(noisy_files))))

    print("\nDone!!!\n")










