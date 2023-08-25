# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 21:38:36 2023

@author: githin

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os 
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../development')

import numpy as np
import math
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import scipy.io as sio
import utils
import options

from tqdm import tqdm
from einops import rearrange, repeat
from torch.utils.data import DataLoader
from ptflops import get_model_complexity_info
from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from dataset.dataset_denoise import *
from model import UNet,Uformer

if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='Image denoiseing')).parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
#------------------------------------------------------------------------------
    # Setting directories and parameters
    test_dir = opt.test_dir
    result_dir = opt.result_dir
    pretrain_weights = opt.pretrain_weights
    test_ps = opt.test_ps
    do_validation = opt.do_validation
    batch_size = opt.batch_size
    
    directory = os.path.join(dir_name, '..')    # for change the folder one backward
    result_dir_path = os.path.join(directory, result_dir) 
    
#------------------------------------------------------------------------------
    #creating result directory
    if os.path.exists(result_dir_path):
        print(f"\nResult directory already existed at {result_dir_path}")
    else:
        utils.mkdir(result_dir_path)
        print(f"\nResult directory created at {result_dir_path}")

#------------------------------------------------------------------------------
    # Loading data
    print('\nLoading datasets...')
    test_dataset = get_validation_data(opt.test_dir)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=opt.batch_size, 
                             shuffle=False, 
                             drop_last=False)
    len_testset = test_dataset.__len__()
    print(f"Size of test dataset: {len_testset}")
    
#------------------------------------------------------------------------------
    # Load architecture
    print(f"\nLoading trained model...")
    model_restoration= utils.get_arch(opt)
    utils.load_checkpoint(model_restoration,pretrain_weights)
    print(f"\nTesting  the trained model using weights: {pretrain_weights}\n")
    
    model_restoration.cuda()
    model_restoration.eval() 

#------------------------------------------------------------------------------
    # generate square images
    def expand2square(timg,factor=16.0,ps=1):
        _, _, h, w = timg.size()
        X = int(math.ceil(max(h,w)/float(factor))*factor)
        X = math.ceil(X/ps)*ps
        img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
        mask = torch.zeros(1,1,X,X).type_as(timg)
        img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
        mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
        return img, mask
    
#------------------------------------------------------------------------------
    # validation of trained model using test data
    if do_validation :
        print(f"Doing Validation on tarined model ...")
        with torch.no_grad():
            model_restoration.eval()
            psnr_model_init = []
            for ii, data_val in enumerate(tqdm(test_loader ), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with torch.cuda.amp.autocast():
                    restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)  
                psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
            psnr_model_init = sum(psnr_model_init)/len_testset
            print(f'\nPSNR: Model_trained & GT => {psnr_model_init:.4f}dB')
    else:
        print(f"\n Validation on trained model skipped !!!")
    
#------------------------------------------------------------------------------
    # testing
    print(f"\nTesting ...")
    with torch.no_grad():
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            rgb_gt = data_test[0].numpy().squeeze().transpose((1,2,0))
            rgb_noisy, mask = expand2square(data_test[1].cpu(), factor=128, ps=test_ps) 
            filenames = data_test[2]
    
            rgb_restored = model_restoration(rgb_noisy.cuda())
            rgb_restored = torch.masked_select(rgb_restored,mask.bool().cuda()).reshape(1,3,rgb_gt.shape[0],rgb_gt.shape[1])
            rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
    
            psnr = psnr_loss(rgb_restored, rgb_gt)
            ssim = ssim_loss(rgb_restored, rgb_gt, channel_axis=2)
            psnr_val_rgb.append(psnr)
            ssim_val_rgb.append(ssim)
            utils.save_img(os.path.join(result_dir,filenames[0]+'.PNG'), img_as_ubyte(rgb_restored))
            with open(os.path.join(result_dir,'psnr_ssim.txt'),'a') as f:
                f.write(filenames[0]+'.PNG ---->'+"PSNR: %.4f, SSIM: %.4f] "% (psnr, ssim)+'\n')
    psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)
    print("-------------------------------------------------------")
    print(f"PSNR: {psnr_val_rgb:.4f}   SSIM: {ssim_val_rgb:.4f}")
    print("-------------------------------------------------------")
    with open(os.path.join(result_dir,'psnr_ssim.txt'),'a') as f:
        f.write("Arch:Uformer_B, PSNR: %.4f, SSIM: %.4f] "% (psnr_val_rgb, ssim_val_rgb)+'\n')
