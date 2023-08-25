# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:01:42 2023

@author: githin
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'./utils'))

import argparse
import options
import torch
from pipeline import *

if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='U_former_image_restoration')).parse_args()
    #print(opt)
    
    if torch.cuda.is_available():
        cuda_flag = True
    else:
        cuda_flag = False
     
    print("----------------------------------------------------")    
    print(f"You Choose: {opt.operation}")
    print("----------------------------------------------------")
    
    
    in_img = load_image(opt.input_dir)
    
    model = load_model(opt.model_weight, cuda_flag)
    
    h,w,c = in_img.shape                                                   
    in_img = in_img.reshape(1,h,w,c)
    in_img_torch = torch.from_numpy(in_img).permute(0,3,1,2)
    
    i_img, i_mask = expand2square(in_img_torch,opt.ps)                         
    patches = patch_image(i_img, opt.ps) 
    pred_patches = process(patches, model, cuda_flag)
    pred_repaches = repatch_image(pred_patches, i_img.shape)
    pred = torch.masked_select(pred_repaches, i_mask.bool()).reshape(1,3,h,w)
    
    plot_images(in_img_torch, 'input image')
    plot_images(pred, 'result image')
    
    print("----------------------------------------------------")
    calculate_psnr(in_img_torch, pred)
    calculate_ssim(in_img_torch, pred)
    print("----------------------------------------------------")
    
    
    if opt.save_result == True:
        save_image(pred, opt.output_dir, opt.operation)
