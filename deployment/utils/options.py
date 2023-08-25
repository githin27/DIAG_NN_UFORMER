# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 13:38:04 2023

@author: githin
"""
"""
Note:
    
model weight should be load to ./model/denoise or deblur/model_weight
input image should be load to ./image_in/ denoise_img or deblur_img/image
create output directory at ./image_out/denoise or deblur
"""

import os
import torch
    
class Options():
    def __init__(self):
        pass
    
    def init(self, parser):
        # parser.add_argument('--operation', type=str, default='denoise', help='denoise or deblur')
        # parser.add_argument('--model_weight', type=str, default='./model/denoise/Uformer_B.pth', help='pretrained model weight')
        # parser.add_argument('--input_dir', type=str, default='./image_in/denoise_img/30_10.png', help='input image directory')
        # parser.add_argument('--output_dir', type=str, default='./image_out/denoise/', help='directory for output image')
        
        parser.add_argument('--operation', type=str, default='deblur', help='denoise or deblur')
        parser.add_argument('--model_weight', type=str, default='./model/deblur/Uformer_B.pth', help='pretrained model weight')
        parser.add_argument('--input_dir', type=str, default='./image_in/deblur_img/GOPR0384_11_00-000018.png', help='input image directory')
        parser.add_argument('--output_dir', type=str, default='./image_out/deblur/', help='directory for output image')
        
        
        parser.add_argument('--ps', type=int, default=128, help='patch size')
        parser.add_argument('--save_result', type=bool, default=True, help='save the output')
        return parser
    
    