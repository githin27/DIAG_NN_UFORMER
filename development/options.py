# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 21:53:32 2023

@author: githin
"""

import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        #parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size for test_deblur, test_denoise')
        parser.add_argument('--nepoch', type=int, default=50, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=2, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=2, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='SIDD')
        #parser.add_argument('--pretrain_weights',type=str, default='../models/pretrained/denoise/model_best.pth', help='path of pretrained_weights')
        #parser.add_argument('--pretrain_weights',type=str, default='../models/pretrained/deblur/model_best.pth', help='path of test_deblur pretrained_weights')
        parser.add_argument('--pretrain_weights',type=str, default='../models/pretrained/denoise/model_best.pth', help='path of test_noise pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='weight decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')
        parser.add_argument('--arch', type=str, default ='Uformer_B',  help='archtechture')
        parser.add_argument('--mode', type=str, default ='denoising',  help='image restoration mode')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
        #parser.add_argument('--result_dir', type=str, default ='../result_dir/test/deblur',  help='Reult directory for test_deblur')
        parser.add_argument('--result_dir', type=str, default ='../result_dir/test/denoise',  help='Reult directory for test_denoise')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        parser.add_argument('--model_dir',type=str, default ='./models/training/denoise', help='dir for train model') 

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
        parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
        parser.add_argument('--modulator', action='store_true', default=False, help='multi-scale modulator')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
        
        # args for custom patching
        parser.add_argument('--patch_src_gt', type=str, default ='./GoPro/train/groundtruth',  help='ground truth source directory for deblur')
        parser.add_argument('--patch_src_in', type=str, default ='./GoPro/train/input',  help='input source directory for deblur')
        parser.add_argument('--patch_src', type=str, default ='./GoPro/train/input',  help='input source directory for deblur')
        parser.add_argument('--patch_tar', type=str, default ='./datasets/deblur/GoPro/customized_dataset/train',  help='target source directory')
        parser.add_argument('--num_patches', type=int, default =50,  help='number of patches')
        parser.add_argument('--num_cores', type=int, default =10,  help='number of cores')
        
        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
        parser.add_argument('--test_ps', type=int, default=128, help='patch size of test sample')
        parser.add_argument('--ps', type=int, default=128, help='patch size')
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='./datasets/denoise/SIDD/customized_dataset/train',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='./datasets/denoise/SIDD/customized_dataset/val',  help='dir of validation data')
        #parser.add_argument('--test_dir', type=str, default ='../datasets/denoise/SIDD/customized_dataset/test',  help='dir of test data')
        #parser.add_argument('--test_dir', type=str, default ='../datasets/deblur/GoPro/customized_dataset/test',  help='dir of test_deblur data')
        parser.add_argument('--test_dir', type=str, default ='../datasets/denoise/SIDD/customized_dataset/test',  help='dir of test_denoise data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 
        parser.add_argument('--log_dir', type=str, default ='./log/denoise/',  help='dir of log data')
        parser.add_argument('--do_validation', action='store_true', default=True, help='for doing validation')
        parser.add_argument('--log', action='store_true', default=True, help='log the data')


        # ddp
        parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')
        parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
        parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
        return parser
