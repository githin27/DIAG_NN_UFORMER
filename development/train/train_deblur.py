# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:55:05 2023

@author: githin

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available. Using CPU.")

import os 
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append('../development')

import torch
import torch.nn as nn
import utils
import torch.optim as optim
import datetime
import time
import matplotlib.pyplot as plt
import logging
import argparse
import options

from model import Uformer
from torch.optim.lr_scheduler import StepLR
from losses import CharbonnierLoss
from torch.utils.data import DataLoader
from utils.loader import get_training_data, get_validation_data
from dataset import get_validation_deblur_data
from tqdm import tqdm
from timm.utils import NativeScaler
from warmup_scheduler import GradualWarmupScheduler

#--------------------------------------------------------------------------------
if __name__ == '__main__':
    opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
    #print(opt)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    
    # Set Directories
    pretrain_weights_path = opt.pretrain_weights
    train_dir = opt.train_dir
    val_dir = opt.val_dir
    model_dir = opt.model_dir
    log_dir  = opt.log_dir
    directory = os.path.join(dir_name, '..')    # for change the folder one backward
    log_path = os.path.join(directory, log_dir)


#------------------------------------------------------------------------------    
    # Argumet parameters from parser
    train_ps = opt.train_ps
    dd_in = opt.dd_in
    optimizer = opt.optimizer
    lr_initial = opt.lr_initial
    weight_decay = opt.weight_decay
    warmup_epochs = opt.warmup_epochs
    pretrain_weights = opt.pretrain_weights
    train_workers = opt.train_workers
    eval_workers = opt.eval_workers
    val_ps = opt.val_ps
    checkpoint = opt.checkpoint
    batch_size = opt.batch_size
    nepoch = opt.nepoch
    resume = opt.resume
    do_validation=opt.do_validation
    warmup = opt.warmup
    log =opt.log

#------------------------------------------------------------------------------
    print("\nLoading model...")
    # Loading Architecture
    model_restoration = Uformer(img_size=train_ps,
                                embed_dim=32,
                                win_size=8,
                                token_projection='linear',
                                token_mlp='leff',
                                depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                                modulator=True,
                                dd_in=dd_in) 
    
#------------------------------------------------------------------------------
    #creating log directory
    if os.path.exists(log_path):
        print(f"\nLog directory already existed at {log_path}")
    else:
        utils.mkdir(log_path)
        print(f"\nLog directory created at {log_path}")
    log_path = os.path.join(log_path,datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")+".log")

    #logging options and model architecture data   
    if log:
        try:
            with open(log_path, 'a') as log_file:            
                log_file.write("options: {}\n".format(opt))
                log_file.write("Model_Architecture: {}\n".format(model_restoration))
        except Exception as e:
            print("An exception occurred:", e)
            
#------------------------------------------------------------------------------
    # Set Optimizer
    start_epoch = 0
    if optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), 
                               lr=lr_initial, 
                               betas=(0.9, 0.999),
                               eps=1e-8, 
                               weight_decay=weight_decay)
    elif optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), 
                                lr=lr_initial, betas=(0.9, 0.999),
                                eps=1e-8, 
                                weight_decay=weight_decay)
    else:
        raise Exception("Error optimizer...")
    print(f"\nOptimizer: {opt.optimizer}")
    
    # Set Data parallel
    model_restoration = torch.nn.DataParallel (model_restoration) 
    model_restoration.cuda();
    
#------------------------------------------------------------------------------
    # Set Scheuler
    if warmup:
        print(f"\nScheduler: Using warmup and cosine strategy!")
        warmup_epochs = warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print(f"\nScheduler: Using StepLR with step = {step}")
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()
        
#------------------------------------------------------------------------------
    # Set Resume code
    if resume: 
        path_chk_rest = pretrain_weights 
        utils.load_checkpoint(model_restoration,path_chk_rest) 
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
        lr = utils.load_optim(optimizer, path_chk_rest) 
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print(f"\nTrain resuming from {path_chk_rest} with learning rate {new_lr}")
        nepoch=start_epoch+nepoch 
    else:
        print(f"\nTraining from scratch with learning rate {opt.lr_initial}")

#------------------------------------------------------------------------------ 
    # set loss
    criterion = CharbonnierLoss().cuda()
    print(f"\nLoss: {criterion}")

#------------------------------------------------------------------------------    
    # Loading Data
    print('\nLoading datasets...')
    img_options_train = {'patch_size':train_ps}
    train_dataset = get_training_data(train_dir, img_options_train)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True,
                              num_workers=train_workers, 
                              pin_memory=False, 
                              drop_last=False)
    img_options_val = {'patch_size':val_ps}
    val_dataset = get_validation_deblur_data(val_dir, img_options_val)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=eval_workers, 
                            pin_memory=False, 
                            drop_last=False)
    len_trainset = train_dataset.__len__()
    len_valset = val_dataset.__len__()
    print(f"Size of training dataset: {len_trainset}\nSizeof validation set: {len_valset}")
 
#------------------------------------------------------------------------------
    # Model validation
    if do_validation :
        print(f"\nDoing Validation ...")
        with torch.no_grad():
            model_restoration.cuda()
            model_restoration.eval()
            psnr_dataset = []
            psnr_model_init = []
            for ii, data_val in enumerate(tqdm(val_loader ), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                with torch.cuda.amp.autocast():
                    restored = model_restoration(input_)
                    restored = torch.clamp(restored,0,1)  
                psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
                psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
            psnr_dataset = sum(psnr_dataset)/len_valset
            psnr_model_init = sum(psnr_model_init)/len_valset
            print(f"PSNR: Input & GT => {psnr_dataset:.4f} dB\nPSNR: Model_init & GT => {psnr_model_init:.4f} dB")
    else:
        print("\nValidation Skipped !!!")
        
#------------------------------------------------------------------------------    
    # Training
    print(f'\nTraining Start Epoch:{start_epoch}, End Epoch:{nepoch}')
    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    eval_now = len(train_loader)//4  
    print(f"\nEvaluation after every {eval_now} iterations")
    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()
    start_time = time.time()   
    psnr_train_rgb_epoch=[]
    psnr_val_best_rgb_epoch=[]
    for epoch in range(start_epoch, nepoch):
        epoch_loss = 0
        train_id = 1
        psnr_train_rgb = []
        for i, data in enumerate(tqdm(train_loader), 0): 
            optimizer.zero_grad()
            target = data[0].cuda()
            input_ = data[1].cuda()
            with torch.cuda.amp.autocast():
                restored = model_restoration(input_)
                loss = criterion(restored, target)  
            restored = torch.clamp(restored,0,1) 
            psnr_train_rgb.append(utils.batch_PSNR(restored, target, False).item())     
            loss_scaler(loss, optimizer,parameters=model_restoration.parameters())
            epoch_loss +=loss.item()
            # Evaluation #
            if (i+1)%eval_now==0 and i>0:
                with torch.no_grad():
                    model_restoration.eval()
                    psnr_val_rgb = []
                    for ii, data_val in enumerate((val_loader), 0):
                        target = data_val[0].cuda()
                        input_ = data_val[1].cuda()
                        filenames = data_val[2]
                        with torch.cuda.amp.autocast():
                            restored = model_restoration(input_)
                        restored = torch.clamp(restored,0,1)  
                        psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())     
                    psnr_val_rgb = sum(psnr_val_rgb)/len_valset
    
                    # calculate best PSNR
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i 
                        torch.save({'epoch': epoch, 
                                    'state_dict': model_restoration.state_dict(),
                                    'optimizer' : optimizer.state_dict()
                                    }, os.path.join(model_dir,"model_best.pth"))                    
        psnr_train_rgb=sum(psnr_train_rgb)/len_trainset
        psnr_train_rgb_epoch.append(psnr_train_rgb)
        psnr_val_best_rgb_epoch.append(best_psnr)
    
        print(f"Epoch:{epoch}, Loss:{epoch_loss:.4f}, PSNR_train:{psnr_train_rgb:.4f}dB, PSNR_val:{best_psnr:.4f}dB")
        if log:
            try:
                with open(log_path, 'a') as log_file:            
                    log_file.write(f"\nEpoch:{epoch}  Loss:{epoch_loss:.4f}  PSNR_train:{psnr_train_rgb:.4f}dB  PSNR_val:{best_psnr:.4f}dB")
            except Exception as e:
                print("An exception occurred:", e)
                
        scheduler.step()
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))   
        if epoch%checkpoint == 0:
              torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
        torch.cuda.empty_cache()
        
    end_time = time.time()
    
    # time calculation
    formatted_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    formatted_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    
    t_ime = end_time-start_time
    total_seconds = int(t_ime)
    ss = total_seconds % 60
    total_minutes = total_seconds // 60
    mm = total_minutes % 60
    total_hours = total_minutes // 60
    hh = total_hours % 24
    dd = total_hours // 24
    
    print("------------------------------------------------------------------")
    print("Training Completed...")
    print(f"PSNR TRAIN RGB : {(sum(psnr_train_rgb_epoch)/len(psnr_train_rgb_epoch)):.4f}dB")
    print(f"PSNR VAL RGB : {(sum(psnr_val_best_rgb_epoch)/len(psnr_val_best_rgb_epoch)):.4f}dB")
    print("------------------------------------------------------------------")
    print(f"Train Start: {formatted_start_time}\nTrain End: {formatted_end_time}\nTraining Time: {dd} days, {hh} Hs, {mm} Ms, {ss} S ")
    
    if log:
        try:
            with open(log_path, 'a') as log_file:            
                log_file.write(f"\nPSNR TRAIN RGB : {(sum(psnr_train_rgb_epoch)/len(psnr_train_rgb_epoch)):.4f}dB")
                log_file.write(f"\nPSNR VAL RGB : {(sum(psnr_val_best_rgb_epoch)/len(psnr_val_best_rgb_epoch)):.4f}dB")
                log_file.write(f"\nTrain Start: {formatted_start_time}\nTrain End: {formatted_end_time}\nTraining Time: {dd} days, {hh} Hs, {mm} Ms, {ss} S ")
        except Exception as e:
            print("An exception occurred:", e)
    
