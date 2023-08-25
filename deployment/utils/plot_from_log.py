# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:36:55 2023

@author: githin
"""
 

import re
import matplotlib.pyplot as plt
    
def plot_from_log(dir):
    log_path = dir
    
    # Define regular expressions to match the relevant lines in the log
    psnr_pattern = r"Ep_best_psnr: ([\d.]+)"
    loss_pattern = r"Loss: ([\d.]+)"
    lr_pattern = r"LearningRate: ([\d.]+)"
    
    psnr_values = []
    loss_values = []
    lr_values = []
    
    with open(log_path, 'r') as file:
        for line in file:
            psnr_match = re.search(psnr_pattern, line)
            loss_match = re.search(loss_pattern, line)
            lr_match = re.search(lr_pattern, line)
          
            if psnr_match:
                psnr_values.append(float(psnr_match.group(1)))
            if loss_match:
                loss_values.append(float(loss_match.group(1)))
            if lr_match:
                lr_values.append(float(lr_match.group(1)))
    
    epochs = list(range(1, len(psnr_values) + 1))
    
    # Plot Train PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, psnr_values, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR Progression')
    plt.grid(True)
    plt.show()
    
    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, marker='o', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Progression')
    plt.grid(True)
    plt.show()
    
    # Plot Validation PSNR
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_values, marker='o', color='g')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    #......denoise
    #log_dir = r"E:\transfer_hub\Chandu_Projects\Project_Uformer\logs\denoise\SIDD\Uformer_B\FAIR\2023_06_12_20_35_28.log"
    #log_dir = r"E:\transfer_hub\Chandu_Projects\Project_Uformer\logs\denoise\SIDD\Uformer_B\FAIR\2023_06_16_09_47_07.log"
    #log_dir = r"E:\transfer_hub\Chandu_Projects\Project_Uformer\logs\denoise\SIDD\Uformer_B\FAIR\2023_06_19_21_31_21.log"
    log_dir = r"E:\transfer_hub\Chandu_Projects\Project_Uformer\logs\denoise\SIDD\Uformer_B\FAIR\2023_06_20_00_00_11.log"

    #......deblur
    #log_dir = r"E:\transfer_hub\Chandu_Projects\Project_Uformer\logs\deblur\GoPro\Uformer_B\FAIR\23_06_05_22_52_11.log"

    plot_from_log(log_dir)