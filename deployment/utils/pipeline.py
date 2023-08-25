# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 14:22:49 2023

@author: githin
"""

import os
import torch
import numpy as np
import math
import tqdm
import utils
import model_utils
import sys
import matplotlib.image as mpimg
import time

from datetime import datetime
from torchvision.utils import make_grid
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from model import Uformer
from PIL import Image
from skimage.metrics import structural_similarity as ssim



def expand2square(timg,factor=16.0,ps=1):
    """
    Expand an image tensor to a square size while maintaining its content in the center.

    Args:
        timg (Tensor): Input image tensor to be expanded.
        factor (float): Factor to determine the size of the expanded image (default is 16.0).
        ps (int): Patch size for resizing (default is 1).

    Returns:
        Tuple[Tensor, Tensor]: Expanded image tensor and a mask indicating the original content area.
    """
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h,w)/float(factor))*factor)
    X = math.ceil(X/ps)*ps
    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    return img, mask

def patch_image(input_tensor, ps):
    """
    Split an image into patches.

    Args:
        input_tensor (Tensor): Input image tensor to be split into patches.
        ps (int): Patch size.

    Returns:
        Tensor: Tensor containing the image patches.
    """
    _, channels, height, width = input_tensor.size()
    assert height == width, "Height and width must be equal"
    assert height % ps == 0, "Patch size must evenly divide height and width"
    num_patches = (height // ps) ** 2
    patches = torch.empty((num_patches, channels, ps, ps), dtype=input_tensor.dtype)
    for i in range(0, height, ps):
        for j in range(0, width, ps):
            patch = input_tensor[:, :, i:i+ps, j:j+ps]
            patch_idx = i // ps * (height // ps) + j // ps
            patches[patch_idx] = patch
    return patches

def repatch_image(patches, original_shape):
    """
    Reconstruct an image from patches.

    Args:
        patches (Tensor): Patches of the image to be reconstructed.
        original_shape (tuple): Original shape (batch size, channels, height, width) of the image.

    Returns:
        Tensor: Reconstructed image tensor.
    """
    _, channels, height, width = original_shape
    ps = patches.size(2)
    assert height == width, "Height and width must be equal"
    assert height % ps == 0, "Patch size must evenly divide height and width"
    assert (height // ps) ** 2 == patches.size(0), "Number of patches doesn't match"
    reconstructed_tensor = torch.empty(original_shape, dtype=patches.dtype)
    for i in range(0, height, ps):
        for j in range(0, width, ps):
            patch_idx = i // ps * (height // ps) + j // ps
            reconstructed_tensor[:, :, i:i+ps, j:j+ps] = patches[patch_idx]
    return reconstructed_tensor

def process(patches, restoration_model, cuda_flag):
    """
    Process a batch of image patches using a restoration model.

    Args:
        patches (Tensor): Batch of image patches to be processed.
        restoration_model (nn.Module): Restoration model to be used for processing.
        cuda_flag (bool): Flag indicating whether to use CUDA (GPU) for processing.

    Returns:
        Tensor: Batch of restored image patches.
    """
    num_patches = patches.size(0)
    flag = cuda_flag
    if flag == True:
        restored_patches = torch.zeros_like(patches).cuda()
    else:
        restored_patches = torch.zeros_like(patches).cpu()
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_patches), desc="Processing", file=sys.stdout):
            if flag == True:
                patch = patches[i:i + 1, :, :, :].cuda()
            else:
                patch = patches[i:i + 1, :, :, :].cpu()
            restored_patch = restoration_model(patch)
            restored_patches[i] = restored_patch[0]
    if flag == True:        
        return restored_patches.cuda()
    else:
        return restored_patches.cuda()

def plot_images(tensor, title):
    """
    Display a grid of images from a PyTorch tensor and provide a title.
    Args   :tensor (Tensor): PyTorch tensor containing images to be plotted.
            title (str): Title for the plot.
    Returns:None
   """
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=int(tensor.size(0) ** 0.5), padding=5)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()


def load_model(path, cuda_flag):
    """
   Load a pre-trained Uformer model from the specified path and configure for CPU or CUDA usage.

   Args:
       path (str): Path to the directory containing the model weights.
       cuda_flag (bool): Flag indicating whether to use CUDA (GPU) for model inference.

   Returns:
       model: Loaded Uformer model with weights loaded and mode set to evaluation.
   """
    flag = cuda_flag
    weight_path = path
    model = Uformer(img_size=256,embed_dim=32,win_size=8,token_projection='linear',token_mlp='leff',
               depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],modulator=True,dd_in=3)
    model_utils.load_checkpoint(model,weight_path)
    if cuda_flag == True:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()
    return model

def load_image(dir):
    """
   Load an image from the specified directory, handle data type conversion and channel adjustments.

   Args:
       dir (str): Path to the image file.

   Returns:
       numpy.ndarray: Loaded and processed image as a NumPy array.
   """
    in_dir = dir
    img = plt.imread(in_dir)
    if img.dtype != np.float32:                                 # cheking for dtype float32
        img = img.astype(np.float32) / 255.0
    num_channels = img.shape[2] if len(img.shape) == 3 else 0   # checking for RGB or RGBA
    if num_channels == 3:
        return img
    elif num_channels == 4:
        rgb_img = rgba_image[:, :, :3]
        return rgb_img
    else:
        print("Loaded image has an unexpected number of channels:", num_channels)



def save_image(image, output_directory, option):
    """
    Save an image to a specified output directory with a filename based on the option and current time.

    Args:
        image (tensor): Input image tensor.
        output_directory (str): Directory where the image will be saved.
        option (str): Option to be included in the filename.

    Returns:
        None
    """
    try:
        img = np.squeeze(image.cpu().detach().numpy())
        # Ensure the image array is in the correct shape for plotting (H x W x C)
        img = img.transpose(1, 2, 0)
        
        #print("saving image shape =", img.shape)
        
        current_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{option}_{current_time}.png"

        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Convert numpy array to PIL image
        pil_img = Image.fromarray(np.uint8(img * 255))

        # Save the image
        output_path = os.path.join(output_directory, filename)
        pil_img.save(output_path)
        
        print(f"Image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def calculate_psnr(org_img, pred_img):
    """
    Calculate the PSNR between an original image and a predicted image, both represented as PyTorch tensors.
    
    Args:
        org_img (tensor): A PyTorch tensor representing the original image with pixel values in the range [0, 255].
        pred_img (tensor): A PyTorch tensor representing the predicted image (reconstructed or compressed) with pixel values in the range [0, 255].
    
    Returns:
        None. The function prints the calculated PSNR value in decibels (dB) to the console.
    
    Note:
        The PSNR calculation assumes that both input images have pixel values in the range [0, 255]. An exception message is printed if an error occurs during the calculation.
    
    """

    try:
        original_image = org_img
        predictd_image = pred_img
        
        # setting all -ve values to 0
        original_image[original_image < 0] = 0
        predictd_image[predictd_image < 0] = 0

        #Normalizing both tensor using the max value
        original_image = original_image / torch.max(original_image ).item()
        predictd_image = predictd_image / torch.max(predictd_image).item()
        
        # converting both tensor to range [0,255]
        original_image=torch.mul(original_image,255)
        predictd_image=torch.mul(predictd_image,255)
        
        # Calculate the mean squared error (MSE)
        mse = torch.mean((original_image - predictd_image) ** 2)

        # Calculate the maximum pixel value (assuming the images have pixel values in the range [0, 255])
        max_pixel_value = 255.0

        # Calculate the PSNR
        psnr = 10 * torch.log10((max_pixel_value ** 2) / mse)

        print("PSNR: {:.3f} dB".format(psnr))
    except Exception as e:
        print(f"An error occurred: {e}")    

    


def calculate_ssim(original_image, predicted_image, win_size=None, channel_axis=-1):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        original_image (numpy.ndarray): The original image with shape (1, 3, H, W).
        predicted_image (numpy.ndarray): The predicted image with shape (1, 3, H, W).
        win_size (int, optional): Window size for SSIM calculation. If not provided, it will be automatically determined.
                                 Must be an odd value less than or equal to the smaller side of your images.
        channel_axis (int, optional): Axis number corresponding to the color channels. Default is -1 (last axis).

    Returns:
        float: The SSIM value between the two images.
    """
    original_image = np.squeeze(original_image)
    predicted_image = np.squeeze(predicted_image)
    
    if original_image.shape == predicted_image.shape:
        
        # Convert the images to the range [0, 255] (uint8) for SSIM calculation
        original_image = np.uint8(original_image * 255)
        predicted_image = np.uint8(predicted_image * 255)
    
        # Determine the window size if not provided
        if win_size is None:
            win_size = min(original_image.shape[0], original_image.shape[1])
            # Ensure the window size is odd
            win_size = win_size if win_size % 2 == 1 else win_size - 1
        
        # Calculate the SSIM
        ssim_value = ssim(original_image, predicted_image, win_size=win_size, channel_axis=channel_axis)
    
        print("SSIM: {:.3f}%".format(ssim_value))
    else:
        raise ValueError("Input images must have the same shape.")

