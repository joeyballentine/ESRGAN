import numpy as np
import torch
import cv2
from torchvision.utils import make_grid
import math

def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    # flip image channels
    out: torch.Tensor = image.flip(-3) #https://github.com/pytorch/pytorch/issues/229
    #out: torch.Tensor = image[[2, 1, 0], :, :] #RGB to BGR #may be faster
    return out

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgr_to_rgb(), flip image channels
    return bgr_to_rgb(image)

def bgra_to_rgba(image: torch.Tensor) -> torch.Tensor:
    out: torch.Tensor = image[[2, 1, 0, 3], :, :]
    return out

def rgba_to_bgra(image: torch.Tensor) -> torch.Tensor:
    #same operation as bgra_to_rgba(), flip image channels
    return bgra_to_rgba(image)

#TODO: Could also automatically detect the possible range with min and max, like in def ssim()
def denorm(x, min_max=(-1.0, 1.0)):
    '''
        Denormalize from [-1,1] range to [0,1]
        formula: xi' = (xi - mu)/sigma
        Example: "out = (x + 1.0) / 2.0" for denorm 
            range (-1,1) to (0,1)
        for use with proper act in Generator output (ie. tanh)
    '''
    out = (x - min_max[0]) / (min_max[1] - min_max[0])
    if isinstance(x, torch.Tensor):
        return out.clamp(0, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, 0, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")

def norm(x): 
    #Normalize (z-norm) from [0,1] range to [-1,1]
    out = (x - 0.5) * 2.0
    if isinstance(x, torch.Tensor):
        return out.clamp(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.clip(out, -1, 1)
    else:
        raise TypeError("Got unexpected object type, expected torch.Tensor or \
        np.ndarray")

#2tensor
def np2tensor(img, bgr2rgb=True, data_range=1., normalize=False, change_range=True, add_batch=True):
    """
    Converts a numpy image array into a Tensor array.
    Parameters:
        img (numpy array): the input image numpy array
        add_batch (bool): choose if new tensor needs batch dimension added 
    """
    if not isinstance(img, np.ndarray): #images expected to be uint8 -> 255
        raise TypeError("Got unexpected object type, expected np.ndarray")
    #check how many channels the image has, then condition, like in my BasicSR. ie. RGB, RGBA, Gray
    #if bgr2rgb:
        #img = img[:, :, [2, 1, 0]] #BGR to RGB -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    if change_range:
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo
        elif np.issubdtype(img.dtype, np.floating):
            info = np.finfo
        img = img*data_range/info(img.dtype).max #uint8 = /255
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float() #"HWC to CHW" and "numpy to tensor"
    if bgr2rgb:
        if img.shape[0] == 3: #RGB
            #BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img = bgr_to_rgb(img)
        elif img.shape[0] == 4: #RGBA
            #BGR to RGB -> in tensor, if using OpenCV, else not needed. Only if image has colors.)
            img = bgra_to_rgba(img)
    if add_batch:
        img.unsqueeze_(0) # Add fake batch dimension = 1 . squeeze() will remove the dimensions of size 1
    if normalize:
        img = norm(img)
    return img

#2np
def tensor2np(img, rgb2bgr=True, remove_batch=True, data_range=255, 
              denormalize=False, change_range=True, imtype=np.uint8):
    """
    Converts a Tensor array into a numpy image array.
    Parameters:
        img (tensor): the input image tensor array
            4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        remove_batch (bool): choose if tensor of shape BCHW needs to be squeezed 
        denormalize (bool): Used to denormalize from [-1,1] range back to [0,1]
        imtype (type): the desired type of the converted numpy array (np.uint8 
            default)
    Output: 
        img (np array): 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Got unexpected object type, expected torch.Tensor")
    n_dim = img.dim()

    #TODO: Check: could denormalize here in tensor form instead, but end result is the same
    
    img = img.float().cpu()  
    
    if n_dim == 4 or n_dim == 3:
        #if n_dim == 4, has to convert to 3 dimensions, either removing batch or by creating a grid
        if n_dim == 4 and remove_batch:
            if img.shape[0] > 1:
                # leave only the first image in the batch
                img = img[0,...] 
            else:
                # remove a fake batch dimension
                img = img.squeeze()
                # squeeze removes batch and channel of grayscale images (dimensions = 1)
                if len(img.shape) < 3: 
                    #add back the lost channel dimension
                    img = img.unsqueeze(dim=0)
        # convert images in batch (BCHW) to a grid of all images (C B*H B*W)
        else:
            n_img = len(img)
            img = make_grid(img, nrow=int(math.sqrt(n_img)), normalize=False)
        
        if img.shape[0] == 3 and rgb2bgr: #RGB
            #RGB to BGR -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgb_to_bgr(img).numpy()
        elif img.shape[0] == 4 and rgb2bgr: #RGBA
            #RGBA to BGRA -> in tensor, if using OpenCV, else not needed. Only if image has colors.
            img_np = rgba_to_bgra(img).numpy()
        else:
            img_np = img.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # "CHW to HWC" -> # HWC, BGR
    elif n_dim == 2:
        img_np = img.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))

    #if rgb2bgr:
        #img_np = img_np[[2, 1, 0], :, :] #RGB to BGR -> in numpy, if using OpenCV, else not needed. Only if image has colors.
    #TODO: Check: could denormalize in the begining in tensor form instead
    if denormalize:
        img_np = denorm(img_np) #denormalize if needed
    if change_range:
        img_np = np.clip(data_range*img_np,0,data_range).round() #clip to the data_range
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    #has to be in range (0,255) before changing to np.uint8, else np.float32
    return img_np.astype(imtype)




####################
# Prepare Images
####################
# https://github.com/sunreef/BlindSR/blob/master/src/image_utils.py
def patchify_tensor(features, patch_size, overlap=10):
    batch_size, channels, height, width = features.size()

    effective_patch_size = patch_size - overlap
    n_patches_height = (height // effective_patch_size)
    n_patches_width = (width // effective_patch_size)

    if n_patches_height * effective_patch_size < height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < width:
        n_patches_width += 1

    patches = []
    for b in range(batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, height - patch_size)
                patch_start_width = min(w * effective_patch_size, width - patch_size)
                patches.append(features[b:b+1, :,
                               patch_start_height: patch_start_height + patch_size,
                               patch_start_width: patch_start_width + patch_size])
    return torch.cat(patches, 0)

def recompose_tensor(patches, full_height, full_width, overlap=10):

    batch_size, channels, patch_size, _ = patches.size()
    effective_patch_size = patch_size - overlap
    n_patches_height = (full_height // effective_patch_size)
    n_patches_width = (full_width // effective_patch_size)

    if n_patches_height * effective_patch_size < full_height:
        n_patches_height += 1
    if n_patches_width * effective_patch_size < full_width:
        n_patches_width += 1

    n_patches = n_patches_height * n_patches_width
    if batch_size % n_patches != 0:
        print("Error: The number of patches provided to the recompose function does not match the number of patches in each image.")
    final_batch_size = batch_size // n_patches

    blending_in = torch.linspace(0.1, 1.0, overlap)
    blending_out = torch.linspace(1.0, 0.1, overlap)
    middle_part = torch.ones(patch_size - 2 * overlap)
    blending_profile = torch.cat([blending_in, middle_part, blending_out], 0)

    horizontal_blending = blending_profile[None].repeat(patch_size, 1)
    vertical_blending = blending_profile[:, None].repeat(1, patch_size)
    blending_patch = horizontal_blending * vertical_blending

    blending_image = torch.zeros(1, channels, full_height, full_width)
    for h in range(n_patches_height):
        for w in range(n_patches_width):
            patch_start_height = min(h * effective_patch_size, full_height - patch_size)
            patch_start_width = min(w * effective_patch_size, full_width - patch_size)
            blending_image[0, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += blending_patch[None]

    recomposed_tensor = torch.zeros(final_batch_size, channels, full_height, full_width)
    if patches.is_cuda:
        blending_patch = blending_patch.cuda()
        blending_image = blending_image.cuda()
        recomposed_tensor = recomposed_tensor.cuda()
    patch_index = 0
    for b in range(final_batch_size):
        for h in range(n_patches_height):
            for w in range(n_patches_width):
                patch_start_height = min(h * effective_patch_size, full_height - patch_size)
                patch_start_width = min(w * effective_patch_size, full_width - patch_size)
                recomposed_tensor[b, :, patch_start_height: patch_start_height + patch_size, patch_start_width: patch_start_width + patch_size] += patches[patch_index] * blending_patch
                patch_index += 1
    recomposed_tensor /= blending_image

    return recomposed_tensor

