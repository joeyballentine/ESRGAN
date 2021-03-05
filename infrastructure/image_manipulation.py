import cv2
import os
import numpy as np
import torch

def create_lr_image(img, is_fp16):
    if img.shape[2] == 3: img = img[:, :, [2, 1, 0]]
    elif img.shape[2] == 4: img = img[:, :, [2, 1, 0, 3]]

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    if is_fp16: img = img.half()
    return img.unsqueeze(0)

def reshape_output(output):
    if output.shape[0] == 3:
        output = output[[2, 1, 0], :, :]
    elif output.shape[0] == 4:
        output = output[[2, 1, 0, 3], :, :]
    return np.transpose(output, (1, 2, 0))

def process_image(img, device, model, is_fp16):
    '''
    Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

            Parameters:
                    img (array): The image to process

            Returns:
                    rlt (array): The processed image
    '''
    img_LR = create_lr_image(img, is_fp16)
    img_LR = img_LR.to(device)
    output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return reshape_output(output)

def make_alpha_black_and_white(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    img2 = np.copy(img[:, :, :3])
    for c in range(3):
        img1[:, :, c] *= img[:, :, 3]
        img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

    output1 = process_image(img1, device, model, is_fp16)
    output2 = process_image(img2, device, model, is_fp16)
    alpha = 1 - np.mean(output2-output1, axis=2)
    output = np.dstack((output1, alpha))
    return np.clip(output, 0, 1)

def upscale_alpha(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
    output1 = process_image(img1, device, model, is_fp16)
    output2 = process_image(img2, device, model, is_fp16)
    return cv2.merge((output1[:, :, 0], output1[:, :, 1], output1[:, :, 2], output2[:, :, 0]))

def make_regular_alpha(img, device, model, is_fp16):
    img1 = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
    img2 = cv2.merge((img[:, :, 1], img[:, :, 2], img[:, :, 3]))
    output1 = process_image(img1, device, model, args.fp16)
    output2 = process_image(img2, device, model, args.fp16)
    return cv2.merge((output1[:, :, 0], output1[:, :, 1], output1[:, :, 2], output2[:, :, 2]))

def remove_alpha(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    output = process_image(img1, device, model, args.fp16)
    return cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)


def crop_seamless(img, scale):
    img_height, img_width = img.shape[:2]
    y, x = 16 * scale, 16 * scale
    h, w = img_height - (32 * scale), img_width - (32 * scale)
    img = img[y:y+h, x:x+w]
    return img

def make_alpha_binary(alpha, threshold):
    _, alpha = cv2.threshold(alpha, threshold, 1, cv2.THRESH_BINARY)
    return alpha

def make_alpha_ternery(alpha, threshold, boundry_offset):
    half_transparent_lower_bound = threshold - boundry_offset
    half_transparent_upper_bound = threshold + boundary_offset
    return np.where(alpha < half_transparent_lower_bound, 0, np.where(
        alpha <= half_transparent_upper_bound, .5, 1))

def upscale_image(img, device, model, args, in_channels, out_channels):
    '''
    Upscales the image passed in with the specified model

            Parameters:
                    img: The image to upscale
                    model_path (string): The model to use

            Returns:
                    output: The processed image
    '''

    img = img * 1. / np.iinfo(img.dtype).max

    if img.ndim == 3 and img.shape[2] == 4 and in_channels == 3 and out_channels == 3:

        # Fill alpha with white and with black, remove the difference
        if args.alpha_mode == 1: output = make_alpha_black_and_white(img, device, model, args.fp16)
        # Upscale the alpha channel itself as its own image
        elif args.alpha_mode == 2: output = upscale_alpha(img, device, model, args.fp16)
        # Use the alpha channel like a regular channel
        elif args.alpha_mode == 3: output = make_regular_alpha(img, device, model, args.fp16)            
        # Remove alpha
        else: output = remove_alpha(img, device, model, args.fp16)            

        alpha = output[:, :, 3]
        threshold = args.alpha_threshold
        if args.binary_alpha: output[:, :, 3] = make_alpha_binary(alpha, threshold)
        elif args.ternary_alpha: output[:, :, 3] = make_alpha_ternery(alpha, threshold, args.alpha_boundary_offset)
           
    else:
        if img.ndim == 2:
            img = np.tile(np.expand_dims(img, axis=2), (1, 1, min(in_channels, 3)))
        if img.shape[2] > in_channels:  # remove extra channels
            print('Warning: Truncating image channels')
            img = img[:, :, :in_channels]
        # pad with solid alpha channel
        elif img.shape[2] == 3 and in_channels == 4:
            img = np.dstack((img, np.full(img.shape[:-1], 1.)))
        output = process_image(img, device, model, args.fp16)

    return (output * 255.).round()

def generate_upscale_function(device, model, args, in_channels, out_channels):
    def upscaleFunction(img):
        return upscale_image(img, device, model, args, in_channels, out_channels)
    return upscaleFunction


def get_all_image_paths(imgPath, reverse):
    images = []
    for root, _, files in os.walk(imgPath):
        for file in sorted(files, reverse=reverse):
            if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga']:
                images.append(os.path.join(root, file))
    return images

def make_image_seamless(img, seam_type): 
    if seam_type == 'tile': return cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    elif seam_type == 'mirror': return cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    elif seam_type == 'replicate':return cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)
    elif seam_type == 'alpha_pad': return cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    return img