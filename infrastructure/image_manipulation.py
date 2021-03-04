import cv2
import numpy as np
import torch

# This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao
def createLrImage(img, is_fp16):
    if img.shape[2] == 3: img = img[:, :, [2, 1, 0]]
    elif img.shape[2] == 4: img = img[:, :, [2, 1, 0, 3]]

    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

    if is_fp16: img = img.half()
    return img.unsqueeze(0)

def reshapeOutput(output):
    if output.shape[0] == 3:
        output = output[[2, 1, 0], :, :]
    elif output.shape[0] == 4:
        output = output[[2, 1, 0, 3], :, :]
    return np.transpose(output, (1, 2, 0))

def process(img, device, model, is_fp16):
    '''
    Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

            Parameters:
                    img (array): The image to process

            Returns:
                    rlt (array): The processed image
    '''
    img_LR = createLrImage(img, is_fp16)
    img_LR = img_LR.to(device)
    output = model(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    return reshapeOutput(output)

def makeAlphaBlackAndWhite(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    img2 = np.copy(img[:, :, :3])
    for c in range(3):
        img1[:, :, c] *= img[:, :, 3]
        img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

    output1 = process(img1, device, model, is_fp16)
    output2 = process(img2, device, model, is_fp16)
    alpha = 1 - np.mean(output2-output1, axis=2)
    output = np.dstack((output1, alpha))
    return np.clip(output, 0, 1)

def upscaleAlpha(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    img2 = cv2.merge((img[:, :, 3], img[:, :, 3], img[:, :, 3]))
    output1 = process(img1, device, model, is_fp16)
    output2 = process(img2, device, model, is_fp16)
    return cv2.merge((output1[:, :, 0], output1[:, :, 1], output1[:, :, 2], output2[:, :, 0]))

def makeRegularAlpha(img, device, model, is_fp16):
    img1 = cv2.merge((img[:, :, 0], img[:, :, 1], img[:, :, 2]))
    img2 = cv2.merge((img[:, :, 1], img[:, :, 2], img[:, :, 3]))
    output1 = process(img1, device, model, args.fp16)
    output2 = process(img2, device, model, args.fp16)
    return cv2.merge((output1[:, :, 0], output1[:, :, 1], output1[:, :, 2], output2[:, :, 2]))

def removeAlpha(img, device, model, is_fp16):
    img1 = np.copy(img[:, :, :3])
    output = process(img1, device, model, args.fp16)
    return cv2.cvtColor(output, cv2.COLOR_BGR2BGRA)


def crop_seamless(img, scale):
    img_height, img_width = img.shape[:2]
    y, x = 16 * scale, 16 * scale
    h, w = img_height - (32 * scale), img_width - (32 * scale)
    img = img[y:y+h, x:x+w]
    return img

def makeAlphaBinary(alpha, threshold):
    _, alpha = cv2.threshold(alpha, threshold, 1, cv2.THRESH_BINARY)
    return alpha

def makeAlphaTernery(alpha, threshold, boundry_offset):
    half_transparent_lower_bound = threshold - boundry_offset
    half_transparent_upper_bound = threshold + boundary_offset
    return np.where(alpha < half_transparent_lower_bound, 0, np.where(
        alpha <= half_transparent_upper_bound, .5, 1))