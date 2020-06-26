import argparse
import glob
import math
import os.path
import sys

import cv2
import numpy as np
import torch

import architecture as arch

parser = argparse.ArgumentParser()
parser.register('type', bool, (lambda x: x.lower()
                               in ("true")))
parser.add_argument('model')
parser.add_argument('--input', default='LR', help='Input folder')
parser.add_argument('--output', default='results',
                    help='Output folder')
parser.add_argument('--tile_size', default=512,
                    help='Tile size for splitting', type=int)
parser.add_argument('--seamless', default=False,
                    help='Seamless upscaling or not', type=bool)
parser.add_argument('--cpu', default=False,
                    help='Use CPU instead of CUDA', type=bool)
parser.add_argument('--binary_alpha', default=False,
                    help='Whether to use a 1 bit alpha transparency channel, Useful for PSX upscaling', type=bool)
parser.add_argument('--alpha_threshold', default=.5,
                    help='Only used when binary_alpha is supplied. Defines the alpha threshold for binary transparency', type=float)
parser.add_argument('--alpha_boundary_offset', default=.2,
                    help='Only used when binary_alpha is supplied. Determines the offset boundary from the alpha threshold for half transparency.', type=float)
args = parser.parse_args()

model_chain = args.model.split('>')
for idx, model in enumerate(model_chain):
    if os.path.exists(model):
        pass
    elif os.path.exists('./models/' + model):
        model_chain[idx] = os.path.join('models', model)
    else:
        print('Error: Model [{:s}] does not exist.'.format(model))
        sys.exit(1)

if not os.path.exists(args.input):
    print('Error: Folder [{:s}] does not exist.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.input):
    print('Error: Folder [{:s}] is a file.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.output):
    print('Error: Folder [{:s}] is a file.'.format(args.output))
    sys.exit(1)
elif not os.path.exists(args.output):
    os.mkdir(args.output)

device = torch.device('cpu' if args.cpu else 'cuda')

test_img_folder = os.path.join(os.path.normpath(args.input), '*')
output_folder = os.path.normpath(args.output)


def split(img, dim, overlap):
    '''
    Creates an array of equal length image chunks to use for upscaling

            Parameters:
                    img (array): Numpy image array
                    dim (int): Number to use for length and height of image chunks
                    overlap (int): The amount of overlap between chunks

            Returns:
                    imgs (array): Array of numpy image "chunks"
                    num_horiz (int): Number of horizontal chunks
                    num_vert (int): Number of vertical chunks
    '''
    img_height, img_width, img_channels = img.shape
    num_horiz = math.ceil(img_width / dim)
    num_vert = math.ceil(img_height / dim)
    imgs = []
    for i in range(num_vert):
        for j in range(num_horiz):
            tile = img[i * dim:i * dim + dim + overlap,
                       j * dim:j * dim + dim + overlap].copy()
            imgs.append(tile)
    return imgs, num_horiz, num_vert

# This method is a somewhat modified version of BlueAmulet's original pymerge script that is able to use my split chunks


def merge(rlts, scale, overlap, img_height, img_width, img_channels, num_horiz, num_vert):
    '''
    Merges the image chunks back together

            Parameters:
                    rlts (array): The resulting images from ESRGAN
                    scale (int): The scale of the model that was applied
                    overlap (int): The amount of overlap between chunks
                    img_height (int): The height of the original image
                    img_width (int): The width of the original image
                    num_horiz (int): Number of horizontal chunks
                    num_vert (int): Number of vertical chunks

            Returns:
                    rlt (array): Numpy image array of the resulting merged image
    '''
    rlt_overlap = int(overlap * scale)

    rlts_fin = [[None for x in range(num_horiz)]
                for y in range(num_vert)]

    rlt = np.zeros((img_height * scale,
                    img_width * scale, img_channels))

    c = 0
    for tY in range(num_vert):
        for tX in range(num_horiz):
            img = rlts[tY*num_horiz+tX]
            shape = img.shape
            c = max(c, shape[2])
            rlts_fin[tY][tX] = img

    for tY in range(num_vert):
        for tX in range(num_horiz):
            img = rlts_fin[tY][tX]
            if img.shape[2] == 3 and c == 4:  # pad with solid alpha channel
                img = np.dstack((img, np.full(img.shape[:-1], 1.)))
                rlts_fin[tY][tX] = img
            shape = img.shape
            # Fade out edges
            # Left edge
            if tX > 0:
                for x in range(rlt_overlap):
                    img[:, x] *= ((x + 1)/(rlt_overlap + 1))
            # Top edge
            if tY > 0:
                for y in range(rlt_overlap):
                    img[y, :] *= ((y + 1)/(rlt_overlap + 1))
            # Right edge
            if tX < num_horiz - 1:
                for x in range(rlt_overlap):
                    iX = x + shape[1] - rlt_overlap
                    img[:, iX] *= ((rlt_overlap - x) /
                                   (rlt_overlap + 1))
            # Bottom edge
            if tY < num_vert - 1:
                for y in range(rlt_overlap):
                    iY = y + shape[0] - rlt_overlap
                    img[iY, :] *= ((rlt_overlap - y) /
                                   (rlt_overlap + 1))

    baseY = 0
    for tY in range(num_vert):
        baseX = 0
        for tX in range(num_horiz):
            img = rlts_fin[tY][tX]
            shape = img.shape

            # Copy non overlapping image data
            x1 = (0 if tX == 0 else rlt_overlap)
            y1 = (0 if tY == 0 else rlt_overlap)
            x2 = shape[1]
            y2 = shape[0]
            rlt[baseY+y1:baseY+y2, baseX +
                x1:baseX+x2] = img[y1:y2, x1:x2]

            # Blend left
            if tX > 0:
                rlt[baseY+y1:baseY+y2, baseX:baseX +
                    rlt_overlap] += img[y1:y2, :rlt_overlap]

            # Blend up
            if tY > 0:
                rlt[baseY:baseY+rlt_overlap, baseX +
                    x1:baseX+x2] += img[:rlt_overlap, x1:x2]

            # Blend corner
            if tX > 0 and tY > 0:
                rlt[baseY:baseY+rlt_overlap, baseX:baseX +
                    rlt_overlap] += img[:rlt_overlap, :rlt_overlap]

            baseX += shape[1] - rlt_overlap
        baseY += shape[0] - rlt_overlap
    return rlt

# This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao


def process(img):
    '''
    Does the processing part of ESRGAN. This method only exists because the same block of code needs to be ran twice for images with transparency.

            Parameters:
                    img (array): The image to process

            Returns:
                    rlt (array): The processed image
    '''
    if img.shape[2] == 3:
        img = img[:, :, [2, 1, 0]]
    elif img.shape[2] == 4:
        img = img[:, :, [2, 1, 0, 3]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    if output.shape[0] == 3:
        output = output[[2, 1, 0], :, :]
    elif output.shape[0] == 4:
        output = output[[2, 1, 0, 3], :, :]
    output = np.transpose(output, (1, 2, 0))
    return output

# This code is a somewhat modified version of BlueAmulet's fork of ESRGAN by Xinntao


def esrgan(imgs, model_path):
    global last_model, last_in_nc, last_out_nc, last_nf, last_nb, last_scale, model
    '''
    Runs ESRGAN on all the images passed in with the specified model

            Parameters:
                    imgs (array): The images to run ESRGAN on
                    model_path (string): The model to use

            Returns:
                    rlts (array): The processed images
    '''
    # torch.device('cpu' if args.cpu else 'cuda')

    if model_path != last_model:
        state_dict = torch.load(model_path)

        if 'conv_first.weight' in state_dict:
            print('Error: Attempted to load a new-format model')
            sys.exit(1)

        # extract model information
        scale2 = 0
        max_part = 0
        for part in list(state_dict):
            parts = part.split('.')
            n_parts = len(parts)
            if n_parts == 5 and parts[2] == 'sub':
                nb = int(parts[3])
            elif n_parts == 3:
                part_num = int(parts[1])
                if part_num > 6 and parts[2] == 'weight':
                    scale2 += 1
                if part_num > max_part:
                    max_part = part_num
                    out_nc = state_dict[part].shape[0]
        upscale = 2 ** scale2
        in_nc = state_dict['model.0.weight'].shape[1]
        nf = state_dict['model.0.weight'].shape[0]

        if in_nc != last_in_nc or out_nc != last_out_nc or nf != last_nf or nb != last_nb or upscale != last_scale:
            model = arch.RRDB_Net(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu',
                                  mode='CNA', res_scale=1, upsample_mode='upconv')
            last_in_nc = in_nc
            last_out_nc = out_nc
            last_nf = nf
            last_nb = nb
            last_scale = upscale

        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

    rlts = []
    for img in imgs:
        # read image
        img = img * 1. / np.iinfo(img.dtype).max

        if img.ndim == 3 and img.shape[2] == 4 and last_in_nc == 3 and last_out_nc == 3:
            shape = img.shape
            img1 = np.copy(img[:, :, :3])
            img2 = np.copy(img[:, :, :3])
            for c in range(3):
                img1[:, :, c] *= img[:, :, 3]
                img2[:, :, c] = (img2[:, :, c] - 1) * img[:, :, 3] + 1

            output1 = process(img1)
            output2 = process(img2)
            alpha = 1 - np.mean(output2-output1, axis=2)

            if args.binary_alpha:
                transparent = 0.
                opaque = 1.
                half_transparent = .5
                half_transparent_lower_bound = args.alpha_threshold - args.alpha_boundary_offset
                half_transparent_upper_bound = args.alpha_threshold + args.alpha_boundary_offset
                alpha = np.where(alpha < half_transparent_lower_bound, transparent,
                                 np.where(alpha <= half_transparent_upper_bound,
                                 half_transparent, opaque))

            output = np.dstack((output1, alpha))
            shape = output1.shape
            divalpha = np.where(alpha < 1. / 510., 1, alpha)
            for c in range(shape[2]):
                output[:, :, c] /= divalpha
            output = np.clip(output, 0, 1)
        else:
            if img.ndim == 2:
                img = np.tile(np.expand_dims(img, axis=2),
                              (1, 1, min(last_in_nc, 3)))
            if img.shape[2] > last_in_nc:  # remove extra channels
                print('Warning: Truncating image channels')
                img = img[:, :, :last_in_nc]
            # pad with solid alpha channel
            elif img.shape[2] == 3 and last_in_nc == 4:
                img = np.dstack((img, np.full(img.shape[:-1], 1.)))
            output = process(img)

        output = (output * 255.).round()

        rlts.append(output)
    # torch.cuda.empty_cache()
    return rlts, upscale


def make_seamless(img):
    img_height, img_width, img_channels = img.shape
    img = cv2.hconcat([img, img, img])
    img = cv2.vconcat([img, img, img])
    y, x = img_height - 16, img_width - 16
    h, w = img_height + 32, img_width + 32
    img = img[y:y+h, x:x+w]
    return img


def crop_seamless(img, scale):
    img_height, img_width, img_channels = img.shape
    y, x = 16 * scale, 16 * scale
    h, w = img_height - (32 * scale), img_width - (32 * scale)
    img = img[y:y+h, x:x+w]
    return img


# state_dict = torch.load(model_path)

last_model = None
last_in_nc = None
last_out_nc = None
last_nf = None
last_nb = None
last_scale = None
model = None

print('Model{:s}: {:s}\nUpscaling...'.format(
      's' if len(model_chain) > 1 else '',
      ', '.join([os.path.splitext(os.path.basename(x))[0] for x in model_chain])))

idx = 0
for path in glob.glob(test_img_folder):
    if os.path.isdir(path):  # skip directories
        continue
    idx += 1
    base = os.path.splitext(os.path.basename(path))[0]
    print(idx, base)
    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # img = img * 1. / np.iinfo(img.dtype).max

    for model_path in model_chain:

        img_height, img_width, img_channels = img.shape
        dim = args.tile_size
        overlap = 16

        while dim > overlap and (img_height % dim < overlap or img_width % dim < overlap):
            dim -= overlap

        do_split = img_height > dim or img_width > dim

        if args.seamless:
            img = make_seamless(img)

        if do_split:
            imgs, num_horiz, num_vert = split(
                img, dim, overlap)
        else:
            imgs = [img]

        rlts, scale = esrgan(imgs, model_path)

        if do_split:
            rlt = merge(rlts, scale, overlap, img_height,
                        img_width, img_channels, num_horiz, num_vert)
        else:
            rlt = rlts[0]

        if args.seamless:
            rlt = crop_seamless(rlt, scale)

        img = rlt.astype('uint8')

    cv2.imwrite(os.path.join(output_folder, '{:s}.png'.format(base)), rlt)
