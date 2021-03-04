import glob
import math
import os.path
from collections import OrderedDict

import cv2
import torch

import infrastructure.utils.dataops as ops
from infrastructure.image_manipulation import generate_upscale_function, make_image_seamless, crop_seamless, get_all_image_paths
from infrastructure.arg_handler import get_arguments
from infrastructure.model_handler import get_model_chain, confirm_and_create_paths, load_model, ModelDetails

# Get arguments from CLI
args = get_arguments()

# Confirm folders exist and normalize paths
confirm_and_create_paths(args.input, args.output)
input_folder = os.path.normpath(args.input)
output_folder = os.path.normpath(args.output)

# Process model chain argument
model_chain = get_model_chain(args.model)

# Setup/configure device
device = torch.device('cpu' if args.cpu else f'cuda:{args.device_id}')
if args.fp16: torch.set_default_tensor_type(torch.HalfTensor)

# Setup placeholders for model data (channels, kind, etc) and actual model
last_model_data = ModelDetails()
model = None

# Start user feedback
print('Model{:s}: {:s}\nUpscaling...'.format(
      's' if len(model_chain) > 1 else '',
      ', '.join([os.path.splitext(os.path.basename(x))[0] for x in model_chain])))

# Get the paths for all images
image_paths = get_all_image_paths(input_folder, args.reverse)

# Store the maximum split depths for each model in the chain
# TODO: there might be a better way of doing this but it's good enough for now
split_depths = {}

# Loop over each image and load it and process it
for idx, path in enumerate(image_paths, 1):
    
    # Generate output paths and confirm dir exists
    base = os.path.splitext(os.path.relpath(path, input_folder))[0]
    output_dir = os.path.dirname(os.path.join(output_folder, base))
    os.makedirs(output_dir, exist_ok=True)

    print(idx, base)
    if args.skip_existing and os.path.isfile(
            os.path.join(output_folder, '{:s}.png'.format(base))):
        print(" == Already exists, skipping == ")
        continue

    # Read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) < 3: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Make image seamless if needed
    img = make_image_seamless(img, args.seamless)
    final_scale = 1

    # for each image within paths run through model
    for i, model_path in enumerate(model_chain):

        img_height, img_width = img.shape[:2]

        # Load the model so we can access the scale
        if model_path != last_model_data.model_path:
            model, model_data = load_model(model_path, last_model_data, model, device)
            last_model_data = model_data

        # Generate a function to be used within auto splitting
        upscaleFunction = generate_upscale_function(device, model, args, last_model_data.in_nc, last_model_data.out_nc)

        # Split if needed
        if args.cache_max_split_depth and len(split_depths.keys()) > 0:
            rlt, depth = ops.auto_split_upscale(img, upscaleFunction, last_model_data.scale, max_depth=split_depths[i])
        else:
            rlt, depth = ops.auto_split_upscale(img, upscaleFunction, last_model_data.scale)
            split_depths[i] = depth

        final_scale *= last_model_data.scale

        # This is for model chaining
        img = rlt.astype('uint8')

    # If its seamless make sure its cropped correctly
    if args.seamless: rlt = crop_seamless(rlt, final_scale)

    # Save image
    cv2.imwrite(os.path.join(output_folder, '{:s}.png'.format(base)), rlt)
