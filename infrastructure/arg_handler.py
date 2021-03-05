import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--input', default='input', help='Input folder')
    parser.add_argument('--output', default='output', help='Output folder')
    parser.add_argument('--reverse', help='Reverse Order', action="store_true")
    parser.add_argument('--skip_existing', action="store_true",
                        help='Skip existing output files')
    parser.add_argument('--seamless', nargs='?', choices=['tile', 'mirror', 'replicate', 'alpha_pad'], default=None,
                        help='Helps seamlessly upscale an image. Tile = repeating along edges. Mirror = reflected along edges. Replicate = extended pixels along edges. Alpha pad = extended alpha border.')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of CUDA')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FloatingPoint16/Halftensor type for images')
    parser.add_argument('--device_id', help='The numerical ID of the GPU you want to use. Defaults to 0.',
                        type=int, nargs='?', default=0)
    parser.add_argument('--cache_max_split_depth', action='store_true',
                        help='Caches the maximum recursion depth used by the split/merge function. Useful only when upscaling images of the same size.')
    parser.add_argument('--binary_alpha', action='store_true',
                        help='Whether to use a 1 bit alpha transparency channel, Useful for PSX upscaling')
    parser.add_argument('--ternary_alpha', action='store_true',
                        help='Whether to use a 2 bit alpha transparency channel, Useful for PSX upscaling')
    parser.add_argument('--alpha_threshold', default=.5,
                        help='Only used when binary_alpha is supplied. Defines the alpha threshold for binary transparency', type=float)
    parser.add_argument('--alpha_boundary_offset', default=.2,
                        help='Only used when binary_alpha is supplied. Determines the offset boundary from the alpha threshold for half transparency.', type=float)
    parser.add_argument('--alpha_mode', help='Type of alpha processing to use. 0 is no alpha processing. 1 is BA\'s difference method. 2 is upscaling the alpha channel separately (like IEU). 3 is swapping an existing channel with the alpha channel.',
                        type=int, nargs='?', choices=[0, 1, 2, 3], default=0)
    return parser.parse_args()