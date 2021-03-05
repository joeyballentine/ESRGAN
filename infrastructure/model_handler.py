from . import state_manipulation
from .utils import architecture as arch
import os.path
import sys

class ModelDetails():
    model_path = None
    in_nc = None
    out_nc = None
    nf = None
    nb = None
    scale = None
    kind = None

    def __eq__(self, other): 
        if not isinstance(other, ModelDetails): return NotImplemented

        return self.in_nc == other.in_nc and self.out_nc == other.out_nc \
        and self.nf == other.nf and self.nb == other.nb \
        and self.scale == other.scale and self.kind == other.kind

def check_model_path(model_path):
    if os.path.exists(model_path):
        return model_path
    elif os.path.exists(os.path.join('./models/', model_path)):
        return os.path.join('./models/', model_path)
    else:
        print('Error: Model [{:s}] does not exist.'.format(model))
        sys.exit(1)

def get_model_chain(modelString):
    model_chain = modelString.split('+') if '+' in modelString else modelString.split('>')

    for idx, model in enumerate(model_chain):

        interpolations = model.split(
            '|') if '|' in modelString else model.split('&')

        if len(interpolations) > 1:
            for i, interpolation in enumerate(interpolations):
                interp_model, interp_amount = interpolation.split(
                    '@') if '@' in interpolation else interpolation.split(':')
                interp_model = check_model_path(interp_model)
                interpolations[i] = f'{interp_model}@{interp_amount}'
            model_chain[idx] = '&'.join(interpolations)
        else:
            model_chain[idx] = check_model_path(model)
    return model_chain

def confirm_and_create_paths(inputFolder, outputFolder):
    if not os.path.exists(inputFolder):
        print('Error: Folder [{:s}] does not exist.'.format(inputFolder))
        sys.exit(1)
    elif os.path.isfile(inputFolder):
        print('Error: Folder [{:s}] is a file.'.format(inputFolder))
        sys.exit(1)
    elif os.path.isfile(outputFolder):
        print('Error: Folder [{:s}] is a file.'.format(outputFolder))
        sys.exit(1)
    elif not os.path.exists(outputFolder):
        os.mkdir(outputFolder)

def extractInformation(state_dict, last_model_data):
    model_data = ModelDetails()

    # extract model information
    scale2 = 0
    max_part = 0
    if 'f_HR_conv1.0.weight' in state_dict:
        model_data.kind = 'SPSR'
        scalemin = 4
    else:
        model_data.kind = 'ESRGAN'
        scalemin = 6

    for part in list(state_dict):
        parts = part.split('.')
        n_parts = len(parts)

        if n_parts == 5 and parts[2] == 'sub':
            model_data.nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > scalemin and parts[0] == 'model' and parts[2] == 'weight':
                scale2 += 1
            if part_num > max_part:
                max_part = part_num
                model_data.out_nc = state_dict[part].shape[0]

    model_data.scale = 2 ** scale2
    model_data.in_nc = state_dict['model.0.weight'].shape[1]

    if model_data.kind == 'SPSR':
        model_data.out_nc = state_dict['f_HR_conv1.0.weight'].shape[0]

    model_data.nf = state_dict['model.0.weight'].shape[0]

    if (model_data != last_model_data):
        if model_data.kind == 'ESRGAN':
            model = arch.RRDB_Net(model_data.in_nc, model_data.out_nc, model_data.nf, model_data.nb, gc=32, 
                upscale=model_data.scale, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
        elif model_data.kind == 'SPSR':
            model = arch.SPSRNet(model_data.in_nc, model_data.out_nc, model_data.nf, model_data.nb, gc=32, 
                upscale=model_data.scale, norm_type=None, act_type='leakyrelu', mode='CNA', upsample_mode='upconv')

        return model, model_data


def load_model(model_path, last_model_data, model, device):
    state_dict = state_manipulation.loadStateDict(model_path)

    if 'conv_first.weight' in state_dict:
        print('Attempting to convert and load a new-format model')
        state_dict = state_manipulation.transposeStateDict(state_dict)

    model, model_data = extractInformation(state_dict, last_model_data)
    model_data.model_path = model_path

    model.load_state_dict(state_dict, strict=True)
    del state_dict
    
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device), model_data