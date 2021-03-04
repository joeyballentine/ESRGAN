import torch

def loadStateDict(model_path):
    # interpolating OTF, example: 4xBox:25&4xPSNR:75
    if (':' in model_path or '@' in model_path) and ('&' in model_path or '|' in model_path):
        interps = model_path.split('&')[:2]
        model_1 = torch.load(interps[0].split('@')[0])
        model_2 = torch.load(interps[1].split('@')[0])
        state_dict = OrderedDict()
        for k, v_1 in model_1.items():
            v_2 = model_2[k]
            state_dict[k] = (int(interps[0].split('@')[1]) / 100) * \
                v_1 + (int(interps[1].split('@')[1]) / 100) * v_2        
        return state_dict
    else:
        return torch.load(model_path)

def transposeStateDict(state_dict):
    old_net = {}
    items = []
    for k, v in state_dict.items():
        items.append(k)

    old_net['model.0.weight'] = state_dict['conv_first.weight']
    old_net['model.0.bias'] = state_dict['conv_first.bias']

    for k in items.copy():
        if 'RDB' in k:
            ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
            if '.weight' in k:
                ori_k = ori_k.replace('.weight', '.0.weight')
            elif '.bias' in k:
                ori_k = ori_k.replace('.bias', '.0.bias')
            old_net[ori_k] = state_dict[k]
            items.remove(k)

    old_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
    old_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
    old_net['model.3.weight'] = state_dict['upconv1.weight']
    old_net['model.3.bias'] = state_dict['upconv1.bias']
    old_net['model.6.weight'] = state_dict['upconv2.weight']
    old_net['model.6.bias'] = state_dict['upconv2.bias']
    old_net['model.8.weight'] = state_dict['HRconv.weight']
    old_net['model.8.bias'] = state_dict['HRconv.bias']
    old_net['model.10.weight'] = state_dict['conv_last.weight']
    old_net['model.10.bias'] = state_dict['conv_last.bias']
    return old_net