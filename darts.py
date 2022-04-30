from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet, evaluate_on_cifar10

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace

def convert(genotype):
    res = {
        'depth': 20,
        'width': 36,
    }
    for t in ['normal', 'reduce']:
        for i in range(4):
            res[f'{t}/op_{i+2}_0'] = genotype[t][i*2][0]
            res[f'{t}/op_{i+2}_1'] = genotype[t][i*2+1][0]
            res[f'{t}/input_{i+2}_0'] = genotype[t][i*2][1]
            res[f'{t}/input_{i+2}_1'] = genotype[t][i*2+1][1]
    return res


DARTS_V2 = dict(
    normal=[
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)
    ],
    reduce_concat=[2, 3, 4, 5]
)


arch = convert(DARTS_V2)

with fixed_arch(arch):
    net = searchspace.DARTS(width=(24, 36))

# import pdb; pdb.set_trace()

data = torch.load('/mnt/data/nni-checkpoints/spacehub/cifar10_model.pt', map_location='cpu')
for k in [k for k in data.keys() if k.startswith("auxiliary_head.")]:
    data.pop(k)
state_dict = match_state_dict(data, net.state_dict())

net.load_state_dict(state_dict)

evaluate_on_cifar10(net)


json.dump(arch, open(f'generate/darts-v2.json', 'w'), indent=2)
json.dump({"width": 36, "num_cells": 20}, open(f'generate/darts-v2.init.json', 'w'), indent=2)
torch.save(state_dict, f'generate/darts-v2.pth')


# template = {
#     'depth': 8,
#     'width': 16,
#     'normal/op_2_0': 'avg_pool_3x3',
#     'normal/op_2_1': 'avg_pool_3x3',
#     'normal/op_3_0': 'none',
#     'normal/op_3_1': 'dil_conv_3x3',
#     'normal/op_4_0': 'max_pool_3x3',
#     'normal/op_4_1': 'dil_conv_3x3',
#     'normal/op_5_0': 'sep_conv_5x5',
#     'normal/op_5_1': 'skip_connect',
#     'normal/input_2_0': 0,
#     'normal/input_2_1': 1,
#     'normal/input_3_0': 1,
#     'normal/input_3_1': 0,
#     'normal/input_4_0': 2,
#     'normal/input_4_1': 2,
#     'normal/input_5_0': 4,
#     'normal/input_5_1': 0,
#     'reduce/op_2_0': 'sep_conv_5x5',
#     'reduce/op_2_1': 'none',
#     'reduce/op_3_0': 'sep_conv_5x5',
#     'reduce/op_3_1': 'none',
#     'reduce/op_4_0': 'dil_conv_5x5',
#     'reduce/op_4_1': 'sep_conv_5x5',
#     'reduce/op_5_0': 'sep_conv_3x3',
#     'reduce/op_5_1': 'sep_conv_5x5',
#     'reduce/input_2_0': 0,
#     'reduce/input_2_1': 0,
#     'reduce/input_3_0': 0,
#     'reduce/input_3_1': 1,
#     'reduce/input_4_0': 3,
#     'reduce/input_4_1': 1,
#     'reduce/input_5_0': 4,
#     'reduce/input_5_1': 3
# }