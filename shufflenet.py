import os
import sys

from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace


input_file = Path("download/bnps-checkpoint-300000.pth.tar")


index_string = (2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)


# Architecture	FLOPs	#Params	Top-1	Top-5
# (2, 1, 0, 1, 2, 0, 2, 0, 2, 0, 2, 3, 0, 0, 0, 0, 3, 2, 3, 3)	323M	3.5M	25.6	8.0

template = {'layer_1': 'k3', 'layer_2': 'k5', 'layer_3': 'k5', 'layer_4': 'xcep', 'layer_5': 'k5', 'layer_6': 'k3', 'layer_7': 'xcep', 'layer_8': 'xcep', 'layer_9': 'xcep', 'layer_10': 'xcep', 'layer_11': 'k5', 'layer_12': 'xcep', 'layer_13': 'k3', 'layer_14': 'k3', 'layer_15': 'k3', 'layer_16': 'k5', 'layer_17': 'k5', 'layer_18': 'k3', 'layer_19': 'k5', 'layer_20': 'k5'}


arch = {}
for i in range(20):
    arch[f'layer_{i + 1}'] = ['k3', 'k5', 'k7', 'xcep'][index_string[i]]

with fixed_arch(arch):
    model = searchspace.ShuffleNetSpace(affine=True)

data = torch.load(input_file, map_location='cpu')['state_dict']
data = match_state_dict(data, model.state_dict())
model.load_state_dict(data)
model.eval()


# evaluate_on_imagenet(model, 'spos')
# model.cuda()
# evaluate_on_imagenet(model, 'spos', gpu=True, full=True, batch_size=256, num_workers=12)

json.dump(arch, open(f'generate/shufflenet.json', 'w'), indent=2)
json.dump({"affine": True}, open(f'generate/shufflenet.init.json', 'w'), indent=2)
torch.save(data, f'generate/shufflenet.pth')
