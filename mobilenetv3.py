from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace


def convert(sample, last_width):
    res = {}
    stage_idx = -1
    local_idx = -1
    for i in range(len(sample)):
        if i == 0 or sample[i][2] != sample[i - 1][2]:
            stage_idx += 1
            res[f's{stage_idx}_width'] = sample[i][2]
            local_idx = 0
        if 1 <= stage_idx <= 5:
            res[f's{stage_idx}_depth'] = sum(sample[k][2] == sample[i][2] for k in range(len(sample)))
        if i == 0:
            res[f's{stage_idx}_i{local_idx}'] = 3
        else:
            res[f's{stage_idx}_i{local_idx}_exp'] = sample[i][1]
            res[f's{stage_idx}_i{local_idx}_ks'] = sample[i][0]
    res['s6_width'] = res['s5_width'] * res['s5_i0_exp']
    res['s7_width'] = last_width
    return res


arch = convert([
    # k, t, c, SE, HS, s 
    [3,   1,  16, 0, 0, 1],
    [3,   4,  24, 0, 0, 2],  # 2
    [3,   3,  24, 0, 0, 1],
    [5,   3,  40, 1, 0, 2],  # 2
    [5,   3,  40, 1, 0, 1],
    [5,   3,  40, 1, 0, 1],
    [3,   6,  80, 0, 1, 2],  # 2
    [3, 2.5,  80, 0, 1, 1],
    [3, 2.3,  80, 0, 1, 1],
    [3, 2.3,  80, 0, 1, 1],
    [3,   6, 112, 1, 1, 1],  #
    [3,   6, 112, 1, 1, 1],
    [5,   6, 160, 1, 1, 2],  # 2
    [5,   6, 160, 1, 1, 1],
    [5,   6, 160, 1, 1, 1]
], 1280)

with fixed_arch(arch):
    net = searchspace.MobileNetV3Space()



# state_dict = match_state_dict(dict(net.state_dict())))

# Template

template = {
    's0_width': 8,
    's0_i0_ks': 5,
    's1_depth': 1,
    's1_i0_exp': 6.0,
    's1_i0_ks': 3,
    's1_width': 24,
    's1_i1_exp': 6.0,
    's1_i1_ks': 7,
    's1_i2_exp': 3.0,
    's1_i2_ks': 7,
    's1_i3_exp': 5.0,
    's1_i3_ks': 3,
    's2_depth': 1,
    's2_i0_exp': 6.0,
    's2_i0_ks': 3,
    's2_i0_se': 'identity',
    's2_width': 24,
    's2_i1_exp': 5.0,
    's2_i1_ks': 5,
    's2_i1_se': 'identity',
    's2_i2_exp': 3.0,
    's2_i2_ks': 5,
    's2_i2_se': 'se',
    's2_i3_exp': 4.0,
    's2_i3_ks': 7,
    's2_i3_se': 'se',
    's3_depth': 4,
    's3_i0_exp': 4.0,
    's3_i0_ks': 3,
    's3_width': 96,
    's3_i1_exp': 5.0,
    's3_i1_ks': 3,
    's3_i2_exp': 4.0,
    's3_i2_ks': 7,
    's3_i3_exp': 6.0,
    's3_i3_ks': 3,
    's4_depth': 1,
    's4_i0_exp': 2.0,
    's4_i0_ks': 7,
    's4_i0_se': 'identity',
    's4_width': 96,
    's4_i1_exp': 5.0,
    's4_i1_ks': 5,
    's4_i1_se': 'se',
    's4_i2_exp': 2.0,
    's4_i2_ks': 7,
    's4_i2_se': 'se',
    's4_i3_exp': 6.0,
    's4_i3_ks': 3,
    's4_i3_se': 'identity',
    's5_depth': 3,
    's5_i0_exp': 2.0,
    's5_i0_ks': 3,
    's5_i0_se': 'se',
    's5_width': 128,
    's5_i1_exp': 1.0,
    's5_i1_ks': 5,
    's5_i1_se': 'se',
    's5_i2_exp': 6.0,
    's5_i2_ks': 7,
    's5_i2_se': 'se',
    's5_i3_exp': 4.0,
    's5_i3_ks': 7,
    's5_i3_se': 'identity',
    's6_width': 320,
    's7_width': 1280
}