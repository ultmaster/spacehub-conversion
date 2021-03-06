import os
import sys

from pathlib import Path
import torch
import pprint
import json

sys.path.insert(0, os.path.expanduser('~/AceNAS'))

from searchspace.proxylessnas import ProxylessConfig, ProxylessNAS

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace


# top-1 75.254 latency: 84.60

ACENAS_MOBILE = {
    # top-1 75.254 latency: 84.60
    'acenas-m1': '0:0:0:0:1:0:0:0:0:0:1:1:1:1:1:0:1:0:0:0:0:0:1:1:1:1:0:0:0:0:1:0:0:0:1:1:0:0:2:0:1:0:2:1:0:0:1:0:0:1:0:2:0:0:0:2:1:0:2:1:0:2:0:0:2:0:0:0:2:1:0:0:0',
    # top-1 75.07 latency: 84.59
    'acenas-m2': '0:0:0:1:0:0:2:0:1:0:0:1:2:1:1:0:0:1:0:0:0:0:2:0:1:1:0:0:0:2:1:0:2:1:1:1:1:0:2:0:1:0:1:1:0:1:0:0:1:1:0:0:1:0:0:2:1:0:1:1:0:1:0:0:1:1:0:0:2:1:0:0:0',
    # top-1 75.11 latency: 84.92
    'acenas-m3': '0:0:0:0:0:0:0:1:0:1:1:1:0:0:1:0:1:0:0:0:0:1:1:0:1:0:0:0:0:1:1:0:2:1:0:0:1:0:2:1:1:0:2:1:0:2:0:0:2:0:0:1:0:0:0:2:1:0:2:0:0:2:1:0:0:0:0:0:1:1:0:0:0'
}


subtype = 'm2'

input_file = Path(f"~/AceNAS/data/checkpoints/acenas-{subtype}.pth.tar").expanduser()
index_string = ACENAS_MOBILE['acenas-' + subtype]

def tf_indices_to_pytorch_spec(tf_indices, pytorch_space):
    tf_indices = list(map(int, tf_indices.split(':')))
    if len(tf_indices) == 22:
        assert len(tf_indices) == len(pytorch_space)
        return {k: v[i] if isinstance(v, list) else v[0][i] for i, (k, v) in zip(tf_indices, pytorch_space.items())}

    indices = [3, 6, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42, 45, 48, 51, 55, 58, 61, 64, 68]
    assert len(indices) == 21
    assert len(pytorch_space) == 22
    result = {}
    for i, key in zip([None] + indices, pytorch_space.keys()):
        if i is None:
            assert len(pytorch_space[key]) == 1
            chosen = pytorch_space[key][0]
        else:
            kernel_size = [3, 5, 7][tf_indices[i]]
            expand_ratio = [3, 6][tf_indices[i + 1]]
            skip = tf_indices[i + 2]
            if skip:
                chosen = 'skip'
            else:
                chosen = f'k{kernel_size}e{expand_ratio}'
        assert chosen in pytorch_space[key] or chosen in pytorch_space[key][0], f'{i}, {chosen}, {pytorch_space[key]}'
        result[key] = chosen
    print(result)
    return result


def create_proxylessnas_model(indices,
                              widths=None,
                              num_classes=1000,
                              drop_rate=0.0,
                              bn_momentum=0.1,
                              bn_eps=1e-3,
                              **kwargs):
    if widths is None:
        widths = [16, 32, 40, 80, 96, 192, 320]
    config = ProxylessConfig(
        stem_width=32,
        final_width=1280,
        width_mult=1.0,
        num_labels=num_classes,
        dropout_rate=drop_rate,
        stages=[
            {'depth_range': [1, 1], 'exp_ratio_range': [1], 'kernel_size_range': [3], 'width': widths[0], 'downsample': False},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[1], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[2], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[3], 'downsample': True},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[4], 'downsample': False},
            {'depth_range': [1, 4], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[5], 'downsample': True},
            {'depth_range': [1, 1], 'exp_ratio_range': [3, 6], 'kernel_size_range': [3, 5, 7], 'width': widths[6], 'downsample': False}
        ]
    )
    model = ProxylessNAS(config, reset_parameters=False)
    model.reset_parameters(bn_momentum=bn_momentum, bn_eps=bn_eps, track_running_stats=True)
    model.activate(tf_indices_to_pytorch_spec(indices, model.searchspace()))
    model.prune()
    return model

model = create_proxylessnas_model(index_string)
model.load_state_dict(torch.load(input_file, map_location='cpu'))


acenas_style_sample = tf_indices_to_pytorch_spec(index_string, model.searchspace())

def convert(sample):
    res = {}
    for s in range(2, 8):
        depth = sum([v != 'skip' for k, v in sample.items() if k.startswith(f's{s}')])
        res[f's{s}_depth'] = depth
        counter = 0
        for i in range(4 if s < 7 else 1):
            candidate = [v for k, v in sample.items() if k.startswith(f's{s}b{i + 1}')][0]
            if candidate == 'skip':
                continue
                # candidate = 'k3e3'  # fallback
            res[f's{s}_i{counter}'] = candidate
            counter += 1
    return res


acenas_m1 = convert(acenas_style_sample)

pprint.pprint(acenas_m1)

with fixed_arch(acenas_m1):
    net = searchspace.ProxylessNAS()
state_dict = match_state_dict(torch.load(input_file, map_location='cpu'), net.state_dict())
net.load_state_dict(state_dict)

evaluate_on_imagenet(net)

json.dump(acenas_m1, open(f'generate/acenas-{subtype}.json', 'w'), indent=2)
torch.save(state_dict, f'generate/acenas-{subtype}.pth')

# template = {
#     's2_depth': 2,
#     's2_i0': 'k5e3',
#     's2_i1': 'k7e3',
#     's2_i2': 'k5e6',
#     's2_i3': 'k3e3',
#     's3_depth': 2,
#     's3_i0': 'k3e6',
#     's3_i1': 'k5e6',
#     's3_i2': 'k3e6',
#     's3_i3': 'k5e3',
#     's4_depth': 1,
#     's4_i0': 'k7e6',
#     's4_i1': 'k7e6',
#     's4_i2': 'k3e3',
#     's4_i3': 'k7e3',
#     's5_depth': 2,
#     's5_i0': 'k7e3',
#     's5_i1': 'k7e3',
#     's5_i2': 'k5e6',
#     's5_i3': 'k5e6',
#     's6_i0': 'k3e6',
#     's7_i0': 'k3e3'
# }
