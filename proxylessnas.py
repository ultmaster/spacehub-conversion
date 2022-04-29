from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace



def convert(sample):
    widths = []
    widths.append(sample["first_conv"]["out_channels"])
    widths.append(sample["blocks"][0]["mobile_inverted_conv"]["out_channels"])
    res = {}
    global_counter = 1
    for s in range(2, 8):
        max_depth = 4 if s < 7 else 1
        local_blocks = sample["blocks"][global_counter:global_counter + max_depth]
        depth = sum([v["mobile_inverted_conv"]["name"] != 'ZeroLayer' for v in local_blocks])
        res[f's{s}_depth'] = depth
        counter = 0
        for i, block in enumerate(local_blocks):
            if block["mobile_inverted_conv"]["name"] != 'ZeroLayer':
                if i == 0:
                    widths.append(block["mobile_inverted_conv"]["out_channels"])
                res[f's{s}_i{counter}'] = f'k{block["mobile_inverted_conv"]["kernel_size"]}e{block["mobile_inverted_conv"]["expand_ratio"]}'
                counter += 1
        global_counter += max_depth
    widths.append(sample["feature_mix_layer"]["out_channels"])
    assert global_counter == len(sample["blocks"])
    return {"base_widths": widths}, res


for suffix in ['gpu', 'cpu', 'mobile']:
    print(suffix)
    kwargs, proxyless = convert(json.load(open(f'download/proxyless_{suffix}.config')))

    with fixed_arch(proxyless):
        net = searchspace.ProxylessNAS(**kwargs)
    state_dict = match_state_dict(torch.load(f'download/proxyless_{suffix}.pth', map_location='cpu')['state_dict'], net.state_dict())
    net.load_state_dict(state_dict)

    evaluate_on_imagenet(net)

    json.dump(proxyless, open(f'generate/proxyless-{suffix}.json', 'w'), indent=2)
    json.dump(kwargs, open(f'generate/proxyless-{suffix}.init.json', 'w'), indent=2)
    torch.save(state_dict, f'generate/proxyless-{suffix}.pth')

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
