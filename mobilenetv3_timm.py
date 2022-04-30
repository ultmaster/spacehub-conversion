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
            res[f's{stage_idx}_i{local_idx}_ks'] = sample[0][0]
        else:
            res[f's{stage_idx}_i{local_idx}_exp'] = sample[i][1]
            res[f's{stage_idx}_i{local_idx}_ks'] = sample[i][0]
            if stage_idx in [2, 4, 5]:
                res[f's{stage_idx}_i{local_idx}_se'] = 'se' if sample[i][3] else 'identity'
        local_idx += 1
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

kwargs = dict(
    base_widths=(16, 24, 40, 80, 112, 160, 960, 1280),
    expand_ratios=(1.0, 2.0, 2.3, 2.5, 3., 4., 6.),
    bn_eps=1e-5,
    bn_momentum=0.1,
    # se_before_activation=True
)

with fixed_arch(arch):
    net = searchspace.MobileNetV3Space(**kwargs)


official = torch.load('download/mobilenetv3_large_100_ra-f55367f5.pth')
# official["classifier.0.weight"] = official["classifier.0.weight"].view(1280, 960, 1, 1)

# trans_keys = [x for x in official.keys() if "conv.5.fc.0.weight" in x or "conv.5.fc.2.weight" in x]
# for k in trans_keys:
#     official[k] = official[k][:, :, None, None]

# official["blocks.9.0.weight"] = official["blocks.9.0.weight"].view(1280, 960)

state_dict = match_state_dict(
    official,
    dict(net.state_dict())
)
# print(list(net.state_dict().keys()))
net.load_state_dict(state_dict)
net.eval()



evaluate_on_imagenet(net)

json.dump(arch, open(f'generate/mobilenetv3-large-100.json', 'w'), indent=2)
json.dump(kwargs, open(f'generate/mobilenetv3-large-100.init.json', 'w'), indent=2)
torch.save(state_dict, f'generate/mobilenetv3-large-100.pth')
