from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace


def convert(sample, last_width):
    from timm.models.efficientnet_builder import decode_arch_def

    print(decode_arch_def(sample))



arch = convert([
    # stage 0, 112x112 in
    ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
    # stage 1, 56x56 in
    ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
    # stage 2, 28x28 in
    ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
    # stage 3, 14x14 in
    ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
    # stage 4, 14x14in
    ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
    # stage 6, 7x7 in
    ['cn_r1_k1_s1_c576'],  # hard-swish
], 1024)

# kwargs = dict(
#     base_widths=(16, 24, 40, 80, 112, 160, 960, 1280),
#     expand_ratios=(1.0, 2.0, 2.3, 2.5, 3., 4., 6.),
#     bn_eps=1e-5,
#     bn_momentum=0.1,
#     # se_before_activation=True
# )

# with fixed_arch(arch):
#     net = searchspace.MobileNetV3Space(**kwargs)


# official = torch.load('download/mobilenetv3_large_100_ra-f55367f5.pth')
# # official["classifier.0.weight"] = official["classifier.0.weight"].view(1280, 960, 1, 1)

# # trans_keys = [x for x in official.keys() if "conv.5.fc.0.weight" in x or "conv.5.fc.2.weight" in x]
# # for k in trans_keys:
# #     official[k] = official[k][:, :, None, None]

# # official["blocks.9.0.weight"] = official["blocks.9.0.weight"].view(1280, 960)

# state_dict = match_state_dict(
#     official,
#     dict(net.state_dict())
# )
# # print(list(net.state_dict().keys()))
# net.load_state_dict(state_dict)
# net.eval()

# # TODO: bicubic interpolation

# evaluate_on_imagenet(net)

# json.dump(arch, open(f'generate/mobilenetv3-small-050.json', 'w'), indent=2)
# json.dump(kwargs, open(f'generate/mobilenetv3-small-050.init.json', 'w'), indent=2)
# torch.save(state_dict, f'generate/mobilenetv3-small-050.pth')
