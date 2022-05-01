from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace

from nni.retiarii.hub.pytorch.proxylessnas import make_divisible

from timm.models.mobilenetv3 import mobilenetv3_small_050, MobileNetV3


def convert(sample):
    from timm.models.efficientnet_builder import decode_arch_def

    decoded = decode_arch_def(sample)
    print(decoded)
    channels = [16]
    stride = [2]
    activation = ['hswish']

    res = {'stem_ks': 3}
    for stage_idx in range(6):
        channels.append(decoded[stage_idx][0]['out_chs'] if len(decoded[stage_idx]) > 0 else channels[-1])
        stride.append(decoded[stage_idx][0]['stride'] if len(decoded[stage_idx]) > 0 else 1)
        activation.append('relu' if len(decoded[stage_idx]) > 0 and decoded[stage_idx][0]['act_layer'] is not None else 'hswish')
        if 0 <= stage_idx <= 4:
            if stage_idx > 0:
                res[f's{stage_idx}_depth'] = len(decoded[stage_idx])
            if len(decoded[stage_idx]) > 0:
                for local_idx in range(len(decoded[stage_idx])):
                    res[f's{stage_idx}_i{local_idx}_se'] = 'se' if decoded[stage_idx][0].get('se_ratio') else 'identity'
                    s = decoded[stage_idx][local_idx]
                    if 'exp_ratio' in s:
                        res[f's{stage_idx}_i{local_idx}_exp'] = s['exp_ratio']
                    if 'dw_kernel_size' in s:
                        res[f's{stage_idx}_i{local_idx}_ks'] = s['dw_kernel_size']
    stride.append(1)
    activation.append('hswish')
    channels.append(1024)
    return res, channels, stride, activation



arch, channels, stride, activation = convert([
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
])

print(arch, channels)

channels = [make_divisible(c * 0.5, 8) if 0 < i < 7 else c for i, c in enumerate(channels)]
print(channels)

ratios = sorted(set([v for k, v in arch.items() if k.endswith('_exp')]))
print(ratios)

kwargs = dict(
    base_widths=channels,
    width_multipliers=1.0,
    expand_ratios=tuple(ratios),
    bn_eps=1e-5,
    bn_momentum=0.1,
    squeeze_excite=['optional'] * 5,
    activation=activation,
    stride=stride,
    depth_range=(1, 2),
)

with fixed_arch(arch):
    net = searchspace.MobileNetV3Space(**kwargs)


official = torch.load('download/mobilenetv3_small_050_lambc-4b7bbe87.pth')
# official["classifier.0.weight"] = official["classifier.0.weight"].view(1280, 960, 1, 1)

state_dict = match_state_dict(
    official,
    dict(net.state_dict())
)
net.load_state_dict(state_dict)
net.eval()

model_ref: MobileNetV3 = mobilenetv3_small_050(pretrained=True)
model_ref.load_state_dict(official)
model_ref.eval()

x = torch.randn(1, 16, 112, 112)
# import pdb; pdb.set_trace()

# print((model_ref.blocks[0](x) - net.blocks[0](x)).abs().sum())


evaluate_on_imagenet(net)

json.dump(arch, open(f'generate/mobilenetv3-small-050.json', 'w'), indent=2)
json.dump(kwargs, open(f'generate/mobilenetv3-small-050.init.json', 'w'), indent=2)
torch.save(state_dict, f'generate/mobilenetv3-small-050.pth')
