from pathlib import Path
import torch
import pprint
import json

from cutils import match_state_dict, evaluate_on_imagenet

from nni.retiarii import fixed_arch
import nni.retiarii.hub.pytorch as searchspace



def convert(sample):
    from timm.models.efficientnet_builder import decode_arch_def

    decoded = decode_arch_def(sample)
    # print(decoded)
    channels = [16]

    res = {'stem_ks': 3}
    for stage_idx in range(7):
        channels.append(decoded[stage_idx][0]['out_chs'] if len(decoded[stage_idx]) > 0 else channels[-1])
        if 0 <= stage_idx <= 5:
            res[f's{stage_idx}_depth'] = len(decoded[stage_idx])
            if res[f's{stage_idx}_depth'] > 0:
                for local_idx in range(len(decoded[stage_idx])):
                    s = decoded[stage_idx][local_idx]
                    if 'exp_ratio' in s:
                        res[f's{stage_idx}_i{local_idx}_exp'] = s['exp_ratio']
                    if 'dw_kernel_size' in s:
                        res[f's{stage_idx}_i{local_idx}_ks'] = s['dw_kernel_size']

    channels.append(1280)
    return res, channels



def convert_outer(arch_list):
    stem = ['ds_r1_k3_s1_e1_c16_se0.25', 'cn_r1_k1_s1_c320_se0.25']
    choice_block_pool = ['ir_r1_k3_s2_e4_c24_se0.25',
                         'ir_r1_k5_s2_e4_c40_se0.25',
                         'ir_r1_k3_s2_e6_c80_se0.25',
                         'ir_r1_k3_s1_e6_c96_se0.25',
                         'ir_r1_k5_s2_e6_c192_se0.25']
    arch_def = [[stem[0]]] + [[choice_block_pool[idx]
                               for repeat_times in range(len(arch_list[idx + 1]))]
                              for idx in range(len(choice_block_pool))] + [[stem[1]]]

    choices = {'kernel_size': [3, 5, 7], 'exp_ratio': [4, 6]}
    choices_list = [[x, y] for x in choices['kernel_size']
                    for y in choices['exp_ratio']]

    new_arch = []
    # change to child arch_def
    for i, (layer_choice, layer_arch) in enumerate(zip(arch_list, arch_def)):
        if len(layer_arch) == 1:
            new_arch.append(layer_arch)
            continue
        else:
            new_layer = []
            for j, (block_choice, block_arch) in enumerate(
                    zip(layer_choice, layer_arch)):
                kernel_size, exp_ratio = choices_list[block_choice]
                elements = block_arch.split('_')
                block_arch = block_arch.replace(
                    elements[2], 'k{}'.format(str(kernel_size)))
                block_arch = block_arch.replace(
                    elements[4], 'e{}'.format(str(exp_ratio)))
                new_layer.append(block_arch)
            new_arch.append(new_layer)

    return convert(new_arch)


for model_type in [14, 43, 114, 287, 481, 604]:
    if model_type == 481:
        arch_list = [
            [0], [
                3, 4, 3, 1], [
                3, 2, 3, 0], [
                3, 3, 3, 1, 1], [
                    3, 3, 3, 3], [
                        3, 3, 3, 3], [0]]
        image_size = 224
    elif model_type == 43:
        arch_list = [[0], [3], [3, 1], [3, 1], [3, 3, 3], [3, 3], [0]]
        image_size = 96
    elif model_type == 14:
        arch_list = [[0], [3], [3, 3], [3, 3], [3], [3], [0]]
        image_size = 64
    elif model_type == 114:
        arch_list = [[0], [3], [3, 3], [3, 3], [3, 3, 3], [3, 3], [0]]
        image_size = 160
    elif model_type == 287:
        arch_list = [[0], [3], [3, 3], [3, 1, 3], [3, 3, 3, 3], [3, 3, 3], [0]]
        image_size = 224
    elif model_type == 604:
        arch_list = [[0], [3, 3, 2, 3, 3], [3, 2, 3, 2, 3], [3, 2, 3, 2, 3],
                     [3, 3, 2, 2, 3, 3], [3, 3, 2, 3, 3, 3], [0]]
        image_size = 224

    arch, channels = convert_outer(arch_list)
    print(arch, channels)

    kwargs = dict(
        base_widths=channels,
        width_multipliers=1.0,
        expand_ratios=[4., 6.],
        bn_eps=1e-5,
        bn_momentum=0.1,
        squeeze_excite=['force'] * 6,
        activation=['swish'] * 9,
    )

    with fixed_arch(arch):
        net = searchspace.MobileNetV3Space(**kwargs)


    official = torch.load(f'download/cream/{model_type}.pth.tar', map_location='cpu')
    # import pdb; pdb.set_trace()
    
    official = official['state_dict_ema']

    state_dict = match_state_dict(
        official,
        dict(net.state_dict())
    )
    net.load_state_dict(state_dict)
    net.eval()

    net.cuda()
    evaluate_on_imagenet(net, f'not224-{image_size}', gpu=True, full=True, batch_size=256, num_workers=12)

# json.dump(arch, open(f'generate/mobilenetv3-cream-481.json', 'w'), indent=2)
# json.dump(kwargs, open(f'generate/mobilenetv3-cream-481.init.json', 'w'), indent=2)
# torch.save(state_dict, f'generate/mobilenetv3-cream-481.pth')
