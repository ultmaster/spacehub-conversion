def convert(decoded):
    channels = [16]

    res = {'stem_ks': 3}
    for stage_idx in range(7):
        res[f's{stage_idx}_width_mult'] = 1.0
        channels.append(decoded[stage_idx][0]['out_chs'] if len(decoded[stage_idx]) > 0 else channels[-1])
        if 0 <= stage_idx <= 5:
            res[f's{stage_idx}_depth'] = len(decoded[stage_idx])
            if res[f's{stage_idx}_depth'] > 0:
                for local_idx in range(len(decoded[stage_idx])):
                    res[f's{stage_idx}_i{local_idx}_se'] = 'se' if decoded[stage_idx][0].get('se_ratio') else 'identity'
                    s = decoded[stage_idx][local_idx]
                    if 'exp_ratio' in s:
                        res[f's{stage_idx}_i{local_idx}_exp'] = s['exp_ratio']
                    if 'dw_kernel_size' in s:
                        res[f's{stage_idx}_i{local_idx}_ks'] = s['dw_kernel_size']
    res['s7_width_mult'] = 1.0

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

    print(new_arch)


config_481 = [
    [0],
    [3, 4, 3, 1],
    [3, 2, 3, 0],
    [3, 3, 3, 1, 1],
    [3, 3, 3, 3],
    [3, 3, 3, 3],
    [0]
]


convert_outer(config_481)