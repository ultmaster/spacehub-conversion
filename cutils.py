import numpy as np
import torch
import tqdm
import pprint

from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet


def match_state_dict(current_values: dict, expected_format: dict):
    current_values = dict(current_values)
    expected_format = dict(expected_format)
    result = {}
    missing_keys = []
    print('Length to checkpoint:', len(current_values))
    print('Length of expected state dict:', len(expected_format))
    for k, v in expected_format.items():
        not_found = True
        for key, cv in current_values.items():
            if cv.shape == v.shape:
                result[k] = cv
                current_values.pop(key)
                not_found = False
                break
        if not_found:
            missing_keys.append(k)
    if current_values or missing_keys:
        print('Remaining values: ')
        pprint.pprint({k: t.shape for k, t in current_values.items()})
        print('Missing keys: ', missing_keys)
    return result


def evaluate_on_imagenet(model):
    dataset = ImageNet('/mnt/data/imagenet', 'val', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]))
    subset = np.random.permutation(50000)[:200]
    dataloader = DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(subset))
    model.eval()
    with torch.no_grad():
        correct = total = 0
        pbar = tqdm.tqdm(dataloader, desc='Evaluating on ImageNet')
        for inputs, targets in pbar:
            logits = model(inputs)
            _, predict = torch.max(logits, 1)
            correct += (predict == targets).cpu().sum().item()
            total += targets.size(0)
            pbar.set_postfix({'correct': correct, 'total': total, 'acc': correct / total * 100})
    print('Overall accuracy (top-1):', correct / total * 100)
