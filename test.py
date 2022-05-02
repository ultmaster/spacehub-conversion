import nni.retiarii.hub.pytorch as searchspace

from cutils import evaluate_on_imagenet, evaluate_on_cifar10

# for arch in ['darts-v2']:
#     model = searchspace.DARTS.load_searched_model('darts-v2', pretrained=True, download=True)
#     model.cuda()
#     evaluate_on_cifar10(model, gpu=True, full=True, batch_size=128, num_workers=12)

for arch in ['spos']:
    model = searchspace.ShuffleNetSpace.load_searched_model('spos', pretrained=True, download=True)

    model.cuda()
    evaluate_on_imagenet(model, 'spos', gpu=True, full=True, batch_size=256, num_workers=12)


# for arch in ['mobilenetv3-small-050', 'mobilenetv3-small-075', 'mobilenetv3-small-100']:
# for name in ['014', '043', '114', '287', '481', '604']:
#     arch = 'cream-' + name
#     model = searchspace.MobileNetV3Space.load_searched_model(arch, pretrained=True, download=True)
#     model.cuda()

#     pre = None
#     if name == '014':
#         pre = 'not224-64'
#     if name == '043':
#         pre = 'not224-96'
#     if name == '114':
#         pre = 'not224-160'
#     evaluate_on_imagenet(model, pre, gpu=True, full=True, batch_size=256, num_workers=12)



