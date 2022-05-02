import nni.retiarii.hub.pytorch as searchspace

from cutils import evaluate_on_imagenet


for arch in ['mobilenetv3-small-050', 'mobilenetv3-small-075', 'mobilenetv3-small-100']:
    model = searchspace.MobileNetV3Space.load_searched_model(arch, pretrained=True, download=True)
    model.cuda()
    evaluate_on_imagenet(model, 'bicubic', gpu=True, full=True, batch_size=256, num_workers=12)

