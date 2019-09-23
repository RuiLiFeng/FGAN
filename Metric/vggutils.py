import torch
import torch.nn.functional as F
import torchvision.models.vgg as vgglib
import torch.utils.model_zoo as model_zoo

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

VGG_URL = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'


def load_vgg_from_local(arch='vgg19', cfg='E', batch_norm=False,
                        pretrained=True, vgg_dir=None, parallel=True, **kwargs):
    vgg = vgglib.VGG(vgglib.make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    vgg.load_state_dict(model_zoo.load_url(url=VGG_URL, model_dir='/gpub/temp/imagenet2012/hdf5'))
    vgg = (vgg.eval()).cuda()
    if parallel:
        print("Parallel VGG model...")
        vgg = torch.nn.DataParallel(vgg)
    return vgg

