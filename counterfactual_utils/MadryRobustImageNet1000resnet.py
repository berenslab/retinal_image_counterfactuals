import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models as torch_models
from typing import Tuple
from torch import Tensor
model_class_dict = {'pt_vgg': torch_models.vgg16_bn,
                    'pt_resnet': torch_models.resnet50,
                    'pt_inception': torch_models.inception_v3,
                    'pt_densenet': torch_models.densenet121}


def rm_substr_from_state_dict(state_dict, substrs: list = None, del_keys: list = None):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():

        if substrs is not None:
            new_key = key
            for substr in substrs:
                new_key = new_key.replace(substr, '')

            new_state_dict[new_key] = state_dict[key]

    new_second_state_dict = OrderedDict()

    for key in new_state_dict.keys():
        if del_keys is not None:
            new_second_state_dict[key] = new_state_dict[key]
            for substr_del in del_keys:
                if substr_del in key:
                    del new_second_state_dict[key]
    return new_second_state_dict


class PretrainedModel():
    def __init__(self, modelname, pretrained=True):
        #super(PretrainedModel, self).__init__()
        model_pt = model_class_dict[modelname](pretrained=pretrained)
        #model.eval()
        self.model = nn.DataParallel(model_pt.cuda())
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    def predict(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)
    def forward(self, x):
        out = (x - self.mu) / self.sigma
        return self.model(out)
    def __call__(self, x):
        return self.predict(x)
class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()
        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))
    def forward(self, input: Tensor) -> Tensor:
        print('Madry input shape', input.shape)
        return (input - self.mean) / self.std
def normalize_model(model: nn.Module, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)
def Engstrom2019RobustnessNet(ckpt, device):
    '''def __init__(self):
        #super(Engstrom2019RobustnessNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=1000)
        model_pt = model_class_dict['pt_resnet'](pretrained=False)
        #super
        #self.model = nn.DataParallel(model_pt.cuda().eval())
        self.model = model_pt.cuda().eval()
        self.model.eval()
        self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(1, 3, 1, 1).cuda()
        self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(1, 3, 1, 1).cuda()
    def forward(self, x, return_features=False):
        x = (x - self.mu) / self.sigma
        #return super(Engstrom2019RobustnessNet, self).forward(x, return_features=return_features)
        return self.model(x)
    def __call__(self, x):
        return self.forward(x)'''
    model_pt = model_class_dict['pt_resnet'](pretrained=False).cuda(device)
    model_pt.eval()

    state_dict = rm_substr_from_state_dict(
            torch.load(ckpt, map_location=torch.device('cpu'))['model'],
            ['module.', 'model.'], ['normalizer.', 'attacker.']
    )

    model_pt.load_state_dict(state_dict, strict=True)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # normalized_model = normalize_model(model=model_pt, mean=mean, std=std)
    model_pt.eval()
    print('model created')
    model_pt.cuda(device)
    return model_pt

models_Linf = OrderedDict([
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'data': 'imagenet_linf_4.pt',
        'ckpt_var': None,
    }),
])
models_L2 = OrderedDict([
    ('Engstrom2019Robustness', {
        'model': Engstrom2019RobustnessNet,
        'data': 'imagenet_l2_3_0.pt',
        'ckpt_var': None
    }),
])
other_models = OrderedDict()
model_dicts = {'linf': models_Linf, 'l2': models_L2, 'other': other_models}


def load_model(modelname, norm='other', device='cuda:1', state_dict_base_path=''):
    model_det = model_dicts[norm][modelname]

    model = model_det['model'](state_dict_base_path + model_det['data'], device)
    print('model loaded')
    print(type(model))
    return model
