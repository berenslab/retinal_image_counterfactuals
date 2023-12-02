import glob

from models.DeiT_utils.DeiT_robust_model import load_model_ext

from timm.models import create_model
import timm
from robust_finetuning.utils_rf import ft

try:
    import clip
    from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
except Exception as err:
    print(str(err))
from torchvision.models import resnet50 as resnet50_in1000
from counterfactual_utils.model_normalization import Cifar10Wrapper, ImageNetWrapper
from counterfactual_utils.run_file_helpers import factory_dict
from counterfactual_utils.run_file_helpers import models_dict as models_wrappers_dict
from counterfactual_utils.datasets.fundus_kaggle import get_EyePacs, get_FundusKaggle
from models.Anon1s_smaller_radius_net import PreActResNet18 as PreActResNet18_Anon1
from models.BiTM import KNOWN_MODELS as BiT_KNOWN_MODELS
from models.SAMNets import WideResNet
# from models.BiTM_2 import ResNetV2
from models.preact_resnet import PreActResNet18
from functools import partial
import torch.nn as nn
import socket
from robustness import model_utils, datasets

from robust_finetuning import utils_rf as rft_utils
from torchvision.datasets.folder import DatasetFolder, has_file_allowed_extension, \
    pil_loader, accimage_loader, default_loader
from .MadryImageNet1000resnet import load_model as load_model_Madry
from .MadryRobustImageNet1000resnet import load_model as load_model_MadryRobust
from .MadryRobustImageNet1000resnet import rm_substr_from_state_dict as rm_substr_from_state_dict_MadryRobust

from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from robustbench.model_zoo import model_dicts

from PIL import Image
from collections import OrderedDict
from torchvision import transforms
from perceptual_advex.utilities import get_dataset_model
import counterfactual_utils.datasets as dl
from counterfactual_utils.datasets.oct import get_oct
from torch.autograd import Variable

from counterfactual_utils.models.load_models_bethge_texture import load_model as load_model_bethge_texture
import os, sys
import shutil
import requests
import io
##from .Wrappers import *
import bagnets.pytorchnet

import numpy as np
import torch

__all__ = ['rm_substr_from_state_dict', 'Gowal2020UncoveringNet_load', 'PAT_load', 'RLAT_load', 'models_dict',
           'descr_args_generate', 'descr_args_rst_stab',
           'temperature_scaling_dl_dict', 'image_loader', 'ask_overwrite_folder', 'pretty',
           'loader_all', 'get_NVAE_MSE', 'get_NVAE_class_model',
           'FIDDataset', 'Evaluator_FID_base_path', 'Evaluator_model_names_dict',
           'FeatureDist_implemented']


def insensitive_glob(pattern):
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c

    return glob.glob(''.join(map(either, pattern)))


def get_base_dir(project_folder=None):
    if project_folder is not None:
        return project_folder
    else:
        raise ValueError('No project folder specified')


def get_base_dir_Anon1_finetuning(project_folder):
    machine_name = socket.gethostname()

    base = 'RATIO/robust_finetuning/trained_models/'

    return base


# To save RAM, for now do not use the FeatureDist
FeatureDist_implemented = lambda dataset_name: dataset_name in ['cifar10'] and False
Ensemble_second_classifier_implemented = lambda dataset_name: dataset_name in ['funduskaggle']


def loader_all(use_wrapper, device, model, kwargs, device_ids=None, prewrapper=None):
    assert prewrapper is not None
    if device_ids is None:
        return prewrapper(model(**kwargs)) if use_wrapper else model(**kwargs).to(device)
    else:
        return nn.DataParallel(prewrapper(model(**kwargs)), device_ids=device_ids)

    # ImageNet1000ModelsPath = f'{get_base_dir()}/ACSM/ImageNet1000Models'


Evaluator_FID_base_path = 'ACSM/exp/logits_evolution_FID/image_samples'

Evaluator_model_names_eyepacs = [f'benchmark-Max:experimental,rmtIN1000:ResNet50,plain_17-01-2023_16:22:33',  #0: binary, plain, bs=128
                                 f'benchmark-MadryRobust:l2:experimental,MadryRobust:l2,TRADES_02-06-2023_12:10:59', #1: binary, eps=0.01, MadryRobust
                                 f'benchmark-Max:experimental,rmtIN1000:ResNet50,plain_24-01-2023_20:42:28',  #2: 5-class, plain, mse+ce, bs=8
                                 f'benchmark-MadryRobust:l2:experimental,MadryRobust:l2,TRADES_17-02-2023_18:30:38',  #3: 5-class, eps 0.01, MadryRobust
                                ]

Evaluator_model_names_oct = [f'benchmark-Max:experimental,ResNet50,plain_29-11-2022_21:53:46', #0: plain, 4-class, scratch
                             f'benchmark-Max:experimental,ResNet50,TRADES_02-12-2022_15:53:20', #1: TRADES, 4-class, scratch
                             ]

Evaluator_model_names_dict = {'eyepacs': Evaluator_model_names_eyepacs,
                              'oct': Evaluator_model_names_oct}

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

#####

model_type_to_folder = {'resnet50': 'ResNet50',
                        'rmtin1000': 'rmtIN1000',
                        'madryrobust': 'MadryRobust'}


model_name_to_folder = {'oct': 'OCTModels', 
                        'eyepacs': 'EyePacsModels'
                       }

dict_noises_like = {'uniform': lambda x: 2 * torch.rand_like(x) - 1,
                    'gaussian': lambda x: torch.randn_like(x)}

#####


def make_dataset(directories, extensions=None):
    instances = []
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for directory in directories:
        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, torch.Tensor([-1])
                    instances.append(item)
    return instances


# ToDo: Class index doesn't matter here, right?
class FIDDataset(DatasetFolder):

    def __init__(self, roots, transform=None):
        super(DatasetFolder, self).__init__(root=roots[0], transform=transform)

        self.samples = make_dataset(roots, IMG_EXTENSIONS)
        self.imgs = self.samples
        self.loader = default_loader

    def _find_classes(self, dir):
        classes = [None for d in os.scandir(dir) if d.name.endswith('last.png')]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class ImageNetCustom(DatasetFolder):

    def __init__(self, root, train=False, download=True, transform=None, target_transform=None,
                 loader=default_loader, label_mapping=None):
        super(ImageNetCustom, self).__init__(root, loader, IMG_EXTENSIONS,
                                             transform=transform,
                                             target_transform=target_transform)  # ,
        # label_mapping=label_mapping)
        # ToDo: only works for testset! Write train/test cases!
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def rm_substr_from_state_dict(state_dict, substr):
    new_state_dict = OrderedDict()
    for key in state_dict.keys():
        if substr in key:  # to delete prefix 'module.' if it exists
            new_key = key[len(substr):]
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def Gowal2020UncoveringNet_load(threat_model='L2', project_folder=None):
    if threat_model == 'L1.5':
        model_path = f'{get_base_dir(project_folder)}/Cifar10Models/Gowal2020Uncovering_improved/' \
                     f'model_2021-10-21 15:47:59.655507 lr=0.01000 piecewise-5-5-5 ep=3 attack=afw fts=50 seed=0 iter=10 finetune_model at=L1.5 eps=1.5 balanced 500k no_rot/' \
                     f'ep_3.pth'
    elif threat_model == 'L2':
        model_path = f'{get_base_dir(project_folder)}/Cifar10Models/Gowal2020Uncovering_improved/' \
                     f'model_2021-03-10 20:21:59.054345 lr=0.01000 piecewise-5-5-5 ep=3 ' \
                     f'attack=apgd fts=50 seed=0 iter=10 finetune_model at=Linf L1 ' \
                     f'balanced 500k no_rot/ep_3.pth'
    else:
        raise ValueError('Such norm is not supported.')

    model = model_dicts[BenchmarkDataset('cifar10')][ThreatModel('L2')]['Gowal2020Uncovering']['model']()
    model.load_state_dict(
        rm_substr_from_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), 'module.')
        , strict=True)
    print('loaded model GU')

    return model


def Augustin2020Adversarial_34_10Net_load(threat_model='L1.5'):
    if threat_model == 'L1.5':
        model_path = f'{get_base_dir()}/Cifar10Models/' \
                     f'Augustin2020Adversarial_34_10_extra/' \
                     f'ratio_finetuned_l15_asam.pth'
    else:
        raise ValueError('Such norm is not supported.')

    model = model_dicts[BenchmarkDataset('cifar10')][ThreatModel('L2')]['Sehwag2021Proxy']['model']()
    model.load_state_dict(
        rm_substr_from_state_dict(
            torch.load(model_path, map_location=torch.device('cpu')), 'module.')
        , strict=True)

    model = Cifar10Wrapper(model)

    return model


def MicrosoftNet(model_arch,
                 norm,
                 epsilon,
                 project_folder=None):
    if 'Wide_ResNet50_4' in model_arch:
        path_model = f'{get_base_dir(project_folder)}/robust_finetuning/external_models/wide_resnet50_4_{norm}_eps{epsilon}.ckpt'
    elif 'ResNet50' in model_arch:
        path_model = f'{get_base_dir(project_folder)}/ImageNet1000Models/microsoft/resnet50_{norm}_eps{epsilon}.ckpt'
    else:
        raise ValueError(f'Model arch {model_arch} is not implemented here.')

    model, checkpoint = model_utils.make_and_restore_model(
        arch=model_arch.lower(), dataset=datasets.ImageNet(''), resume_path=path_model)

    return model


def MadryNet(norm,
             improved,
             num_pretrained_epochs,
             epsilon_finetuned=None,
             project_folder=None,
             device='cuda:1'
             ):
    model_paths_dict = {
        'l2_improved_1l1.5_ep': 'Madry_l2_improved/model_2021-10-21 20:21:20.122644 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=afw fts=2 seed=0 iter=10 finetune_model at=L1.5 eps=12.5 balanced no_rot/ep_1.pth',
        'l2_improved_1l1_ep': 'Madry_l2_improved/model_2021-03-17 09:25:30.985477 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=2 seed=0 iter=15 finetune_model at=L1 balanced no_rot/ep_1.pth',
        'l2_improved_1_ep': 'Madry_l2_improved/model_2021-03-16 11:38:45.988619 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=2 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
        'l2_improved_3_ep': 'Madry_l2_improved/checkpoint/ep_3.pt', #'Madry_l2_improved/model_2021-05-02 15:03:55.621750 imagenet lr=0.01000 piecewise-5-5-5 ep=3 wd=0.0001 attack=apgd fts=2 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_3.pth',
        'linf_improved_1_ep': 'Madry_linf_improved/model_2021-03-15 13:03:18.873067 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=1 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
    }

    ImageNet1000ModelsPath = f'{get_base_dir(project_folder)}/ImageNet1000Models'

    model = load_model_Madry(modelname='Engstrom2019Robustness',
                             norm=norm,
                             device=device)
    if improved:
        if epsilon_finetuned is not None:
            model_path = os.path.join(ImageNet1000ModelsPath,
                                      model_paths_dict[norm + f'_improved_{str(epsilon_finetuned)}_eps'])
        else:
            model_path = os.path.join(ImageNet1000ModelsPath,
                                      model_paths_dict[norm + f'_improved_{str(num_pretrained_epochs)}_ep'])
        state_dict = torch.load(model_path, map_location='cpu')
        model.model.load_state_dict(state_dict, strict=True)

    return model

def MadryRobustNet(dataset_name,
                   arch,
                   model_name_id,
                   num_classes,
                   norm,
                   device,
                   project_folder=''):
    dataset_name_to_folder_dict = {'oct': 'OCTModels',
                                   'eyepacs': 'EyePacsModels'}
    state_dict_base_path = os.path.join(get_base_dir(project_folder), 'ImageNet1000Models/MadryRobustResNet50/')
    model = load_model_MadryRobust(modelname='Engstrom2019Robustness',
                                   norm=norm,
                                   device=device,
                                   state_dict_base_path=state_dict_base_path)
    while hasattr(model, 'model'):
        model = model.model

    model_arch = 'resnet50'
    # num_classes = 2

    print(model)

    model = ft(
        model_arch, model, num_classes, False)

    model.cuda()
    model.eval()

    model_path =  \
            insensitive_glob(
                os.path.join(get_base_dir(project_folder), dataset_name_to_folder_dict[dataset_name.lower()], arch,
                             f"*{model_name_id}*/best.pth"))[0]
    state_dict = torch.load(model_path, map_location=device)
    state_dict = rm_substr_from_state_dict_MadryRobust(state_dict,
                                            ['module.', 'model.'], ['normalizer.', 'attacker.', 'normalize.'])
    model.load_state_dict(state_dict)
    model = models_wrappers_dict[dataset_name.lower()](model)

    return model

def Anon1s_smaller_radius(eps, project_folder=None):
    model_paths_dict = {
        '0.25l2': 'model_2021-05-06 18:29:13.639852 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.25 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
        '0.1l2': 'model_2021-05-06 19:57:35.842192 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.1 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
        '0.02l2': 'model_2021-05-06 20:21:58.358943 lr=0.05000 piecewise-5-5-5 ep=3 attack=apgd act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L2 eps=.02 balanced no_rot continued [model_2021-03-15 14:05:42.938809 ep_72]/ep_3.pth',
        '0.5l2': 'l2_0.5/pretr_L2.pth',
        '0.75l2': 'model_2021-09-23 16:42:33.694120 lr=0.05000 superconverge ep=30 attack=apgd act=softplus1 fts=rand seed=0 iter=5 finetune_model at=L2 eps=.75 balanced no_wd4bn no_rot/ep_30.pth',
        '1l2': 'model_2021-09-23 12:39:26.072195 lr=0.05000 superconverge ep=30 attack=apgd act=softplus1 fts=rand seed=0 iter=5 finetune_model at=L2 eps=1. balanced no_wd4bn no_rot/ep_30.pth',
        '12l1': 'l1_12/pretr_L1.pth',
        '8,255linf': 'linf_8_255/pretr_Linf.pth',
        # l1.5 finetuned
        '0.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-31 13:50:41.062428 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=.5 balanced no_wd4bn no_rot/ep_30.pth',
        '1l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:23.725710 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=1 balanced no_wd4bn no_rot/ep_30.pth',
        '1.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-20 17:56:14.827435 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=1.5 balanced no_wd4bn no_rot/ep_30.pth',
        '2l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:23.735285 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=2. balanced no_wd4bn no_rot/ep_30.pth',
        '2.5l1.5': f'{get_base_dir(project_folder)}/ImageNet1000Models/Anon1_smaller_radius/model_2021-10-30 14:25:58.906229 lr=0.05000 superconverge ep=30 attack=afw act=softplus1 fts=rand seed=0 iter=10 finetune_model at=L1.5 eps=2.5 balanced no_wd4bn no_rot/ep_30.pth'
    }

    model = PreActResNet18_Anon1(n_cls=10, activation='softplus1')
    model_path = os.path.join(get_base_dir(project_folder), 'Anon1_smaller_radius', model_paths_dict[eps])
    state_dict = torch.load(model_path)
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)

    return model


def PAT_load(arch='resnet50',
             project_folder=None):
    model_path = f'{get_base_dir(project_folder)}/Cifar10Models/PAT/cifar/pat_self_0.5.pt',
    # This works for resnet50
    _, model = get_dataset_model(
        dataset='cifar',
        arch=arch,
        checkpoint_fname=model_path,
    )

    # model = getattr(torchvision_models, arch)(pretrained=pretrained)
    # state = torch.load(model_path)
    # model.load_state_dict(state['model'])

    # model = AlexNetFeatureModel(model)
    # model.load_state_dict(torch.load(model_path)['model'])

    return model


def RLAT_load(project_folder=None):
    model_path = f'{get_base_dir(project_folder)}/Cifar10Models/RLAT/rlat-eps=0.05-augmix-cifar10/rlat-eps=0.05-augmix-cifar10.pt'
    model = PreActResNet18(n_cls=10)
    model.load_state_dict(torch.load(model_path)['last'])
    model.eval()
    return model


def BiTM_get_weights(bit_variant):
    response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


def BiTM_load(model_name='', class_labels=None, dataset='cifar10'):
    # weights_cifar10 = get_weights(model_name)
    # model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)  # NOTE: No new head.
    # model.load_from(weights_cifar10)
    # model.eval()
    print('BiT model name is', model_name, 'dataset is', dataset)
    print('head_size is', len(class_labels))
    # model_name = 'BiT-M-R50x1'
    ##model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)
    model = BiT_KNOWN_MODELS[model_name](head_size=len(class_labels), zero_head=True)
    # ToDo: only BiT-M-R50x1 is supported currently
    print('model is', model)

    model.load_from(np.load(
        f"{get_base_dir()}/BiT-pytorch/big_transfer/BiT-M-R50x1.npz"))  ## (BiTM_get_weights('BiT-M-R50x1-CIFAR10'))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(f"{get_base_dir()}/BiT-pytorch/big_transfer/output/{dataset}/bit.pth.tar",
                            map_location="cpu")
    print('checkpoint', f"{get_base_dir()}/BiT-pytorch/big_transfer/output/{dataset}/bit.pth.tar")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model



def CLIP_model(model_name='', device=None, class_labels=None):
    model_, preprocess = clip.load(name=model_name, device=device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_labels]).to(device)
    text_features = model_.encode_text(text_inputs).type(torch.float32)
    text_features /= text_features.norm(dim=1, keepdim=True)
    model = partial(CLIP_model_, model=model_, text_features=text_features, preprocess=preprocess)
    return model


def CLIP_model_(image, text_features, model, preprocess):
    # image_input = preprocess(image)
    image_features = model.encode_image(image).type(torch.float32)
    # print('norm', image_features.norm(dim=1, keepdim=True))
    image_features = image_features / (image_features.norm(dim=1, keepdim=True))
    logits = 100 * image_features @ text_features.T
    # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    # values, indices = similarity[0].topk(10)
    return logits


def SAMNets(device='cuda:0'):
    depth, width_factor, dropout = 16, 8, 0.0
    dataset = 'cifar10'
    m_path = f'{get_base_dir()}/SAM_pytorch/sam/example/trained_models/model_2021-10-19 16:37:26.596415/ep_200.pth'
    model = WideResNet(depth, width_factor, dropout, in_channels=3, labels=10).to(device)
    state_dict = torch.load(m_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = models_wrappers_dict[dataset](model)
    model.eval()
    return model


def BethgeNets(model_name, device):
    return load_model_bethge_texture(model_name, device)

def BagNets(model_name, device):

    BagNetsDict = {'bagnet17': bagnets.pytorchnet.bagnet17}

    return BagNetsDict[model_name.lower()](pretrained=True).to(device)


def MaxNets(dataset_name, arch, model_name_id, num_classes, img_size, device='cuda:0', project_folder=''):
    if 'rmtIN1000:' in arch:
        # if dataset_name.lower() == 'oct':
        #     n_cls = 4
        # elif dataset_name.lower() == 'eyepacs':
        #     n_cls = 2 #5
        # else:
        if dataset_name.lower() not in ['oct', 'eyepacs']:
            raise ValueError(f"The dataset {dataset_name.lower()} is not supported!")

        model_arch = arch.replace("rmtIN1000:", "").lower()
        additional_hidden = 0

        print(f'[Replacing the last layer with {additional_hidden} '
              f'hidden layers and 1 classification layer that fits the {dataset_name} dataset with num classes = {num_classes}.]')

        model = timm.create_model(model_arch, num_classes=num_classes, in_chans=3, pretrained=True)

    else:
        if img_size in factory_dict:
            model, model_name, config = factory_dict[img_size].build_model(arch, num_classes)
        else:
            raise ValueError(f'Model factories are supported only for image sizes {factory_dict.keys()},'
                             f' and not for {img_size}!')

    dataset_name_to_folder_dict = {'funduskaggle': 'fundusKaggleModels',
                                   'cifar10': 'Cifar10Models',
                                   'oct': 'OCTModels',
                                   'eyepacs': 'EyePacsModels'} #temp change to EyePacsModels 

    if '_ft' in model_name_id:
        ep = model_name_id.split('ep=')[1]
        print(
            f'searching in {os.path.join(get_base_dir(project_folder), dataset_name_to_folder_dict[dataset_name.lower()], f"*{model_name_id}*/ep_{ep}.pth")}')

        state_dict_file = \
        insensitive_glob(os.path.join(get_base_dir(project_folder), dataset_name_to_folder_dict[dataset_name.lower()],
                                      f"*{model_name_id}*/ep_{ep}.pth"))[0]
    elif 'lr=' in model_name_id:
        ep = 3
        temp_base = 'FundusModels/plain'
        print(
            f'searching in {os.path.join(temp_base, f"*{model_name_id}*/ep_{ep}.pth")}')
        state_dict_file = insensitive_glob(
            os.path.join(project_folder, temp_base, "ft_ep_3.pth"))[0]

        #state_dict_file = \
        #    insensitive_glob(os.path.join(temp_base,
        #                                  f"*{model_name_id}*/ep_{ep}.pth"))[0]
    else:

        print(
            f'searching in {os.path.join(get_base_dir(project_folder), dataset_name_to_folder_dict[dataset_name.lower()], arch, f"*{model_name_id}*/best.pth")}')

        state_dict_file = \
            insensitive_glob(
                os.path.join(get_base_dir(project_folder), dataset_name_to_folder_dict[dataset_name.lower()], arch,
                             f"*{model_name_id}*/best.pth"))[0]

    print(f'resotring file from {state_dict_file}')

    state_dict = torch.load(state_dict_file, map_location=device)
    model.load_state_dict(state_dict)
    model = models_wrappers_dict[dataset_name.lower()](model)

    return model


def DeiTNets(model_name='deit_small_patch16_224_adv',
             model_path='ImageNet1000Models/DeiT/model_2021-12-16 16:48:41.667201 imagenet lr=0.01000 piecewise-5-5-5 ep=1 wd=0.0001 attack=apgd fts=3 seed=0 iter=5 15 finetune_model at=Linf L1 balanced no_rot/ep_1.pth',
             project_folder=None):
    model = load_model_ext(model_name)
    ckpt_path = os.path.join(project_folder, model_path)
    a = torch.load(ckpt_path)
    print(dir(a))
    model.load_state_dict(a)

    return model


def XCITNets(model_name='',
             model_path='',
             project_folder=None):
    model_path = model_path.replace('_', '-') + '.pth'
    model = create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None
    )

    ckpt_path = os.path.join(project_folder, 'ImageNet1000Models', 'XCIT', model_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    checkpoint_model = checkpoint['model']

    """
    state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
    """
    model.load_state_dict(checkpoint_model, strict=True)

    return model


def Anon1FinetuningNets(dataset_name, arch, model_name_id, num_classes, additional_hidden=0, project_folder=None):
    # ToDo: do we even need this resume path?
    arch = arch.lower()
    resume_path = f'{get_base_dir(project_folder)}/RATIO/robust_finetuning/external_models/wide_resnet50_4_l2_eps1.ckpt'
    if 'max=' in arch:
        arch = arch.replace('max=', '')
        # ToDo: use variable for img_size
        img_size = 224
        if img_size in factory_dict:
            model, model_name, config = factory_dict[img_size].build_model(arch, num_classes)
        else:
            raise ValueError(f'Model factories are supported only for image sizes {factory_dict.keys()},'
                             f' and not for {img_size}!')

        state_dict_file = \
            insensitive_glob(os.path.join(get_base_dir_Anon1_finetuning(),
                                          f'{model_name_id.replace("SPACE", " ")}*/ep_9.pth'))[0]

        print(f'resotring file from {state_dict_file}')
        state_dict = torch.load(state_dict_file, map_location='cpu')
        model.load_state_dict(state_dict)
        model = models_wrappers_dict[dataset_name.lower()](model)

    else:
        print(f'[Replacing the last layer with {additional_hidden} '
              f'hidden layers and 1 classification layer that fits the {dataset_name} dataset.]')

        model, checkpoint = model_utils.make_and_restore_model(arch=arch, dataset=datasets.ImageNet(''),
                                                               resume_path=resume_path)

        while hasattr(model, 'model'):
            model = model.model

        model = rft_utils.ft(
            arch, model, num_classes, additional_hidden)

        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=datasets.ImageNet(''),
                                                               add_custom_forward=False)
        state_dict_file = insensitive_glob(os.path.join(get_base_dir_Anon1_finetuning(),
                                                        f'{model_name_id.replace("SPACE", " ")}*/ep_26.pth'))[0]  # 36

        state_dict = torch.load(state_dict_file)
        model.load_state_dict(state_dict)

        # model.cuda()
        # model.eval()
    return model


BiTMWrapper_mean = torch.tensor([0.5, 0.5, 0.5])
BiTMWrapper_std = torch.tensor([0.5, 0.5, 0.5])

BiTMCIFAR10Wrapper_size = 128
BiTMIN1000Wrapper_size = 480

try:
    models_dict = {
        # CIFAR10 Models
        'ResNet50': lambda **kwargs: Cifar10Wrapper(resnet50(**kwargs)),
        'ResNet50IN1000': lambda **kwargs: ImageNetWrapper(resnet50_in1000(**kwargs)),
        'BiT-M-R50x1': lambda **kwargs: BiTMWrapper(BiTM_load(**kwargs),
                                                    mean=BiTMWrapper_mean,
                                                    std=BiTMWrapper_std,
                                                    size=BiTMCIFAR10Wrapper_size),
        """
        'BiT-M-R50x1IN1000': lambda **kwargs: BiTMWrapper(BiTM_load(**kwargs),
                                                          mean=BiTMWrapper_mean,
                                                          std=BiTMWrapper_std,
                                                          size=BiTMIN1000Wrapper_size),
        """

        'ViT-B': lambda **kwargs: BiTMWrapper(ViT_load(**kwargs),
                                              mean=BiTMWrapper_mean,
                                              std=BiTMWrapper_std,
                                              size=None),
        'Gowal2020Uncovering_improved': Gowal2020UncoveringNet_load,
        'Augustin2020Adversarial_34_10_extra_improved': Augustin2020Adversarial_34_10Net_load,
        'PAT_improved': PAT_load,
        'RLAT_improved': RLAT_load,
        # 'ResNet18_finetuned_ep15_improved': ResNet18_finetuned_load,
        'CLIP': CLIP_model,
        # ImageNet1000 Models
        'Madry': MadryNet,
        'MadryRobust': MadryRobustNet,
        'Anon1_small_radius_experimental': Anon1s_smaller_radius,
        'Microsoft': MicrosoftNet,
        'Max': MaxNets,
        'Anon1:finetuning': Anon1FinetuningNets,
        'SAM': SAMNets,
        'DeiTrobust_experimental': DeiTNets,
        'XCITrobust': XCITNets,
        'bethge-texture': BethgeNets,
        'bagnet': BagNets,
    }

except Exception as err:
    print(str(err))

descr_args_rst_stab = lambda project_folder: {
    'path': f'{get_base_dir(project_folder)}/Cifar10Models/RST_stab/AdvACET_24-02-2020_14:41:39/'
            'cifar10_rst_stab.pt.ckpt',
    'model': 'wrn-28-10'}


def descr_args_generate(threat_model=None, pretrained=False,
                        is_experimental=False, dataset_='cifar10', model_name=None, project_folder=None):
    if is_experimental:
        if threat_model is not None:
            return {'threat_model': threat_model, 'project_folder': project_folder}
        elif pretrained:
            print('using pretrained model')
            return {'pretrained': True}
        else:
            return {'project_folder': project_folder}
    else:
        return {
            'model_name': model_name,
            'dataset': dataset_,
            'threat_model': threat_model,
            'project_folder': project_folder
        }


# ToDo: generalize for different BiT models
temperature_scaling_dl_dict = lambda batch_size, img_size, project_folder, data_folder, model_name=None: \
    {
        # 'oct': get_oct(split='test', batch_size=batch_size, size=img_size, 
        #                project_folder=project_folder,
        #                data_folder=data_folder, augm_type='none'),
        'eyepacs': get_EyePacs(split='val', batch_size=batch_size, augm_type='none', size=img_size, 
                               balanced= False, data_folder=data_folder)
    }

full_dataset_dict = lambda batch_size, img_size, project_folder, data_folder, model_name=None: \
    {
         'imagenet1000': dl.get_ImageNet1000_idx(
            idx_path=None,
            no_subset_sampler=True,
            model_name=model_name,
            batch_size=batch_size, project_folder=project_folder, data_folder=data_folder),

    }

loader = lambda imsize: transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])


def tensor_loader(img_filepath, imsize, img_id):
    tensor_filepath = '/'.join(img_filepath.split('/')[:-1])
    tensor_filepath = glob.glob(os.path.join(tensor_filepath, '*.pt'), recursive=True)
    assert len(tensor_filepath) == 1, 'Only one tensor is expected to be in the folder.'
    tensor_ = torch.load(tensor_filepath)

    return tensor_[img_id][:, :, -imsize:].unsqueeze(0).cuda()


def image_loader(image_name, imsize):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(imsize)(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU


def ask_overwrite_folder(folder, no_interactions, fatal=True, FID_calculation=False):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except Exception as err:
            print(str(err))
    elif no_interactions and not FID_calculation:
        pass
        # shutil.rmtree(folder)
        # os.makedirs(folder)
    elif not FID_calculation:
        response = input(f"Folder '{folder}' already exists. Overwrite? (Y/N)")
        if response.upper() == 'Y':
            shutil.rmtree(folder)
            os.makedirs(folder)
        elif fatal:
            print("Output image folder exists. Program halted.")
            sys.exit(0)
    else:
        pass


def get_weights(bit_variant):
    response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


def get_NVAE_class_model(models_dict, class_id):
    if class_id in models_dict:
        return models_dict[class_id]
    else:
        return None


def get_NVAE_MSE(image, model, batch_size):
    if model is not None:
        image_out = model(image)
        output = model.decoder_output(image_out[0])
        output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) \
            else output.sample()
        return (output_img - image).view(batch_size, -1).norm(p=2, dim=1)
    else:
        return 'NA'


