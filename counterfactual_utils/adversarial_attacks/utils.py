import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import math
import torch.optim as optim
from .l1_projection import project_onto_l1_ball

from utils.spurious_features.activation_space import ActivationSpace
from utils.spurious_features.activations import activations_with_grad


l1_quantile = 0.99

def normalize_perturbation(perturbation, p):
    if p in ['inf', 'linf', 'Linf']:
        return perturbation.sign()
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        bs = perturbation.shape[0]
        pert_flat = perturbation.view(bs, -1)
        pert_normalized = F.normalize(pert_flat, p=2, dim=1)
        return pert_normalized.view_as(perturbation)
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        bs = perturbation.shape[0]

        spatial_threhsoliding = True
        apply_sign = False

        if spatial_threhsoliding:
            pert_channels_summed = torch.sum(perturbation.abs(), dim=1)
            pert_channels_summed_flat = pert_channels_summed.view(bs,-1)
            dim = pert_channels_summed_flat.shape[1]
            threshold_idx = int(dim * l1_quantile)
            sort_elements= torch.sort(pert_channels_summed_flat, dim=1, descending=False)[0]
            threshold_element = sort_elements[torch.arange(bs), threshold_idx]
            remove_idcs = (pert_channels_summed < threshold_element[:,None,None]).unsqueeze(1)
            perturbation = perturbation * (~remove_idcs)
            pert_flat = perturbation.view(bs, -1)

            if apply_sign:
                raise NotImplementedError()
            else:
                pert_flat = F.normalize(pert_flat, p=1, dim=1)
            return pert_flat.view_as(perturbation)
        else:
            pert_flat = perturbation.view(bs, -1)
            dim = pert_flat.shape[1]
            threshold_idx = int(dim * l1_quantile)
            num_non_zero = dim - threshold_idx
            sort_elements= torch.sort(pert_flat.abs(), dim=1, descending=False)[0]
            threshold_element = sort_elements[torch.arange(bs), threshold_idx]
            remove_idcs = pert_flat.abs() < threshold_element.unsqueeze(1)

            if apply_sign:
                pert_flat = (1. / num_non_zero) * pert_flat.sign()
                pert_flat[remove_idcs] = 0.0
            else:
                pert_flat[remove_idcs] = 0.0
                pert_flat = F.normalize(pert_flat, p=1, dim=1)
            return pert_flat.view_as(perturbation)
    else:
        raise NotImplementedError('Normalization only supports l2 and inf norm')


def project_perturbation(perturbation, eps, p):
    if p in ['inf', 'linf', 'Linf']:
        pert_normalized = torch.clamp(perturbation, -eps, eps)
        return pert_normalized
    elif p in [2, 2.0, 'l2', 'L2', '2']:
        pert_normalized = torch.renorm(perturbation, p=2, dim=0, maxnorm=eps)
        return pert_normalized
    elif p in [1, 1.0, 'l1', 'L1', '1']:
        pert_normalized = project_onto_l1_ball(perturbation, eps)
        return pert_normalized
    else:
        raise NotImplementedError('Projection only supports l1, l2 and inf norm')


def reduce(loss, reduction) :
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError('reduction not supported')

#############################################iterative PGD attack
def logits_diff_loss(out, y_oh, reduction='mean'):
    #out: density_model output
    #y_oh: targets in one hot encoding
    #confidence:
    out_real = torch.sum((out * y_oh), 1)
    out_other = torch.max(out * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = out_other - out_real

    return reduce(diff, reduction)

def conf_diff_loss(out, y_oh, reduction='mean'):
    #out: density_model output
    #y_oh: targets in one hot encoding
    #confidence:
    confidences = F.softmax(out, dim=1)
    conf_real = torch.sum((confidences * y_oh), 1)
    conf_other = torch.max(confidences * (1. - y_oh) - y_oh * 1e13, 1)[0]

    diff = conf_other - conf_real

    return reduce(diff, reduction)

def confidence_loss(out, y, reduction='mean'):
    confidences = F.softmax(out, dim=1)
    confidences_y = confidences[torch.arange(0, confidences.shape[0]), y]
    return reduce(confidences_y, reduction)

def log_confidence_loss(out, y, reduction='mean'):
    log_confidences = F.log_softmax(out, dim=1)
    confidences_y = log_confidences[torch.arange(0, log_confidences.shape[0]), y]
    return reduce(confidences_y, reduction)

def pca_component_loss(x, classifier, reduction='mean', args=None):
    # Load Activation Space, for now fixed
    #target_class = 94  # hummingbird
    #component_idx = 1  # hummingbird feeder

    #target_class = 554  # fireboat
    #component_idx = 0  # water streams

    #target_class = 2  # white shark
    #component_idx = 8 #10  # cages

    #target_class = 269  # timber wolf
    #component_idx = 2 #10  # cages

    #target_class = 563  # fountain pen
    #component_idx = 1  # text
    #component_idx = 2  # another text

    target_class = args.class_id_spurious
    component_idx = args.component_idx_spurious
    print('component id, class', component_idx, target_class)
    # Temperature Wrapper (DataParallel (ResizeMean ) )
    last_layer = classifier.model.module.model.model.fc

    act_space = ActivationSpace(classifier, last_layer, target_class, x.device)
    #act_space.load(f'utils/spurious_features/554_fireboat/act_space.npy')
    #act_space.load(f'utils/spurious_features/94_hummingbird/act_space.npy')
    #act_space.load(f'utils/spurious_features/563_fountain_pen/act_space.npy')

    #act_space.load(f'utils/spurious_features/2_white_shark/act_space.npy')
    act_space.load(f'utils/spurious_features/{target_class}/act_space.npy')
    #act_space.load(f'utils/spurious_features/269_timber_wolf/act_space.npy')


    act = activations_with_grad(x, classifier, last_layer, grad_enabled=True)

    weighted_act = act * act_space._last_layer.weight[target_class]

    pca_act = weighted_act @ act_space.eigenvectors
    pca_act = pca_act * torch.sum(act_space.eigenvectors, dim=0)

    return reduce(pca_act[:, component_idx], reduction)

###################################
def create_early_stopping_mask(out, y, conf_threshold, targeted):
    finished = False
    conf, pred = torch.max(torch.nn.functional.softmax(out, dim=1), 1)
    conf_mask = conf > conf_threshold
    if targeted:
        correct_mask = torch.eq(y, pred)
    else:
        correct_mask = (~torch.eq(y, pred))

    mask = 1. - (conf_mask & correct_mask).float()

    if sum(1.0 - mask) == out.shape[0]:
        finished = True

    mask = mask[(..., ) + (None, ) * 3]
    return finished, mask

def initialize_perturbation(x, eps, norm, x_init=None, noise_generator=None):
    if x_init is None:
        pert = torch.zeros_like(x)
    else:
        pert = x_init - x

    if noise_generator is not None:
        pert += noise_generator(x)

    pert = project_perturbation(pert, eps, norm)
    return pert