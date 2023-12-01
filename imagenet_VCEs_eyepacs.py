import random

from torchvision.utils import save_image
from torchvision import transforms

from blended_diffusion.optimization import DiffusionAttack
from blended_diffusion.optimization.arguments import get_arguments
from configs import get_config
from counterfactual_utils.datasets.paths import get_imagenet_path
from counterfactual_utils.datasets.imagenet import get_imagenet_labels
import counterfactual_utils.datasets as dl
from counterfactual_utils.functions import blockPrint
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib as mpl
import seaborn as sns
mpl.use('Agg')
import matplotlib.pyplot as plt
from counterfactual_utils.load_trained_model import load_model
from tqdm import trange
from torchvision.transforms import functional as TF
import cv2
from PIL import Image
from time import sleep
from counterfactual_utils.train_types.helpers import create_attack_config, get_adversarial_attack
from matplotlib.colors import LinearSegmentedColormap
from skimage import feature, transform

from counterfactual_utils.Evaluator import Evaluator

def get_difference_image(img_original, img_vce, dilation=0.5):
    img_original = img_original.numpy()
    img_vce = img_vce.numpy()
    
    grayscale_original = np.dot(img_original[...,:3], [0.2989, 0.5870, 0.1140])
    grayscale_vce = np.dot(img_vce[...,:3], [0.2989, 0.5870, 0.1140])

    grayscale_diff = grayscale_original - grayscale_vce
    grayscale_diff = np.abs(grayscale_diff)
    min_diff = np.min(grayscale_diff)
    max_diff = np.max(grayscale_diff)
    min_diff = -max(abs(min_diff), max_diff)
    max_diff = -min_diff
    diff_scaled = (grayscale_diff - min_diff) / (max_diff - min_diff)
    diff_scaled = np.clip(diff_scaled, 0, 1) * 255

    original_greyscale = img_original if len(img_original.shape) == 2 else np.mean(img_original, axis=-1)
    in_image_upscaled = transform.rescale(original_greyscale, dilation, mode='constant',
                                          multichannel=False, anti_aliasing=True)
    edges = feature.canny(in_image_upscaled).astype(float)
    edges[edges < 0.5] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan
    overlay = edges

    return diff_scaled, overlay

def _plot_counterfactuals(dir, original_imgs, orig_labels, masks, targets,
                          perturbed_imgs, perturbed_probabilities, original_probabilities,
                          l2_distances, l4_distances,
                          radii, class_labels, filenames=None, img_idcs=None, num_plot_imgs=None,
                          method_descr=None):
    num_imgs = num_plot_imgs
    num_radii = len(radii)
    target_idx = 0
    all_values = []

    if img_idcs is None:
        img_idcs = torch.arange(num_imgs, dtype=torch.long)

    pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)
    
    ### Currently we only generate one counterfactual per image, due to stochastic nature of the method, it is possible to generate multiple counterfactuals for the same image
    num_VCEs_per_image = 1 
    for lin_idx in trange(int(len(img_idcs)/num_VCEs_per_image), desc=f'Image write'):

        # we fix only one radius
        radius_idx = 0
        lin_idx *= num_VCEs_per_image

        img_idx = img_idcs[lin_idx]
        in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]

        pred_original = in_probabilities.argmax()
        pred_value = in_probabilities.max()

        img_label = int(orig_labels[img_idx])
        print('Image label: ', img_label)
        
        ###Saving original image
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        original_image = img_original.clip(0, 1)
        original_image = original_image.numpy() * 255
        original_image = original_image.astype(np.uint8)
        original_image_final = Image.fromarray(original_image)
        original_image_final.save(os.path.join(dir, 'single_images', f'{img_idx}_original.png'), dpi=(300, 300))

        mask_original = masks[img_idx, :].permute(1,2,0).cpu().detach()
        print(mask_original.max())

        for i in range(num_VCEs_per_image):
            img = torch.clamp(perturbed_imgs[img_idx+i, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
                              max=1.0)
            img_probabilities = perturbed_probabilities[img_idx+i, target_idx, radius_idx]

            img_target = targets[img_idx+i]
            target_original = in_probabilities[img_target]

            target_conf = img_probabilities[img_target]

            l2 = l2_distances[img_idx + i, target_idx, radius_idx][0]
            l4 = l4_distances[img_idx + i, target_idx, radius_idx][0]

            values = {'img_idx': img_idx.numpy(), 'pred': class_labels[pred_original.numpy()],
                      'conf': pred_value.numpy(), 'target': class_labels[img_target],
                      'target_conf': target_conf.numpy(), 'i': target_original.numpy(),
                      'l2': l2.numpy(), 'l4':l4.numpy()}
            all_values.append(values)

            if method_descr is not None:
                target_label = class_labels[img_target]
                name_str = f'{img_idx}_{method_descr}_{target_label}'
            else:
                name_str = f'{img_idx}'

            ### Creating and saving difference map
            diff_scaled, overlay = get_difference_image(img_original.cpu(), img.cpu())

            dx, dy = 0.05, 0.05
            xx = np.arange(0.0, diff_scaled.shape[1], dx)
            yy = np.arange(0.0, diff_scaled.shape[0], dy)
            xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
            extent = xmin, xmax, ymin, ymax
            cmap = LinearSegmentedColormap.from_list('', ['white', 'red'])
            cmap_original = plt.get_cmap('Greys_r')
            cmap_original.set_bad(alpha=0)

            diff_fig, diff_ax = plt.subplots(figsize=(5, 5), dpi=300, layout='constrained')
            diff_ax.set_xticks([])
            diff_ax.set_yticks([])
            diff_ax.imshow(diff_scaled, extent=extent, interpolation='none', cmap=cmap)
            if overlay is not None:
                diff_ax.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_original, alpha=0.25)
            #diff_fig.savefig(os.path.join(dir, 'single_images', f'{name_str}_diff.png'), dpi=300)

            ### Saving counterfactual image
            if 'svce' in method_descr:
                img = img * mask_original 
            perturbed_image = img.numpy() * 255
            perturbed_image = perturbed_image.astype(np.uint8)
            perturbed_image_final = Image.fromarray(perturbed_image)
            perturbed_image_final.save(os.path.join(dir, 'single_images', f'{name_str}.png'), dpi=(300,300))
            

    #Saving predictions, probabilities of predicted class for both original and counterfactual 
    df = pd.DataFrame(data=all_values)
    df.to_csv(os.path.join(dir, f'meta_{method_descr}.csv'))

plot = False
plot_top_imgs = True

def set_device(hps):
    if len(hps.gpu)==0:
        device = torch.device('cpu')
        print('Warning! Computing on CPU')
        num_devices = 1
    elif len(hps.gpu)==1:
        hps.device_ids = [int(hps.gpu[0])]
        device = torch.device('cuda:' + str(hps.gpu[0]))
        num_devices = 1
    else:
        hps.device_ids = [int(i) for i in hps.gpu]
        device = torch.device('cuda:' + str(min(hps.device_ids)))
        num_devices = len(hps.device_ids)
    hps.device = device
    return device, num_devices 

if __name__ == '__main__':
    hps = get_config(get_arguments())

    if not hps.verbose:
        blockPrint()

    device, num_devices = set_device(hps)

    #Image size is 256 as diffusion model is trained with image size 256. 
    img_size = 256
    out_dir = 'EyePacsVCEs'
    bs = hps.batch_size * len(hps.device_ids)

    #Setting seed. Note that results can change if order of images are changed. 
    torch.manual_seed(hps.seed)
    random.seed(hps.seed)
    np.random.seed(hps.seed)

    #Class labels are specified in blended_diffusion/configs/fundus_binary.yml or blended_diffusion/configs/fundus_5class.yml
    in_labels = hps.data.class_labels 
    
    #Reading and storing image samples in tensors for processing 
    filenames = os.listdir(hps.image_dir)
    filenames = sorted(filenames)[:2]
    print(filenames)

    files_list = filenames #* 2
    # files_list = files_list[80:]
    files_list = files_list[::-1]
    num_imgs = len(files_list) #* 2

    imgs = torch.zeros((num_imgs, 3, img_size, img_size))
    masks = torch.zeros((num_imgs, 1, img_size, img_size))
    targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
    labels_tensor = torch.zeros(num_imgs, dtype=torch.long)

    for img_name in files_list:
        init_image_pil = Image.open(f'{hps.image_dir}/{img_name}').convert("RGB")
        init_image_pil = init_image_pil.resize((img_size, img_size), Image.LANCZOS)  # type: ignore
        init_image = TF.to_tensor(init_image_pil).to(device).unsqueeze(0)
        print('prepending img', imgs.shape)
        imgs = torch.cat([init_image, imgs.to(device)], 0)
        print('prepending img', imgs.shape)

        label = img_name.split('_')[0]
        print(label)
        label = int(int(label) > 1)
        print('prepending labels_tensor', labels_tensor.shape)
        labels_tensor = torch.cat([torch.tensor(label).unsqueeze(dim=0), labels_tensor], 0)
        print('prepending labels_tensor', labels_tensor.shape)
        
        if hps.method == 'svces':
            mask_pil = Image.open(f'masks_eyepacs/{img_name}')
            mask_pil = mask_pil.resize((img_size, img_size), Image.LANCZOS)  # type: ignore
            mask_image = TF.to_tensor(mask_pil).to(device).unsqueeze(0)
            print('prepending img', masks.shape)
            masks = torch.cat([mask_image, masks.to(device)], 0)
            print('prepending img', masks.shape)


    print('prepending target', targets_tensor.shape)
    targets_ = [1]*len(filenames)
    #targets_.extend([1]*len(filenames))
    # targets_.extend([2]*len(filenames))
    # targets_.extend([3]*len(filenames))
    # targets_.extend([4]*len(filenames))
    targets_tensor = torch.cat([torch.tensor(targets_), targets_tensor], 0)
    # labels_tensor = torch.cat([torch.tensor(targets_), labels_tensor], 0)
    print('prepending target', targets_tensor.shape)

    imgs = imgs[:num_imgs]
    targets_tensor = targets_tensor[:num_imgs]

    use_diffusion = False
    for method in [hps.method]:
        if method.lower() == 'svces':
            radii = np.array([0.3])
            attack_type = 'afw'
            norm = 'L4'
            stepsize = None
            steps = 75
            method_descr = f'{method.lower()}_{norm}_{radii[0]}'
        elif method.lower() == 'dvces':
            attack_type = 'diffusion'
            radii = np.array([0.])
            norm = 'L2'
            steps = 150
            stepsize = None
            use_diffusion=True
            method_descr = f'{method.lower()}_{norm}_{hps.lp_custom}_{hps.lp_custom_value}_{hps.skip_timesteps}'
        else:
            raise NotImplementedError()


        attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, momentum=0.9,
                                         pgd=attack_type)

        num_classes = len(in_labels)
        if attack_type == 'diffusion':
            img_dimensions = (3, 256, 256)
        else:
            img_dimensions = imgs.shape[1:]
        num_targets = 1
        num_radii = len(radii)
        num_imgs = len(imgs)

        with torch.no_grad():

            model_bs = bs
            if hps.second_classifier_type == -1:
                dir = f'EyePacsVCEs/EyePacsModel_{hps.classifier_type}_{hps.method}'
            else:
                dir = f'EyePacsVCEs/EyePacsModel_{hps.classifier_type}_{hps.second_classifier_type}_{hps.method}'
            pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

            out_imgs = torch.zeros((num_imgs, num_targets, num_radii) + img_dimensions)
            out_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
            in_probabilities = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
            out_l4_distances = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
            out_l2_distances = torch.zeros((num_imgs, num_targets, num_radii, num_classes))
            model_original_probabilities = torch.zeros((num_imgs, num_classes))

            n_batches = int(np.ceil(num_imgs / model_bs))


            if use_diffusion or attack_config['pgd'] == 'afw':
                if use_diffusion:
                    att = DiffusionAttack(hps)
                else:
                    loss = 'log_conf'
                    print('using loss', loss)
                    model = None
                    att = get_adversarial_attack(attack_config, model, loss, num_classes,
                                             args=hps, Evaluator=Evaluator)
                if att.args.second_classifier_type != -1:
                    print('setting model to second classifier')
                    model = att.second_classifier
                else:
                    model = att.classifier
            else:
                model = None

            for batch_idx in trange(n_batches, desc=f'Batches progress'):
                sleep(0.1)
                batch_start_idx = batch_idx * model_bs
                batch_end_idx = min(num_imgs, (batch_idx + 1) * model_bs)

                batch_data = imgs[batch_start_idx:batch_end_idx, :]
                batch_targets = targets_tensor[batch_start_idx:batch_end_idx]
                target_idx = 0

                orig_out = model(batch_data)
                with torch.no_grad():
                    orig_confidences = torch.softmax(orig_out, dim=1)
                    model_original_probabilities[batch_start_idx:batch_end_idx, :] = orig_confidences.detach().cpu()

                for radius_idx in range(len(radii)):
                    batch_data = batch_data.to(device)
                    batch_targets = batch_targets.to(device)


                    if not use_diffusion:
                        att.eps = radii[radius_idx]

                    if use_diffusion:
                        batch_adv_samples_i = att.perturb(batch_data,
                                                            batch_targets, dir)[
                        0].detach()
                    else:
                        if attack_config['pgd'] == 'afw':

                            batch_adv_samples_i = att.perturb(batch_data,
                                                                batch_targets,
                                                                targeted=True).detach()



                        else:
                            batch_adv_samples_i = att.perturb(batch_data,
                                                                batch_targets,
                                                                best_loss=True)[0].detach()
                    out_imgs[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                :] = batch_adv_samples_i.cpu().detach()

                    print(batch_data.shape)
                    print(batch_adv_samples_i.shape)

                    """
                    transform = transforms.Resize(224)
                    batch_adv_samples_resized_i = transform(batch_adv_samples_i)
                    """

                    diffs = (batch_data - batch_adv_samples_i).abs()
                    diffs_squared_i = diffs.view(diffs.shape[0], -1) ** 2
                    l2_distances_i = diffs_squared_i.sum(1, keepdim=True) ** 0.5

                    diffs_pow4_i = diffs.view(diffs.shape[0], -1) ** 4
                    l4_distances_i = diffs_pow4_i.sum(1, keepdim=True) ** 0.25

                    print(diffs.shape)
                    print(l4_distances_i.shape)
                    batch_model_out_i = model(batch_adv_samples_i)
                    batch_model_in_i = model(batch_data)
                    batch_probs_i = torch.softmax(batch_model_out_i, dim=1)
                    batch_probs_in_i = torch.softmax(batch_model_in_i, dim=1)

                    out_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                    :] = batch_probs_i.cpu().detach()
                    in_probabilities[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                    :] = batch_probs_in_i.cpu().detach()
                    out_l2_distances[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                    :] = l2_distances_i.cpu().detach()
                    out_l4_distances[batch_start_idx:batch_end_idx, target_idx, radius_idx,
                    :] = l4_distances_i.cpu().detach()

                if (batch_idx + 1) % hps.plot_freq == 0 or batch_idx == n_batches-1:
                    data_dict = {}

                    data_dict['gt_imgs'] = imgs[:batch_end_idx]
                    data_dict['gt_labels'] = labels_tensor[:batch_end_idx]
                    data_dict['targets'] = targets_tensor[:batch_end_idx]
                    data_dict['counterfactuals'] = out_imgs[:batch_end_idx]
                    data_dict['out_probabilities'] = out_probabilities[:batch_end_idx]
                    data_dict['in_probabilities'] = in_probabilities[:batch_end_idx]
                    data_dict['radii'] = radii
                    data_dict['l4_distances'] = out_l2_distances[:batch_end_idx]
                    data_dict['l2_distances'] = out_l4_distances[:batch_end_idx]
                    torch.save(data_dict, os.path.join(dir, f'{num_imgs}.pth'))
                    _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, masks[:batch_end_idx], 
                                      targets_tensor[:batch_end_idx],out_imgs[:batch_end_idx],
                                      out_probabilities[:batch_end_idx],in_probabilities[:batch_end_idx],
                                      out_l2_distances[:batch_end_idx], out_l4_distances[:batch_end_idx],
                                      radii, in_labels, filenames=None,num_plot_imgs=len(imgs[:batch_end_idx]),
                                      method_descr=method_descr)
