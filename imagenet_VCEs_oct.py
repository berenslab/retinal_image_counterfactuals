import random

from torchvision.utils import save_image

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
import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from counterfactual_utils.load_trained_model import load_model
from tqdm import trange
from torchvision.transforms import functional as TF
import cv2
from PIL import Image
from time import sleep
from counterfactual_utils.datasets.oct import get_oct
from counterfactual_utils.train_types.helpers import create_attack_config, get_adversarial_attack
from matplotlib.colors import LinearSegmentedColormap

from counterfactual_utils.Evaluator import Evaluator


hps = get_config(get_arguments())

if not hps.verbose:
    blockPrint()


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


img_size = 256
num_imgs = hps.num_imgs
pixel_d_min = 5
conf_threshold = 0.01
out_dir = 'OCTVCEs'
dataset = 'imagenet'
imagenet_mode = 'examples'
bs = hps.batch_size * len(hps.device_ids)

torch.manual_seed(hps.seed)
random.seed(hps.seed)
np.random.seed(hps.seed)

in_labels = hps.data.class_labels #get_imagenet_labels(hps.data_folder)

# in_loader = get_oct(**{'split': 'test', 'augm_type': 'none', 'size': 256,
#                     'balanced': False,
#                      'data_folder': hps.data_folder, 'num_examples_each': None})

# in_dataset = in_loader.dataset

accepted_wnids = []


def _plot_counterfactuals(dir, original_imgs, orig_labels, segmentations, targets,
                          perturbed_imgs, perturbed_probabilities, original_probabilities,
                          l2_distances, l4_distances,
                          radii, class_labels, filenames=None, img_idcs=None, num_plot_imgs=hps.num_imgs,
                          method_descr=None):
    num_imgs = num_plot_imgs
    num_radii = len(radii)
    scale_factor = 4.0
    target_idx = 0
    all_values = []

    if img_idcs is None:
        img_idcs = torch.arange(num_imgs, dtype=torch.long)

    pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)

    # Two VCEs per starting image - we fix them to 1
    num_VCEs_per_image = 1
    for lin_idx in trange(int(len(img_idcs)/num_VCEs_per_image), desc=f'Image write'):

        # we fix only one radius
        radius_idx = 0
        lin_idx *= num_VCEs_per_image

        img_idx = img_idcs[lin_idx]
        in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]

        pred_original = in_probabilities.argmax()
        pred_value = in_probabilities.max()

        # num_rows = 1
        # num_cols = num_VCEs_per_image + 1
        # fig, ax = plt.subplots(num_rows, num_cols,
        #                        figsize=(scale_factor * num_cols, num_rows * 1.3 * scale_factor))
        img_label = int(orig_labels[img_idx])
        print('Image label: ', img_label)
        # title = f'GT: {class_labels[img_label]}' #, predicted: {class_labels[pred_original]},{pred_value:.2f}'

        img_segmentation = segmentations[img_idx]
        bin_segmentation = torch.sum(img_segmentation, dim=0) > 0.0
        img_segmentation[:, bin_segmentation] = 0.5
        mask_color = torch.zeros_like(img_segmentation)
        mask_color[1, :, :] = 1.0

        # plot original:
        # ax[0].axis('off')
        # ax[0].set_title(title)
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        original_image = img_original.clip(0, 1)
        original_image = original_image.numpy() * 255
        original_image = original_image.astype(np.uint8)
        original_image_final = Image.fromarray(original_image)
        original_image_final.save(os.path.join(dir, 'single_images', f'{img_idx}_original.png'), dpi=(300, 300))
        # ax[0].imshow(img_original, interpolation='lanczos')

        # save_image(original_imgs[img_idx, :].clip(0, 1),
        #            os.path.join(dir, 'single_images', f'{img_idx}_original.png'))
        for i in range(num_VCEs_per_image):

            img = torch.clamp(perturbed_imgs[img_idx+i, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
                              max=1.0)
            img_probabilities = perturbed_probabilities[img_idx+i, target_idx, radius_idx]

            img_target = targets[img_idx+i]
            target_original = in_probabilities[img_target]

            target_conf = img_probabilities[img_target]

            l2 = l2_distances[img_idx + i, target_idx, radius_idx][0]
            l4 = l4_distances[img_idx + i, target_idx, radius_idx][0]
            # ax[i+1].axis('off')
            # ax[i+1].imshow(img, interpolation='lanczos')

            values = {'img_idx': img_idx.numpy(), 'pred': class_labels[pred_original.numpy()],
                      'conf': pred_value.numpy(), 'target': class_labels[img_target],
                      'target_conf': target_conf.numpy(), 'i': target_original.numpy(),
                      'l2': l2.numpy(), 'l4':l4.numpy()}
            all_values.append(values)

            # title = f'{class_labels[img_target]}: {target_conf:.2f}, i:{target_original:.2f}, l2:{l2:.3f}, l4:{l4:.3f}'
            # ax[i+1].set_title(title)

            if method_descr is not None:
                target_label = class_labels[img_target]
                name_str = f'{img_idx}_{method_descr}_{target_label}'
            else:
                name_str = f'{img_idx}'

            """
            diff = (img_original.cpu() - img.cpu()).sum(2)
            diff = diff.numpy()
            im_max = np.percentile(diff, 99)
            im_min = np.min(diff)
            diff = ((diff - im_min) / (im_max - im_min)).clip(0, 1)
            diff = (diff - diff.min()) / (diff.max() - diff.min())
            values = diff.flatten()
            threshold = np.percentile(values, 98)
            diff[diff < threshold] = 0
            """

            diff = (img_original.cpu() - img.cpu()).sum(2)
            diff = diff.abs()
            diff = diff.numpy()
            min_diff_pixels = np.percentile(diff.flatten(), 99)#diff.min()
            max_diff_pixels = np.percentile(diff.flatten(), 99)#diff.max()
            # min_diff_pixels = -max(abs(min_diff_pixels), max_diff_pixels)
            # max_diff_pixels = -min_diff_pixels
            diff_scaled = (diff - min_diff_pixels) / (max_diff_pixels - min_diff_pixels)
            diff_scaled = np.clip(diff_scaled, 0, 1)
            # diff_scaled = diff_scaled.numpy()

            #Thresholding
            # values = diff_scaled.flatten()
            # threshold = np.percentile(values, 99)
            # diff_scaled[diff_scaled < threshold] = 0

            # fig, (ax1, ax2) = plt.subplots(1, 2)
            #
            # ax1.imshow(img)
            # ax1.set_xticks([])
            # ax1.set_yticks([])

            diff_fig, diff_ax = plt.subplots(figsize=(5, 5), dpi=300, layout='constrained')

            cmap = LinearSegmentedColormap.from_list('', ['white', 'red'])
            diff_ax = sns.heatmap(diff_scaled, cmap=cmap, cbar=False, square=True, ax=diff_ax)
            diff_ax.set_xticks([])
            diff_ax.set_yticks([])
            diff_fig.savefig(os.path.join(dir, 'single_images', f'{name_str}_diff.png'), dpi=300)

            # cm = plt.get_cmap('seismic')
            # colored_image = cm(diff_scaled.numpy())
            # ax[2 * i + 1].axis('off')
            # ax[2 * i + 1].imshow(colored_image, interpolation='lanczos')

            # save_image(diff_scaled, os.path.join(dir, 'single_images', f'{name_str}_diff.png'))

            # perturbed_img = perturbed_imgs[img_idx, target_idx, radius_idx].permute(1,2,0).clip(0,1)
            perturbed_image = img.numpy() * 255
            perturbed_image = perturbed_image.astype(np.uint8)
            perturbed_image_final = Image.fromarray(perturbed_image)
            perturbed_image_final.save(os.path.join(dir, 'single_images', f'{name_str}.png'), dpi=(300,300))
            # save_image(perturbed_imgs[img_idx, target_idx, radius_idx].clip(0, 1), os.path.join(dir, 'single_images', f'{name_str}.png'))

        # plt.tight_layout()
        # if filenames is not None:
        #     fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.png'))
        #     fig.savefig(os.path.join(dir, f'{filenames[lin_idx]}.pdf'))
        # else:
        #     fig.savefig(os.path.join(dir, f'{lin_idx}.png'))
        #     fig.savefig(os.path.join(dir, f'{lin_idx}.pdf'))
        #
        # plt.close(fig)
    df = pd.DataFrame(data=all_values)
    df.to_csv(os.path.join(dir, f'meta_{method_descr}.csv'))

plot = False
plot_top_imgs = True
"""
imgs = torch.zeros((num_imgs, 3, img_size, img_size))
segmentations = torch.zeros((num_imgs, 3, img_size, img_size))
targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
labels_tensor = torch.zeros(num_imgs, dtype=torch.long)
filenames = []

image_idx = 0
kernel = np.ones((5, 5), np.uint8)
# some_vces = {i: [1] for i in [2747, 26422, 23720, 22259, 3950, 2761, 27673, 29030, 8449, 29047, 12888, 29061]} # (ids for DR images)
# some_vces = {i: [1] for i in [0, 25848, 39, 115, 13441, 105, 730, 1629, 13777, 19, 64, 221, 26927, 1039, 25857, 7497, 26593, 690, 2546, 684, 13501, 15852]} 
# some_vces = {i: [1,2,3,4] for i in [24455, 22552, 6948, 28771, 29307]} # (ids for healthy images)
some_vces = {i: [0, 1] for i in [2747, 26422, 23720, 22259, 3950, 2761, 27673, 29030, 8449, 29047, 12888, 29061, 24455, 22552, 6948, 28771, 29307, 10594]} # (ids for both healthy and DR images)
selected_vces = list(some_vces.items())


if hps.world_size > 1:
    print('Splitting relevant classes')
    print(f'{hps.world_id} out of {hps.world_size}')
    splits = np.array_split(np.arange(len(selected_vces)), hps.world_size)
    print(f'Using clusters {splits[hps.world_id]} out of {len(targets_tensor)}')

for i, (img_idx, target_classes) in enumerate(selected_vces):
    if hps.world_size > 1 and i not in splits[hps.world_id]:
        pass
    else:
        in_image, label = in_dataset[img_idx]
        for i in range(len(target_classes)):
            targets_tensor[image_idx+i] = target_classes[i]
            labels_tensor[image_idx+i] = label
            imgs[image_idx+i] = in_image
        image_idx += len(target_classes)
        if image_idx >= num_imgs:
            break    
"""

filenames = os.listdir('samples_oct_val')
filenames = sorted(filenames)[:40]

num_imgs = len(filenames) * 3 

imgs = torch.zeros((num_imgs, 3, img_size, img_size))
segmentations = torch.zeros((num_imgs, 3, img_size, img_size))
targets_tensor = torch.zeros(num_imgs, dtype=torch.long)
labels_tensor = torch.zeros(num_imgs, dtype=torch.long)

files_list = filenames * 3 
files_list = files_list[::-1]
for img_name in files_list:#[f'{name}.png' for name in filenames]: #[15, 16, 17, 18][::-1]]: #[1, 2, 3, 4, 5, 6, 7][::-1]]:
    init_image_pil = Image.open(f'samples_oct_val/{img_name}').convert("RGB")
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

#     mask_pil = Image.open('input_example/original-mask.png').convert("RGB")
#     mask_pil = mask_pil.resize((img_size, img_size), Image.NEAREST)  # type: ignore
#     image_mask_pil_binarized = ((np.array(mask_pil) > 0.5) * 255).astype(np.uint8)
#     mask = TF.to_tensor(Image.fromarray(image_mask_pil_binarized))
#     print('test mask shape before', mask.shape)
#     mask = torch.tensor(cv2.dilate(mask.cpu().numpy(), kernel, iterations=8)).unsqueeze(0)
#     #mask = mask.unsqueeze(0) #mask[0, ...].unsqueeze(0).unsqueeze(0)#.to(device)
#     print('test mask shape after', mask.shape)

#     print('prepending mask', segmentations.shape, mask.shape)
#     segmentations = torch.cat([mask, segmentations], 0)
#     print('prepending mask', segmentations.shape)


print('prepending target', targets_tensor.shape)
targets_ = [1]*len(filenames)
targets_.extend([2]*len(filenames))
targets_.extend([3]*len(filenames))
# targets_.extend([3]*len(filenames))#[0, 0, 0, 2, 2, 948, 948, 948]#[649, 649] #[724, 959, 458, 991]#[938, 936]  #[0, 0] #[293, 293] #[701, 701] #[0, 0, 2, 2, 948, 948, 948]
targets_tensor = torch.cat([torch.tensor(targets_), targets_tensor], 0)
# labels_tensor = torch.cat([torch.tensor(targets_), labels_tensor], 0)
print('prepending target', targets_tensor.shape)


imgs = imgs[:num_imgs]
segmentations = segmentations[:num_imgs]
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
        # dir = f'{out_dir}/{imagenet_mode}/healthy_{norm}_{hps.l2_sim_lambda}_l1_{hps.l1_sim_lambda}_l{hps.lp_custom}_{hps.lp_custom_value}_classifier_{hps.classifier_type}_{hps.second_classifier_type}_{hps.third_classifier_type}_{hps.classifier_lambda}_reg_lpips_{hps.lpips_sim_lambda}_example_{hps.timestep_respacing}_steps_skip_{hps.skip_timesteps}_start_{hps.gen_type}_deg_{str(hps.deg_cone_projection)}_s_{str(hps.seed)}{"_bl" if hps.use_blended else ""}_wid_{hps.world_id}_{hps.world_size}_{hps.method}/'
        if hps.second_classifier_type == -1:
            dir = f'OCTVCEs/OCTModel_{hps.classifier_type}/'
        else:
            dir = f'OCTVCEs/OCTModel_{hps.classifier_type}_{hps.second_classifier_type}_from_normal/'

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
            print('batch segmentations before', segmentations.shape)
            batch_segmentations = segmentations[batch_start_idx:batch_end_idx, :]
            print('batch segmentations after', batch_segmentations.shape)
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
                data_dict['segmentations'] = segmentations[:batch_end_idx]
                data_dict['targets'] = targets_tensor[:batch_end_idx]
                data_dict['counterfactuals'] = out_imgs[:batch_end_idx]
                data_dict['out_probabilities'] = out_probabilities[:batch_end_idx]
                data_dict['in_probabilities'] = in_probabilities[:batch_end_idx]
                data_dict['radii'] = radii
                data_dict['l4_distances'] = out_l2_distances[:batch_end_idx]
                data_dict['l2_distances'] = out_l4_distances[:batch_end_idx]
                torch.save(data_dict, os.path.join(dir, f'{num_imgs}.pth'))
                _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, segmentations[:batch_end_idx],
                                      targets_tensor[:batch_end_idx],
                                      out_imgs[:batch_end_idx], out_probabilities[:batch_end_idx], in_probabilities[:batch_end_idx],
                                      out_l2_distances[:batch_end_idx], out_l4_distances[:batch_end_idx],
                                      radii, in_labels, filenames=None, num_plot_imgs=len(imgs[:batch_end_idx]),
                                      method_descr=method_descr)