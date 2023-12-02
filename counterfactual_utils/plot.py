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
import torch



plot = False
plot_top_imgs = True

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

def save_image(image, dir, img_name):
    image = image.numpy() * 255
    image = image.astype(np.uint8)
    image_final = Image.fromarray(image)
    image_final.save(os.path.join(dir, 'single_images', f'{img_name}.png'), dpi=(300, 300))
    
    
def save_diff_image(diff_scaled, overlay, dir, name_str):
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
    diff_fig.savefig(os.path.join(dir, 'single_images', f'{name_str}_diff.png'), dpi=300)

def _plot_counterfactuals(dir, original_imgs, orig_labels, masks, targets,
                          perturbed_imgs, perturbed_probabilities, original_probabilities,
                          l2_distances, l4_distances, radii, class_labels, 
                          filenames=None, img_idcs=None, num_imgs=None,
                          method_descr=None, save_diffs=False):
    num_radii = len(radii)
    target_idx = 0
    all_values = []

    if img_idcs is None:
        img_idcs = torch.arange(num_imgs, dtype=torch.long)

    pathlib.Path(dir+'/single_images').mkdir(parents=True, exist_ok=True)
    
    ### Currently we only generate one counterfactual per image, due to stochastic nature of the method, it is possible to generate multiple counterfactuals for the same image
    num_VCEs_per_image = 1 
    for lin_idx in trange(int(len(img_idcs)/num_VCEs_per_image), desc=f'Image write'):

        radius_idx = 0
        lin_idx *= num_VCEs_per_image

        img_idx = img_idcs[lin_idx]
        in_probabilities = original_probabilities[img_idx, target_idx, radius_idx]

        pred_original = in_probabilities.argmax()
        pred_value = in_probabilities.max()

        img_label = int(orig_labels[img_idx])
        print('Image label: ', img_label)
        
        mask_original = masks[img_idx, :].permute(1,2,0).cpu().detach()
        print(mask_original.max())
        
        ###Saving original image
        img_original = original_imgs[img_idx, :].permute(1, 2, 0).cpu().detach()
        original_image = img_original.clip(0, 1)
        save_image(original_image, dir, f'{img_idx}_original')

        for i in range(num_VCEs_per_image):
            img = torch.clamp(perturbed_imgs[img_idx+i, target_idx, radius_idx].permute(1, 2, 0), min=0.0,
                              max=1.0)
            img_probabilities = perturbed_probabilities[img_idx+i, target_idx, radius_idx]

            #Predictions, probabilities and distances between original and counterfactual images. 
            img_target = targets[img_idx+i]
            target_original = in_probabilities[img_target]
            target_conf = img_probabilities[img_target]
            l2 = l2_distances[img_idx + i, target_idx, radius_idx][0]
            l4 = l4_distances[img_idx + i, target_idx, radius_idx][0]

            #For storing predictions and probabilities of each image and it's counterfactual
            values = {'img_idx': img_idx.numpy(), 'pred': class_labels[pred_original.numpy()],
                      'pred_conf': pred_value.numpy(), 'target': class_labels[img_target],
                      'final_target_conf': target_conf.numpy(), 
                      'initial_target_conf': target_original.numpy(),
                      'l2': l2.numpy(), 'l4':l4.numpy()}
            all_values.append(values)

            if method_descr is not None:
                target_label = class_labels[img_target]
                name_str = f'{img_idx}_{method_descr}_{target_label}'
            else:
                name_str = f'{img_idx}'

            ### Creating and saving difference map
            if save_diffs:
                diff_scaled, overlay = get_difference_image(img_original.cpu(), img.cpu())
                save_diff_image(diff_scaled, overlay, dir, name_str)
            
            ### Saving counterfactual image
            ### TO DO: uncomment after adding masks folder
            # if 'svce' in method_descr and 'eyepacs' in dir.lower():
            #     img = img * mask_original 
            save_image(img, dir, name_str)
            
    #Saving predictions, probabilities of predicted class for both original and counterfactual 
    df = pd.DataFrame(data=all_values)
    df.to_csv(os.path.join(dir, f'meta_{method_descr}.csv'))