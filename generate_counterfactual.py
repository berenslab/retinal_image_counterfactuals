import random

from torchvision import transforms

from blended_diffusion.optimization import DiffusionAttack
from blended_diffusion.optimization.arguments import get_arguments
from configs import get_config

from counterfactual_utils.functions import blockPrint
from counterfactual_utils.plot import _plot_counterfactuals
import torch
import numpy as np
import os
import pathlib
import matplotlib as mpl
mpl.use('Agg')
from tqdm import trange
from torchvision.transforms import functional as TF
from PIL import Image
from time import sleep
from counterfactual_utils.train_types.helpers import create_attack_config, get_adversarial_attack

from counterfactual_utils.Evaluator import Evaluator

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
    print(hps.save_diffs)

    if not hps.verbose:
        blockPrint()

    device, num_devices = set_device(hps)
    
    dataset = hps.data.dataset_for_scorenet
    #Image size is 256 as diffusion model is trained with image size 256. 
    img_size = 256
    if dataset.lower() == 'eyepacs':
        out_dir = 'FundusCounterfactuals'
    elif dataset.lower() == 'oct':
        out_dir = 'OCTCounterfactuals'
    else:
        raise ValueError(f"The dataset {dataset.lower()} is not supported!")
    bs = hps.batch_size * len(hps.device_ids)

    #Setting seed. Note that results can change if order of images are changed. 
    torch.manual_seed(hps.seed)
    random.seed(hps.seed)
    np.random.seed(hps.seed)

    #Class labels are specified in blended_diffusion/configs/fundus_binary.yml or blended_diffusion/configs/fundus_5class.yml
    in_labels = hps.data.class_labels 
    num_classes = len(in_labels)
    
    #Reading and storing image samples in tensors for processing 
    filenames = os.listdir(hps.image_dir)
    filenames = sorted(filenames)
    print(filenames)

    # Counterfactuals are generated to all classes 
    files_list = filenames * num_classes 
    files_list = files_list[::-1]
    num_imgs = len(files_list) 

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

        if dataset.lower() == 'eyepacs' and hps.method.lower() == 'svces':
            mask_pil = Image.open(f'fundus_masks/{img_name}')
            mask_pil = mask_pil.resize((img_size, img_size), Image.LANCZOS)  # type: ignore
            mask_image = TF.to_tensor(mask_pil).to(device).unsqueeze(0)
            print('prepending img', masks.shape)
            masks = torch.cat([mask_image, masks.to(device)], 0)
            print('prepending img', masks.shape)

    #Defining target classes here, counterfactuals are generated to all classes 
    print('prepending target', targets_tensor.shape)
    targets_ = [0]*len(filenames)
    for i in np.arange(num_classes-1):
        targets_.extend([i+1]*len(filenames))
    targets_tensor = torch.cat([torch.tensor(targets_), targets_tensor], 0)
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


        attack_config = create_attack_config(eps=radii[0], steps=steps, stepsize=stepsize, norm=norm, 
                                             momentum=0.9, pgd=attack_type)

        
        if attack_type == 'diffusion':
            img_dimensions = (3, 256, 256)
        else:
            img_dimensions = imgs.shape[1:]
        num_targets = 1
        num_radii = len(radii)
        num_imgs = len(imgs)

        with torch.no_grad():

            model_bs = bs
            sub_dir = hps.config.split('.')[0]
            dir = f'{out_dir}/{sub_dir}'
            # if hps.second_classifier_type == -1:
            #     dir = f'{out_dir}/Retina_{hps.method}_{hps.classifier_type}'
            # else:
            #     dir = f'{out_dir}/Retina_{hps.method}_{hps.classifier_type}_{hps.second_classifier_type}'
            
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
  
                    resize_transform = transforms.Resize(256)
                    batch_adv_samples_resized_i = resize_transform(batch_adv_samples_i)
                    
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
                    _plot_counterfactuals(dir, imgs[:batch_end_idx], labels_tensor, 
                                          masks[:batch_end_idx],
                                          targets_tensor[:batch_end_idx],out_imgs[:batch_end_idx],
                                          out_probabilities[:batch_end_idx],
                                          in_probabilities[:batch_end_idx],
                                          out_l2_distances[:batch_end_idx],
                                          out_l4_distances[:batch_end_idx],radii, in_labels, 
                                          filenames=None,num_imgs=len(imgs[:batch_end_idx]),
                                          method_descr=method_descr, save_diffs=hps.save_diffs)
