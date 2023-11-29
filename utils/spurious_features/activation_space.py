import torch
import numpy as np
from captum import attr as cattr

from tqdm import tqdm
from .data import imagenet_label2class
from .pca import compute_pca

class  ActivationSpace():

    def __init__(self, model, last_layer, target_idx, device, train_loader=None, weighted=True):
        self.target = target_idx
        self.target_name = imagenet_label2class[self.target] 
        
        self.device=device        
        self.model = model.to(self.device)

        self._weighted = weighted
        self._last_layer = last_layer 
        self._layer_activations = cattr.LayerActivation(model, last_layer)
        self._train_loader = train_loader
        self._last_weights = last_layer.weight.data[self.target]
        self.indices_high_conf = []
        
    def fit(self, remove_mean=False, conf_threshold=0, eigvec_scale=True):
        if self._train_loader is None:
            print('Missing train_loader. Use AcitvationSpace.load() or init with train_loader.')
        self._remove_mean = remove_mean
        # get activations
        print('Computing activations\n')
        self.activations_train = self._compute_training_activations(conf_threshold)

        # compute pca
        print('Computing principal components\n')
        _, self.eigenvectors, pca_mean = compute_pca(self.activations_train.T.cpu().numpy())
        self.eigenvectors = torch.tensor(self.eigenvectors, dtype=torch.float32, device=self.device)
        self.pca_mean = pca_mean.T
        
        # transform training points
        self.pca_train = self._pca_transform(self.activations_train, eigvec_scale=eigvec_scale)
        
    def transform(self, images, eigvec_scale=True):
        act = self._compute_activations(images)
        return self._pca_transform(act, eigvec_scale=eigvec_scale)

    def maximizing_train_points(self, pca_dim, k=5, return_indices=False, order='alpha'):
        if order == 'alpha':
            return self._maximizing_train_points_alpha(pca_dim, k=k, return_indices=return_indices)
        elif order == 'alpha_conf':
            return self._maximizing_train_points_alpha_conf(pca_dim, k=k, return_indices=return_indices)
    
    def _maximizing_train_points_alpha_conf(self, pca_dim, k=5, return_indices=False):
        representation = self.pca_train
        logits = self.activations_train@self._last_layer.weight.T + self._last_layer.bias
        self.objective = representation[:, pca_dim] - torch.log(torch.sum(torch.exp(logits))) 
        sorted_idcs = torch.argsort(self.objective, descending=True)
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self._train_loader.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        return max_images

    def _maximizing_train_points_alpha(self, pca_dim, k=5, return_indices=False):
        representation = self.pca_train
        self.objective = representation[:, pca_dim]
        sorted_idcs = torch.argsort(representation[:, pca_dim], descending=True)
        max_idcs = sorted_idcs[:k]
        max_images = []
        for idx in max_idcs:
            max_images.append(self._train_loader.dataset[idx][0])
        if return_indices:
            return max_images, max_idcs
        return max_images

    def _compute_training_activations(self, conf_threshold=0):
        act_train = None
        self.confidences_train = None
        for batch_idx, (imgs, _) in enumerate(self._train_loader):
            act = self._compute_activations(imgs)

            out = self.model(imgs.to(self.device))
            prob = torch.softmax(out, dim=1).cpu().detach().numpy()
            pred = torch.max(out, dim=1)[1].cpu().detach().numpy()

            mask_high_conf = prob[:, self.target] > conf_threshold
            if np.sum(mask_high_conf) == 0:
                continue

            act_high_conf = act[mask_high_conf]
            prob_high_conf = prob[mask_high_conf]
            pred_high_conf = pred[mask_high_conf]

            start_idx = batch_idx * self._train_loader.batch_size
            indices = np.arange(start_idx, start_idx + imgs.shape[0])
            self.indices_high_conf.extend(indices[mask_high_conf])

            if act_train is None:
                act_train = act_high_conf
                self.confidences_train = prob_high_conf
                self.predictions_train = pred_high_conf
            else:
                act_train = torch.cat((act_train, act_high_conf))
                self.confidences_train = np.concatenate((self.confidences_train, prob_high_conf))
                self.predictions_train = np.concatenate((self.predictions_train, pred_high_conf))
        return act_train

    def _compute_activations(self, images):
        if len(images.shape) == 3:
            images = images[None, :]
        act = self._layer_activations.attribute(images.to(self.device), attribute_to_layer_input=True)
        act = act.squeeze()            
        if self._weighted:
            return act * self._last_weights
        return act

    def _pca_transform(self, activations, eigvec_scale=True):
        if self._remove_mean:
            activations -= torch.tensor(self.pca_mean).to(activations.device)
        act_pca = activations@self.eigenvectors 
        if eigvec_scale:
            #if self._only_torch:
            act_pca = act_pca * torch.sum(self.eigenvectors, dim=0)
            #else:
            #    act_pca = act_pca * np.sum(self.eigenvectors, axis=0)
        return act_pca

    def save(self, fpath='actspace.npy'):
        save_dict = {
            'act_train':self.activations_train,
            'eigenvectors':self.eigenvectors,
            'alpha_train':self.pca_train,
            'pca_mean':self.pca_mean,
            'remove_mean':self._remove_mean
        }
        np.save(fpath, save_dict)

    def load(self, fpath='actspace.npy'):
        load_dict = np.load(fpath, allow_pickle=True)[()]
        self.activations_train = load_dict['act_train']
        self.eigenvectors = load_dict['eigenvectors'].to(self.device)
        self.pca_train = load_dict['alpha_train']
        self.pca_mean = load_dict['pca_mean']
        self._remove_mean = load_dict['remove_mean']