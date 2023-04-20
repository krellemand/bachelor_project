#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from torch.nn.functional import one_hot
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json

# imports from the paper
from models.classifier32 import classifier32
from utils.utils import strip_state_dict
from data.tinyimagenet import subsample_classes
from data.tinyimagenet import TinyImageNet
from data.open_set_splits.osr_splits import osr_splits

# imports from our modules
from our_modules.adv_tools import fgsm, fn_osr_fgsm, fp_osr_fgsm
from our_modules.adv_tools import save_grad_norms
from our_modules.eval_tools import get_osr_targets as _get_osr_targets


mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
image_size=64

# Paper
test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

# Min and max for image value after test transform is aplied
transform_range = (-2.4097, 2.7537)

splits = osr_splits['tinyimagenet']

# Load pretrained tiny-imagenet weights into the model for tinyimagenet.
def get_model_for_split(split_num, path_to_pretrained_weights_folder, device):
    state_dict = torch.load(path_to_pretrained_weights_folder + "tinyimagenet_" + str(split_num) + ".pth",
                        map_location=torch.device(device)) # <- Maybe this arg should only be there for cpu?
    state_dict = strip_state_dict(state_dict)

    # Instantiating a model for tinyimagenet and putting it into EVAL MODE!
    model = classifier32(num_classes=20)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model
 
def get_csr_dataloader_for_split(split_num, dataset, batch_size=100, shuffle=False):
    tiny_img_net_split = splits[split_num] 
    csr_dataset = subsample_classes(dataset, tiny_img_net_split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_csr_accuracy(model, dataloader, device):
    num_correct = 0
    for i, (input_batch, target_batch, _) in enumerate(dataloader):
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        logits = model(input_batch)
        preds = torch.argmax(logits, dim=-1)
        num_correct += torch.sum((torch.eq(preds, target_batch)))
        #print(f"{(i+1)/len(dataloader)*100}% done")

    return (num_correct/len(dataloader.dataset)).item()

# A "function" for evaluating average OSR accuracy across all 5 splits. The results match the ones found in table 5 of the paper (second to last row)
def get_avg_csr_acc_across_splits(path_to_pretrained_weights_folder, device, tin_val_root_dir, batch_size=100, shuffle=False):
    accs = []
    for split_num in tqdm(range(5)):
        dataset = TinyImageNet(root=tin_val_root_dir, transform=test_transform)
        model = get_model_for_split(split_num, path_to_pretrained_weights_folder, device=device)
        dataloader = get_csr_dataloader_for_split(split_num, dataset, batch_size=batch_size, shuffle=shuffle)
        acc = evaluate_csr_accuracy(model, dataloader, device=device)
        accs += acc,
    
    return sum(accs)/len(accs)

# ----------
#     OSR
# ----------

def get_osr_targets(csr_targets, split_num):
    return _get_osr_targets(csr_targets, splits[split_num])

def get_osr_dataloader_for_split(split_num, tin_val_root_dir, batch_size=100, shuffle=False): 
    dataset = TinyImageNet(root=tin_val_root_dir, transform=test_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate_osr_auroc(model, dataloader, split_num, device, logdir=None, adv_attack=lambda x, y, model: x):
    targets = []
    all_csr_targets = []
    mls_scores = []
    all_logits = []
    uq_idxs = []
    for i, (input_batch, target_batch, uq_idxs_for_batch) in enumerate(dataloader):
        all_csr_targets += target_batch.tolist()
        input_batch = input_batch.to(device)
        target_batch = get_osr_targets(target_batch, split_num)
        target_batch = target_batch.to(device)
        input_batch = adv_attack(input_batch, target_batch, model) # perform the adversarial attack
        logits = model(input_batch).detach().to('cpu')
        all_logits += logits,
        #logits = nn.Softmax(dim=-1)(logits) # Test for comparing with MSP
        mls_for_osr = -logits.max(dim=-1)[0] # (-) because low osr-score corresponds to known class
        targets += target_batch.tolist()
        mls_scores += mls_for_osr.tolist()
        uq_idxs += uq_idxs_for_batch.tolist()
        #print(f"{int((i+1)/len(dataloader)*100)}% done")

    if logdir is not None:
        os.makedirs(logdir, exist_ok = True)
        torch.save(torch.cat(all_logits), logdir + "logits_" + str(split_num) + ".pt")
        torch.save(torch.tensor(all_csr_targets), logdir + "csr_targets_" + str(split_num) + ".pt")
        torch.save(torch.tensor(uq_idxs), logdir + "index_" + str(split_num) + ".pt")

    return roc_auc_score(targets, mls_scores)

def get_avg_osr_auroc_across_splits(path_to_pretrained_weights_folder, tin_val_root_dir, device, logdir=None, batch_size=100, shuffle=False, 
                                    adv_attack=lambda x, y, model:x, number_of_splits=5):
    aurocs = []
    for split_num in tqdm(range(number_of_splits)):
        model = get_model_for_split(split_num, path_to_pretrained_weights_folder, device=device)
        dataloader = get_osr_dataloader_for_split(split_num, tin_val_root_dir, batch_size=batch_size, shuffle=shuffle)
        auroc = evaluate_osr_auroc(model, dataloader, split_num, device=device, logdir=logdir, adv_attack=adv_attack)
        aurocs += auroc,
    
    return sum(aurocs)/len(aurocs)

def save_grad_norms_across_splits(path_to_pretrained_weights_folder, tin_val_root_dir, logdir, loss_func, device, number_of_splits=5):
    for split_num in tqdm(range(number_of_splits)):
        model = get_model_for_split(split_num, path_to_pretrained_weights_folder, device=device)
        dataloader = get_osr_dataloader_for_split(split_num, tin_val_root_dir, batch_size=100, shuffle=False)
        save_grad_norms(loss_func, model, dataloader, logdir, device, split_num, ord = 1)

def save_informed_attack(logdir, path_to_fn_attack, path_to_fp_attack, split_num):
    fn_logits = torch.load(path_to_fn_attack + 'logits_' + str(split_num) + '.pt')
    fp_logits = torch.load(path_to_fp_attack + 'logits_' + str(split_num) + '.pt')
    csr_targets = torch.load(path_to_fn_attack + 'csr_targets_' + str(split_num) + '.pt')
    osr_targets = get_osr_targets(csr_targets, split_num)
    informed_mls = [fn_logits[i][None] if osr_targets[i] else fp_logits[i][None] for i in range(len(osr_targets))]
    os.makedirs(logdir, exist_ok = True)
    torch.save(torch.cat(informed_mls), logdir + "logits_" + str(split_num) + ".pt")
    torch.save(csr_targets, logdir + "csr_targets_" + str(split_num) + ".pt")
    attack_details = {'FN attack path' : path_to_fn_attack, 'FP attack path' : path_to_fp_attack}
    with open(logdir + 'attack_details.json', 'w') as file:
        json.dump(attack_details, file, indent=4)

def perturb_tin_image(eps, img, path_to_pretrained_weights_folder, device, split_num=0, attack=fn_osr_fgsm, **attack_kwargs):
    model = get_model_for_split(split_num=split_num,
                                path_to_pretrained_weights_folder=path_to_pretrained_weights_folder,
                                device=device)
    adv_img_and_step = [attack(model, img.to(device).detach()[None], ep, clip_range=transform_range, return_step=True, **attack_kwargs)
                       for ep in eps]
    adv_imgs, adv_steps = list(zip(*adv_img_and_step))
    return torch.cat(adv_imgs), torch.cat(adv_steps)
    
