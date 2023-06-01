from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score
import torch
from data.open_set_splits.osr_splits import osr_splits
import numpy as np
import os
import random

def balance_binary(lst, bool_func, seed=777):
    random.seed(seed)
    lst = list(lst)
    false = [x for x in lst if not bool_func(x)]
    true = [x for x in lst if bool_func(x)]
    if len(false) > len(true):
        false = random.sample(false, len(true))
    if len(true) > len(false):
        true = random.sample(true, len(false))
    return true + false

def osr_roc_stats(open_set_scores, open_set_labels):
    """
    Compute performance metrics based on the logits of
    an OSR model obtained by evaluating the model on a 
    test set.
    """
    open_set_scores = -open_set_scores
    fprs, tprs, thresholds = roc_curve(open_set_labels, open_set_scores, drop_intermediate=False)
    auroc = roc_auc_score(open_set_labels, open_set_scores)
    return (fprs, tprs, thresholds), auroc

def mls_osr_score(logits):
    return torch.amax(logits, dim=-1) # In theory high osr score corresponds to known class. We however want novel to correspond to label 1 and thus the minus sign.

def get_osr_targets(csr_targets, split_targets):
    return (~(sum(csr_targets == i for i in split_targets).bool()))

def eval_osr(osr_scores, osr_targets, balance=True, return_avg_score=False):
    # print((sum(osr_targets)/len(osr_targets)).data) # Balance ratio before
    if balance:
        score_label_zipped = balance_binary(zip(osr_scores.tolist(), osr_targets.tolist()), lambda x: bool(x[1]))
        osr_scores, osr_targets = (torch.tensor([mls for mls, _ in score_label_zipped ]), 
                                       [osr_target for _, osr_target in score_label_zipped])
    # print(sum(osr_targets)/len(osr_targets)) # Balance ratio after
    if return_avg_score:
        return osr_roc_stats(osr_scores, osr_targets), torch.median(osr_scores)
    return osr_roc_stats(osr_scores, osr_targets)

def eval_osr_quantiles(osr_scores, osr_targets, balance=True, return_avg_score=False):
    # print((sum(osr_targets)/len(osr_targets)).data) # Balance ratio before
    if balance:
        score_label_zipped = balance_binary(zip(osr_scores.tolist(), osr_targets.tolist()), lambda x: bool(x[1]))
        osr_scores, osr_targets = (torch.tensor([mls for mls, _ in score_label_zipped ]), 
                                       [osr_target for _, osr_target in score_label_zipped])
    # print(sum(osr_targets)/len(osr_targets)) # Balance ratio after
    if return_avg_score:
        return osr_roc_stats(osr_scores, osr_targets), np.quantile(osr_scores,[0.25, 0.5, 0.75])
    return osr_roc_stats(osr_scores, osr_targets)

def load_and_eval_mls_osr(logit_file_path, csr_targets_file_path, split_num, dataset_name='tinyimagenet', balance=True, return_avg_mls=False, return_quantiles=False, msp=False):
    assert int(logit_file_path[-4]) == split_num, "The split_num does not correspond to the split num of the file name"
    split = osr_splits[dataset_name][split_num]
    logits = torch.load(logit_file_path)
    csr_targets = torch.load(csr_targets_file_path)
    mls_scores = mls_osr_score(logits)
    if msp:
        mls_scores = torch.amax(torch.softmax(logits, dim = 1), dim=-1)
        print('allert msp')
    open_set_labels = get_osr_targets(csr_targets, split)
    if return_quantiles:
        return eval_osr_quantiles(mls_scores, open_set_labels, balance=balance, return_avg_score=return_avg_mls)
    return eval_osr(mls_scores, open_set_labels, balance=balance, return_avg_score=return_avg_mls)

# def load_and_eval_mls_osr(logit_file_path, csr_targets_file_path, split_num, dataset_name='tinyimagenet', balance=False, return_avg_mls=False):
#     assert int(logit_file_path[-4]) == split_num, "The split_num does not correspond to the split num of the file name"
#     split = osr_splits[dataset_name][split_num]
#     logits = torch.load(logit_file_path)
#     csr_targets = torch.load(csr_targets_file_path)
#     mls_scores = mls_osr_score(logits)
#     open_set_labels = get_osr_targets(csr_targets, split)
#     # print((sum(open_set_labels)/len(open_set_labels)).data) # Balance ratio before
#     if balance:
#         mls_oslabel = balance_binary(zip(mls_scores.tolist(), open_set_labels.tolist()), lambda x: bool(x[1]))
#         mls_scores, open_set_labels = (torch.tensor([mls for mls, _ in mls_oslabel ]), 
#                                        [oslabel for _, oslabel in mls_oslabel])
#     # print(sum(open_set_labels)/len(open_set_labels)) # Balance ratio after
#     if return_avg_mls:
#         return osr_roc_stats(mls_scores, open_set_labels), -torch.mean(mls_scores)
#     return osr_roc_stats(mls_scores, open_set_labels)



def load_and_eval_mls_osr_for_all_eps(path_to_eps_dirs, split_num, dataset_name='tinyimagenet', balance=True, return_avg_mls=False, return_quantiles=False, msp=False):
    eps_dir_list = [dir_name for dir_name in os.listdir(path_to_eps_dirs) if dir_name[:3] == 'eps']
    eps_list = [float(dir_name[4:]) for dir_name in eps_dir_list]
    roc_stats = []
    avg_mls_list = []
    for dir in eps_dir_list:
        stats = load_and_eval_mls_osr(path_to_eps_dirs + dir + '/logits_' + str(split_num) + '.pt',
                                               path_to_eps_dirs + dir + '/csr_targets_' + str(split_num) + '.pt',
                                               split_num=split_num,
                                               dataset_name=dataset_name,
                                               balance=balance,
                                               return_avg_mls=return_avg_mls,
                                               return_quantiles=return_quantiles,
                                               msp=msp)
        if return_avg_mls:
            roc_stat_tuple, avg_mls = stats
            avg_mls_list += avg_mls,
        else:
            roc_stat_tuple = stats
        roc_stats += roc_stat_tuple,
    if return_avg_mls:
        eps_roc_mls = sorted(zip(eps_list, roc_stats, avg_mls_list), key=lambda x: x[0])
        eps_list = [eps for eps, _, _ in eps_roc_mls]
        roc_stats = [roc_stat for _, roc_stat, _ in eps_roc_mls]
        avg_mls_list = [mls for _, _, mls in eps_roc_mls]
        return eps_list, roc_stats, avg_mls_list
    eps_roc = sorted(zip(eps_list, roc_stats), key=lambda x: x[0])
    eps_list = [eps for eps, _ in eps_roc]
    roc_stats = [roc_stat for _, roc_stat in eps_roc]
    return eps_list, roc_stats

def load_and_eval_logit_change_score(plain_logit_file_path, adv_logit_file_path, csr_targets_file_path, split_num,
                                     similarity_func=lambda after, before: torch.amax(after, dim=-1) - torch.amax(before, dim=-1),
                                     dataset_name='tinyimagenet', balance=True, return_avg_score=False):
    assert int(plain_logit_file_path[-4]) == split_num, "The split_num does not correspond to the split num of the file name"
    split = osr_splits[dataset_name][split_num]
    plain_logits = torch.load(plain_logit_file_path)
    adv_logits = torch.load(adv_logit_file_path)
    similarity_scores = similarity_func(adv_logits, plain_logits)
    csr_targets = torch.load(csr_targets_file_path)
    osr_targets = get_osr_targets(csr_targets, split)
    return eval_osr(similarity_scores, osr_targets, balance=balance, return_avg_score=return_avg_score)

def load_and_eval_logit_change_scores_for_all_eps(path_to_eps_dirs, path_to_plain_logit_file, split_num, similarity_func=lambda after, before: torch.amax(after, dim=-1) - torch.amax(before, dim=-1), dataset_name='tinyimagenet', balance=True, return_avg_score=False):
    eps_dir_list = [dir_name for dir_name in os.listdir(path_to_eps_dirs) if dir_name[:3] == 'eps']
    eps_list = [float(dir_name[4:]) for dir_name in eps_dir_list]
    roc_stats = []
    avg_score_list = []
    for dir in eps_dir_list:
        stats = load_and_eval_logit_change_score(path_to_plain_logit_file, 
                                                 path_to_eps_dirs + dir + '/logits_' + str(split_num) + '.pt', 
                                                 path_to_eps_dirs + dir + '/csr_targets_' + str(split_num) + '.pt',
                                                 split_num=split_num,
                                                 similarity_func=similarity_func,
                                                 return_avg_score=return_avg_score,
                                                 balance=balance,
                                                 dataset_name=dataset_name)
        if return_avg_score:
            roc_stat_tuple, avg_score = stats
            avg_score_list += avg_score,
        else:
            roc_stat_tuple = stats
        roc_stats += roc_stat_tuple,
    if return_avg_score:
        eps_roc_score = sorted(zip(eps_list, roc_stats, avg_score_list), key=lambda x: x[0])
        eps_list = [eps for eps, _, _ in eps_roc_score]
        roc_stats = [roc_stat for _, roc_stat, _ in eps_roc_score]
        avg_score_list = [mls for _, _, mls in eps_roc_score]
        return eps_list, roc_stats, avg_score_list
    eps_roc = sorted(zip(eps_list, roc_stats), key=lambda x: x[0])
    eps_list = [eps for eps, _ in eps_roc]
    roc_stats = [roc_stat for _, roc_stat in eps_roc]
    return eps_list, roc_stats

def max_logit_change_compared_id_vs_ood(path_plain_logits, path_fn_logits, path_csr_targets, split_num, dataset_name='tinyimagenet'):
    split = osr_splits[dataset_name][split_num]
    csr_targets = torch.load(path_csr_targets)
    osr_targets = get_osr_targets(csr_targets, split)
    plain_logits = torch.load(path_plain_logits)
    fn_logits = torch.load(path_fn_logits)
    max_logit_plain = torch.amax(plain_logits, dim=1)
    max_logit_fn = torch.amax(fn_logits, dim=1)
    diffs = max_logit_fn - max_logit_plain
    id_stats = [(diff, mls_plain) for diff, target, mls_plain in zip(diffs, osr_targets, max_logit_plain) if not target]
    ood_stats = [(diff, mls_plain) for diff, target, mls_plain in zip(diffs, osr_targets, max_logit_plain) if target]
    return id_stats, ood_stats

def get_grad_norm_stats(path_grad_norms, path_plain_logits, split_num, dataset_name='tinyimagenet', score_func=lambda x:torch.amax(x, dim=-1)):
    split = osr_splits[dataset_name][split_num]

    csr_targets = torch.load(path_grad_norms + 'csr_targets_' + str(split_num) + '.pt')
    osr_targets = get_osr_targets(csr_targets, split)

    plain_logits = torch.load(path_plain_logits + 'logits_' + str(split_num) + '.pt')
    osr_scores = score_func(plain_logits)

    grad_norms = torch.load(path_grad_norms + 'grad_norms_' + str(split_num) + '.pt')

    id_grad_norms = [(grad_norm, osr_score) for grad_norm, osr_score, target in zip(grad_norms, osr_scores, osr_targets) if not target]
    ood_grad_norms = [(grad_norm, osr_score) for grad_norm, osr_score, target in zip(grad_norms, osr_scores, osr_targets) if target]

    return id_grad_norms, ood_grad_norms

def get_diff_stats_for_eps(path_plain_logits, path_to_attack_folder, path_csr_targets, split_num=0, dataset_name='tinyimagenet'):
    dirs = os.listdir(path_to_attack_folder)
    eps_list = []
    id_box_list = []
    ood_box_list = []
    for name in dirs:
        eps_list.append(float(name[4:]))
        id_stats, ood_stats = max_logit_change_compared_id_vs_ood(path_plain_logits, 
                                            path_to_attack_folder + name + '/logits_' + str(split_num) + '.pt',
                                            path_csr_targets,
                                            split_num=split_num,
                                            dataset_name=dataset_name)
        id_diffs, _ = list(zip(*id_stats))
        ood_diffs, _ = list(zip(*ood_stats))
        id_box_stats = np.quantile(id_diffs,[0.25, 0.5, 0.75])
        ood_box_stats = np.quantile(ood_diffs,[0.25, 0.5, 0.75])
        id_box_list.append(id_box_stats)
        ood_box_list.append(ood_box_stats)
    sorted_zip = sorted(zip(eps_list, id_box_list, ood_box_list), key=lambda x: x[0])
    return list(zip(*sorted_zip))