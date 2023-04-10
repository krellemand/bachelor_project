from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score
import torch
from data.open_set_splits.osr_splits import osr_splits
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

def evaluate_osr(open_set_scores, open_set_labels):
    """
    Compute performance metrics based on the logits of
    an OSR model obtained by evaluating the model on a 
    test set.
    """
    fprs, tprs, thresholds = roc_curve(open_set_labels, open_set_scores, drop_intermediate=False)
    auroc = roc_auc_score(open_set_labels, open_set_scores)
    return (fprs, tprs, thresholds), auroc

def mls_osr_score(logits):
    return -torch.amax(logits, dim=1) # Low osr score corresponds to known class

def get_osr_targets(csr_targets, split_targets):
    return (~(sum(csr_targets == i for i in split_targets).bool()))

def load_and_eval_mls_osr(logit_file_path, csr_targets_file_path, split_num, dataset_name='tinyimagenet', balance=False, return_avg_mls=False):
    assert int(logit_file_path[-4]) == split_num, "The split_num does not correspond to the split num of the file name"
    split = osr_splits[dataset_name][split_num]
    logits = torch.load(logit_file_path)
    csr_targets = torch.load(csr_targets_file_path)
    mls_scores = mls_osr_score(logits)
    open_set_labels = get_osr_targets(csr_targets, split)
    # print((sum(open_set_labels)/len(open_set_labels)).data) # Balance ratio before
    if balance:
        mls_oslabel = balance_binary(zip(mls_scores.tolist(), open_set_labels.tolist()), lambda x: bool(x[1]))
        mls_scores, open_set_labels = (torch.tensor([mls for mls, _ in mls_oslabel ]), 
                                       [oslabel for _, oslabel in mls_oslabel])
    # print(sum(open_set_labels)/len(open_set_labels)) # Balance ratio after
    if return_avg_mls:
        return evaluate_osr(mls_scores, open_set_labels), -torch.mean(mls_scores)
    return evaluate_osr(mls_scores, open_set_labels)

def load_and_eval_mls_osr_for_all_eps(path_to_eps_dirs, split_num, dataset_name='tinyimagenet', balance=False, return_avg_mls=False):
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
                                               return_avg_mls=return_avg_mls)
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

def max_logit_change_compared_id_vs_ood(path_plain_logits, path_fn_logits, path_csr_targets, split_num, dataset_name='tinyimagenet'):
    split = osr_splits[dataset_name][split_num]
    csr_targets = torch.load(path_csr_targets)
    osr_targets = get_osr_targets(csr_targets, split)
    plain_logits = torch.load(path_plain_logits)
    fn_logits = torch.load(path_fn_logits)
    max_logit_plain = torch.amax(plain_logits, dim=1)
    max_logit_fn = torch.amax(fn_logits, dim=1)
    diffs = max_logit_fn - max_logit_plain
    id_diffs = [diff for diff, target in zip(diffs, osr_targets) if not target]
    ood_diffs = [diff for diff, target in zip(diffs, osr_targets) if target]
    return id_diffs, ood_diffs

def get_grad_norm_stats(path_grad_norms, path_plain_logits, split_num, dataset_name='tinyimagenet', score_func=lambda x:torch.amax(x), dim=-1):
    split = osr_splits[dataset_name][split_num]

    csr_targets = torch.load(path_grad_norms + 'csr_targets_' + split + '.pt')
    osr_targets = get_osr_targets(csr_targets, split)

    plain_logits = torch.load(path_plain_logits)
    osr_scores = score_func(plain_logits)

    grad_norms = torch.load(path_grad_norms)

    id_grad_norms = [(grad_norm, osr_score) for grad_norm, osr_score, target in zip(grad_norms, osr_scores, osr_targets) if not target]
    ood_grad_norms = [(grad_norm, osr_score) for grad_norm, osr_score, target in zip(grad_norms, osr_scores, osr_targets) if target]

    return id_grad_norms, ood_grad_norms

