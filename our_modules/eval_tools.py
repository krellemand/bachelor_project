from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score
import torch
from data.open_set_splits.osr_splits import osr_splits

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

def load_and_eval_mls_osr(logit_file_path, csr_targets_file_path, split_num, dataset_name='tinyimagenet'):
    assert int(logit_file_path[-4]) == split_num, "The split_num does not correspond to the split num of the file name"
    split = osr_splits[dataset_name][split_num]
    logits = torch.load(logit_file_path)
    csr_targets = torch.load(csr_targets_file_path)
    mls_scores = mls_osr_score(logits)
    open_set_labels = get_osr_targets(csr_targets, split)
    return evaluate_osr(mls_scores, open_set_labels)