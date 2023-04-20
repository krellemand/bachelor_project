import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import torch

from our_modules.eval_tools import load_and_eval_mls_osr_for_all_eps
from our_modules.eval_tools import load_and_eval_logit_change_scores_for_all_eps
from our_modules.eval_tools import get_grad_norm_stats
from our_modules.eval_tools import max_logit_change_compared_id_vs_ood
from our_modules.eval_tools import get_diff_stats_for_eps

def plot_roc(ax, roc_stats, **plt_kwargs):
    fprs, tprs, thresholds = roc_stats
    ax.plot(fprs, tprs, **plt_kwargs)


def plot_image_on_ax(ax, normalized_img, mean, std, **plt_kwargs):
    normalized_img = normalized_img.to('cpu')
    if normalized_img.requires_grad:
        normalized_img = normalized_img.detach()
    img = normalized_img*np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    ax.imshow(np.clip(img.permute(1,2,0).numpy(), 0.0, 1.0), **plt_kwargs)


def plot_image(normalized_img, mean, std, **plt_kwargs):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    plot_image_on_ax(ax, normalized_img.to('cpu'), mean=mean, std=std, **plt_kwargs)
    plt.show()


def plot_image_i(i, dataset, mean, std, **plt_kwargs):
    img, label, uq_idx = dataset[i]
    # Unnormalize
    img = img*np.array(std)[:, None, None] + np.array(mean)[:, None, None]
    # The image is clipped due to numerical imprecision.
    plt.imshow(np.clip(img.permute(1,2,0).numpy(), 0.0, 1.0), **plt_kwargs)
    print(label, uq_idx)
    plt.show()


class EpsExperimentPlot():
    def __init__(self, eps_figsize=(10,4), adv_fisize=(15,6), which_lines='both', add_zoom=(-0.003, 0.012, 0.825, 0.84)):
        self.which_lines = which_lines
        self.add_zoom = add_zoom
        eps_fig, eps_ax = plt.subplots(1,1, figsize=eps_figsize)
        self.fig = eps_fig
        self.ax1 = eps_ax
        self.ax1.set_xlabel('$\\epsilon$ - the size of the advesarial perturbation.')
        self.ax1.set_ylabel('AUROC', c='black')
        if self.add_zoom:
            self.axins = zoomed_inset_axes(self.ax1, 10, loc=1)
        if self.which_lines == 'both':
            self.ax1.set_xlabel('$\\epsilon$ - the size of the advesarial perturbation.')
            self.ax1.set_ylabel('AUROC', c='red')
            self.ax2 = eps_ax.twinx()
            self.ax2.set_ylabel('Average OSR Score - $\\mathcal{S}\\:(y\\in\\mathcal{C}\\mid x)$', c='blue')
        if self.which_lines == 'mls':
            self.ax2 = self.ax1
            self.ax2.set_ylabel('Average OSR Score - $\\mathcal{S}\\:(y\\in\\mathcal{C}\\mid x)$', c='black')
        self.recent_eps = None
        self.eps = None
        self.roc_stats = None
        self.avg_scores = None

    def load_mls_stats(self, path_to_eps_dirs, split_num, balance=True, dataset_name='tinyimagenet'):
        self.eps, self.roc_stats, self.avg_scores = load_and_eval_mls_osr_for_all_eps(path_to_eps_dirs, split_num, dataset_name=dataset_name, balance=balance, return_avg_mls=True)

    def load_logit_change_stats(self, path_to_eps_dirs, path_to_plain_logit_file, split_num, similarity_func=lambda after, before: torch.amax(after, dim=-1) - torch.amax(before, dim=-1), balance=True, dataset_name='tinyimagenet'):
        self.eps, self.roc_stats, self.avg_scores = load_and_eval_logit_change_scores_for_all_eps(path_to_eps_dirs, path_to_plain_logit_file, split_num, similarity_func=similarity_func, dataset_name=dataset_name, balance=balance, return_avg_score=True)
   
    def add_to_eps_plot(self, label_suffix='', **plt_kwargs):
        aurocs = [x[1] for x in self.roc_stats]
        self.recent_eps = np.array(self.eps)
        if self.which_lines == 'both':
            self.ax1.plot(self.eps, aurocs, c='red', label='AUROC' + label_suffix)
            if self.add_zoom:
                self.axins.plot(self.eps, aurocs, c='red')
            self.ax2.plot(self.eps, self.avg_scores, c='blue', label='Average OSR Score' + label_suffix)
            if self.add_zoom:
                self.axins.plot(self.eps, self.avg_scores, c='blue')
        if self.which_lines == 'AUROC':
            self.ax1.plot(self.eps, aurocs, label='AUROC' + label_suffix, **plt_kwargs)
            if self.add_zoom:
                self.axins.plot(self.eps, aurocs, **plt_kwargs)
        if self.which_lines == 'mls':
            self.ax2.plot(self.eps, self.avg_scores, label='Average OSR Score' + label_suffix, **plt_kwargs)
            if self.add_zoom:
                self.axins.plot(self.eps, aurocs, **plt_kwargs)
        if self.add_zoom:
            self.axins.set_xlim(self.add_zoom[0], self.add_zoom[1])
            self.axins.set_ylim(self.add_zoom[2], self.add_zoom[3])
            mark_inset(self.ax1, self.axins, loc1=2, loc2=3, fc="none", ec="0.5")

    def load_and_add_mls_to_eps_plot(self, path_to_eps_dirs, split_num, balance=True, label_suffix='', 
                                     dataset_name='tinyimagenet', **plt_kwargs):
        self.load_mls_stats(path_to_eps_dirs, split_num, balance=balance, dataset_name=dataset_name)
        self.add_to_eps_plot(label_suffix=label_suffix, **plt_kwargs)

    def load_and_add_logit_change_to_eps_plot(self, path_to_eps_dirs, path_to_plain_logit_file, split_num, 
                                              similarity_func=lambda after, before: torch.amax(after, dim=-1) - torch.amax(before, dim=-1), 
                                              balance=True, label_suffix='', dataset_name='tinyimagenet', **plt_kwargs):
        self.load_logit_change_stats(path_to_eps_dirs, path_to_plain_logit_file, split_num, similarity_func=similarity_func, 
                                     balance=balance, dataset_name=dataset_name)
        self.add_to_eps_plot(label_suffix, **plt_kwargs)

    def set_legend_and_highlight_eps(self, eps_idxs=[], legend_loc=(0.72,0.8), h_line=False):
        for i in eps_idxs:
            self.ax1.axvline(round(self.recent_eps[i],2), 0, 1, linestyle='dashed', c='gray', alpha=0.5)
            if self.add_zoom:
                self.axins.axvline(self.recent_eps[i], 0, 1, linestyle='dashed', c='gray', alpha=0.5)
        if h_line:
            self.ax1.axhline(h_line, 0, 1, linestyle = 'dashed', c='salmon', alpha=0.5, label='MLS AUROC')
            if self.add_zoom:
                self.axins.axhline(h_line, 0, 1, linestyle = 'dashed', c='salmon', alpha=0.5)
        chosen_eps = [self.recent_eps[i] for i in eps_idxs]
        locs, labels = plt.xticks()
        locs = locs[1:-1]
        locs = [l for l in locs if not np.isclose(l, chosen_eps, rtol=0.0, atol=0.2).any()]
        locs += chosen_eps
        plt.xticks([round(l, 2) for l in locs])
        self.fig.legend(loc=legend_loc)

    def show_and_save(self, save_path=False):
        plt.show()
        if save_path:
            plt.savefig(save_path, transparent=True, bbox_inches='tight')

class IdOodPlot():
    def __init__(self):
        self.fig = None
        self.ax = None
        self.id_ys = None
        self.id_xs = None
        self.ood_ys = None
        self.ood_xs = None
    
    def load_grad_norm_stats(self, path_grad_norms, path_plain_logits, split_num, dataset_name='tinyimagenet', score_func=lambda x:torch.amax(x, dim=-1), balance=True):
        id_stats, ood_stats = get_grad_norm_stats(path_grad_norms, path_plain_logits, split_num, dataset_name, score_func)
        id_stats = sorted(id_stats, key=lambda x: x[1])
        ood_stats = sorted(ood_stats, key=lambda x: x[1])
        if balance:
            selected_idxs = np.random.choice(len(ood_stats), len(id_stats), replace=False)
            ood_stats = [stat for i, stat in enumerate(ood_stats) if i in selected_idxs]
        self.id_ys = [gn for gn, _ in id_stats]
        self.id_xs = [s for _, s in id_stats]
        self.ood_ys = [gn for gn, _ in ood_stats]
        self.ood_xs = [s for _, s in ood_stats]

    def load_mls_diffs_stats(self, path_plain_logits, path_fn_logits, path_csr_targets, split_num, dataset_name='tinyimagenet', balance=True):
        id_stats, ood_stats = max_logit_change_compared_id_vs_ood(path_plain_logits, path_fn_logits, path_csr_targets, split_num, dataset_name=dataset_name)
        id_stats = sorted(id_stats, key=lambda x: x[1])
        ood_stats = sorted(ood_stats, key=lambda x: x[1])
        if balance:
            selected_idxs = np.random.choice(len(ood_stats), len(id_stats), replace=False)
            ood_stats = [stat for i, stat in enumerate(ood_stats) if i in selected_idxs]
        self.id_ys = [gn for gn, _ in id_stats]
        self.id_xs = [s for _, s in id_stats]
        self.ood_ys = [gn for gn, _ in ood_stats]
        self.ood_xs = [s for _, s in ood_stats]

    def make_boxplot(self, figsize=(6,6), **boxplot_kwargs):
        self.fig, self.ax = plt.subplots(1,1, figsize=figsize)
        self.ax.boxplot((self.id_ys, self.ood_ys), **boxplot_kwargs)

    def make_scatter_plot(self, window_size=5,figsize=(6,6), xlabel=r'$\mathcal{S}$ - Maximum Logit Score (MLS)', ylabel=r'Gradient Norm - $\mathbb{E}\:\left[||\:\nabla_{\bf{x}} \log S_{\hat{y}}({\bf{x}}) \:||_1 \mid \mathcal{S}\:\right]$'):
        self.fig, self.ax = plt.subplots(1,1, figsize=figsize)
        self.ax.scatter(self.id_xs, self.id_ys, alpha=0.2, c='cornflowerblue')
        self.ax.scatter(self.ood_xs, self.ood_ys, alpha=0.2, c='salmon')
        id_scores_vs_means = [(score, np.mean(self.id_ys[max(i-window_size, 0):i + window_size + 1])) \
                              for i, score in enumerate(self.id_xs)][window_size:-window_size]
        ood_scores_vs_means = [(score, np.mean(self.ood_ys[max(i-window_size, 0):i + window_size + 1])) \
                              for i, score in enumerate(self.ood_xs)][window_size:-window_size]
        id_scores, id_means = list(zip(*id_scores_vs_means))
        ood_scores, ood_means = list(zip(*ood_scores_vs_means))
        self.ax.plot(id_scores, id_means, label='ID')
        self.ax.plot(ood_scores, ood_means, label='OOD', c='red')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    # def make_conditional_mean_plot(self, figsize=(6,6), **plot_kwargs):
    def set_legend(self):
        self.ax.legend()

    def show_and_save(self, save_path=False):
        plt.show()
        if save_path:
            plt.savefig(save_path, transparent=True, bbox_inches='tight')


def plot_diff_stats_for_eps(path_plain_logits, path_to_attack_folder, path_csr_targets, split_num=0, dataset_name='tinyimagenet', figsize = (6,6), highlight_eps_idx=5):
    eps_list, id_stats, ood_stats = get_diff_stats_for_eps(path_plain_logits, path_to_attack_folder, path_csr_targets, split_num=split_num, dataset_name=dataset_name)
    id_q1, id_q2, id_q3 = list(zip(*id_stats))
    ood_q1, ood_q2, ood_q3 = list(zip(*ood_stats))
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.plot(eps_list, id_q2, label='ID', c='b')
    ax.fill_between(eps_list, id_q1, id_q3, color='cornflowerblue', alpha=0.2)
    ax.plot(eps_list, ood_q2, label='OOD', c='r')
    ax.fill_between(eps_list, ood_q1, ood_q3, color='salmon', alpha=0.2)
    ax.axvline(eps_list[highlight_eps_idx], 0, 1, linestyle='dashed', c='gray', alpha=0.5, label=f'$\epsilon$ = {eps_list[highlight_eps_idx]:.2}')
    ax.legend()
    ax.set_xlabel('$\\epsilon$ - the size of the advesarial perturbation.')
    ax.set_ylabel(r'Signed Maximum Logit Change;  $\mathcal{S}_{adv} - \mathcal{S}$')
    plt.show()

def plot_adv_imgs(eps, adv_imgs, adv_steps, mean, std, figsize=(15,10)):
    img_stack = torch.vstack((adv_imgs[None], adv_steps[None]))
    fig, axs = plt.subplots(2, len(adv_imgs), figsize=figsize)
    if len(axs.shape) == 1:
        axs = axs[:, None]
    for i in range(2):
        for j in range(len(adv_imgs)):
            plot_image_on_ax(axs[i, j], img_stack[i, j], mean, std)
            axs[i, j].set_title((f"$\\epsilon = {eps[j]:.3}$"))
            axs[i, j].axis('off')
    plt.show()
