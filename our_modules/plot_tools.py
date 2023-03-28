import matplotlib.pyplot as plt
import numpy as np

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