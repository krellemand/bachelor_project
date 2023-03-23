import matplotlib.pyplot as plt

def plot_roc(ax, roc_stats, **plt_kwargs):
    fprs, tprs, thresholds = roc_stats
    ax.plot(fprs, tprs, **plt_kwargs)
 
