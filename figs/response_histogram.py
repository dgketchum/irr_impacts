import os
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings(action='once')

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white", {'axes.linewidth': 0.5})


def plot_response(d, out_fig, glob=None):
    l = [(int(x.split('.')[0].split('_')[-1]), os.path.join(d, x)) for x in os.listdir(d) if glob in x]
    dct = {}
    for k, v in l:
        with open(v, 'r') as fp:
            mdct = json.load(fp)
        dct[k] = ([np.log10(v['AREA']) for k, v in mdct.items()], [v['lag'] for k, v in mdct.items()])

    fig = plt.figure(figsize=(16, 10), dpi=80)
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)

    ax_main = fig.add_subplot(grid[:-1, :-1])
    ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
    ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

    for k, v in dct.items():
        sns.kdeplot(y=v[1], ax=ax_right,)
        ax_main.scatter(v[0], v[1], s=25, alpha=.9,
                        cmap="tab10", edgecolors='gray', linewidths=.5)
    ax_right.set_xlabel(None)
    ax_right.set_ylabel(None)
    ax_right.set_ylim([-20, 60])

    sns.kdeplot(data=dct[1][0], ax=ax_bottom)
    ax_bottom.set_xlabel(None)
    ax_bottom.set_ylabel(None)
    ax_bottom.invert_yaxis()

    ax_main.set(title='Streamflow - Climate Response Time', xlabel='log(Area)',
                ylabel='Lag Time')
    ax_main.title.set_fontsize(20)

    plt.sca(ax_main)

    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=12)
    plt.tick_params(width=2, length=10)
    plt.tick_params(width=3, length=10)

    sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

    plt.show()
    plt.savefig(out_fig)

    if __name__ == '__main__':
        pass
# ========================= EOF ====================================================================
