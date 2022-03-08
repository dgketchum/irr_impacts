import os

from matplotlib import pyplot as plt
from astroML.plotting import plot_regressions, plot_regression_from_trace


def plot_trace(cc, qres, cc_err, qres_err, model, qres_cc_lr, fig_dir,
               desc_str=''):
    plot_regressions(0.0, 0.0, cc[0], qres,
                     cc_err[0], qres_err,
                     add_regression_lines=True,
                     alpha_in=qres_cc_lr.intercept, beta_in=qres_cc_lr.slope)

    plt.scatter(cc, qres)
    plt.xlim([cc.min(), cc.max()])
    plt.ylim([qres.min(), qres.max()])

    plot_regression_from_trace(model, (cc, qres, cc_err, qres_err),
                               ax=plt.gca(), chains=50)

    plt.suptitle(desc_str)
    fig_file = os.path.join(fig_dir, '{}.png'.format(desc_str))
    plt.savefig(fig_file)
    plt.close()


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
