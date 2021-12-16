import os

from matplotlib import rcParams, pyplot as plt


def plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d, cci_per, flow_per):
    resid_line = fit_resid.params[1] * cc + fit_resid.params[0]
    clim_line = fit_clim.params[1] * ai + fit_clim.params[0]
    rcParams['figure.figsize'] = 16, 10
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(ai, q)
    ax1.plot(ai, clim_line)
    ax1.set(xlabel='ETr / PPT [-]')
    ax1.set(ylabel='q [m^3]')

    for i, y in enumerate(years):
        ax1.annotate(y, (ai[i], q[i]))
        plt.suptitle(desc_str)

    ax2.set(xlabel='cc [m]')
    ax2.set(ylabel='q epsilon [m^3]')
    ax2.scatter(cc, resid)
    ax2.plot(cc, resid_line)
    for i, y in enumerate(years):
        ax2.annotate(y, (cc[i], resid[i]))

    desc_split = desc_str.strip().split('\n')
    file_name = desc_split[0].replace(' ', '_')

    fig_name = os.path.join(fig_d, '{}_cc_{}-{}_q_{}-{}.png'.format(file_name, cci_per[0], cci_per[1],
                                                                    flow_per[0], flow_per[1]))

    plt.savefig(fig_name)
    plt.close('all')


def plot_water_balance_trends(data, data_line, data_str, years, desc_str, fig_d):
    rcParams['figure.figsize'] = 16, 10
    fig, ax1 = plt.subplots(1, 1)

    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.scatter(years, data, color=color)
    ax1.plot(years, data_line, color=color)
    ax1.set_ylabel(data_str, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    desc_split = desc_str.strip().split('\n')
    file_name = desc_split[0].replace(' ', '_')

    fig_name = os.path.join(fig_d, '{}_{}.png'.format(file_name, data_str))

    plt.savefig(fig_name)
    plt.close('all')


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
