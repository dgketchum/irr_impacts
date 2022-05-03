import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize
from astroML.linear_model import TLS_logL, LinearRegression


# TLS:
def get_m_b(beta):
    b = np.dot(beta, beta) / beta[1]
    m = -beta[0] / beta[1]
    return m, b


def plot_regressions(x, y, sigma_x, sigma_y):
    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    ax.scatter(x, y, alpha=0.5, color='b')
    ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, alpha=0.1, ls='', color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_regression_from_trace(fitted, observed, ax=None, chains=4, multidim_ind=None,
                               traces=None, legend=True, n_lines=100, burn=500,
                               chain_idx=None):
    if not traces:
        traces = [fitted.trace, ]
    else:
        traces = [traces, ]

    xi, yi, sigx, sigy = observed

    if multidim_ind is not None:
        xi = xi[multidim_ind]

    x = np.linspace(np.min(xi), np.max(xi), 50)

    for i, trace in enumerate(traces):
        try:
            trace_slope = trace['slope'][:, 0]
        except KeyError:
            trace_slope = trace['posterior']['slope'].values.ravel()
        except IndexError:
            trace_slope = trace['slope']

        trace_slope = trace_slope.reshape(-1, chains)

        try:
            trace_inter = trace['inter']
        except KeyError:
            trace_inter = trace['posterior']['inter'].values.ravel()

        trace_inter = trace_inter.reshape(-1, chains)

        if chain_idx:
            trace_slope = trace_slope[:, chain_idx]
            trace_inter = trace_inter[:, chain_idx]
            chains = len(chain_idx)

        sample_idx = np.array((np.random.randint(trace_slope.shape[0], size=n_lines),
                               np.random.randint(0, chains, size=n_lines)))

        slope_samples = [x for x in zip(sample_idx[0], sample_idx[1])]

        for chain in slope_samples:
            y = trace_inter[chain] + trace_slope[chain] * x
            ax.plot(x, y, alpha=0.1, c='red')

        # plot the best-fit line only
        H2D, bins1, bins2 = np.histogram2d(trace_slope.ravel(),
                                           trace_inter.ravel(), bins=50)
        w = np.where(H2D == H2D.max())

        # choose the maximum posterior slope and intercept
        slope_best = bins1[w[0][0]]
        intercept_best = bins2[w[1][0]]

        print("beta:", slope_best, "alpha:", intercept_best)
        y = intercept_best + slope_best * x

        # y_pre = fitted.predict(x[:, None])
        # ax.plot(x, y, label='MAP')

        break
    if legend:
        ax.legend()

    return slope_best
