import numpy as np
import matplotlib.pyplot as plt


def plot_regressions(x, y, sigma_y, sigma_x=None):
    figure = plt.figure(figsize=(8, 6))
    ax = figure.add_subplot(111)
    ax.scatter(x, y, alpha=0.5, color='b')
    if isinstance(sigma_x, np.ndarray):
        ax.errorbar(x, y, xerr=sigma_x, yerr=sigma_y, alpha=0.1, ls='', color='b')
    else:
        ax.errorbar(x, y, yerr=sigma_y, alpha=0.1, ls='', color='b')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_regression_from_trace(fitted, observed, ax=None, chains=None):
    traces = [fitted.trace, ]
    xi, yi, sigx, sigy = observed

    for i, trace in enumerate(traces):

        trace_slope = trace['slope']
        trace_inter = trace['inter']

        if chains is not None:
            for chain in range(100, len(trace), 5):
                alpha, beta = trace_inter[chain], trace_slope[chain]
                y = alpha + beta * xi
                ax.plot(xi, y, alpha=0.03, c='red')

        # plot the best-fit line only
        H2D, bins1, bins2 = np.histogram2d(trace_slope,
                                           trace_inter, bins=50)

        w = np.where(H2D == H2D.max())

        # choose the maximum posterior slope and intercept
        slope_best = bins1[w[0][0]]
        intercept_best = bins2[w[1][0]]

        print("beta:", slope_best, "alpha:", intercept_best)
