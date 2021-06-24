import numpy as np
import pandas as pd
from scipy.signal import lfilter, lfilter_zi, argrelmin, lfiltic
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg


def eckhardt(q, alpha, beta, init_value=None, window=7):
    """Eckhardt recursive filter for hydrograph separation.
    Implementation of the Eckhardt digital recursive filter.
    .. math:
        q_{b(i)} = \frac{(1-\beta)\alpha q_{b(i-1)} + (1-\alpha)\beta q_i}
        {1-\alpha \beta}
    Parameters
    ----------
    q : array_like
        Time series of streamflow values.
    alpha : float
        Recession coefficient representing the proportion of remaining
        streamflow on the next time step. It should be strictly greather than
        0 and lower than 1.
    beta : float
        Maximum baseflow index representing the long term ratio between
        baseflow and streamflow.
    init_value : None, float, optional
        Initial value of the baseflow used to initiate Eckhardt filter. If None,
        then the first minimum of the first window
    window : int, optional
    Returns
    -------
    qb : array_like
        Time series of baseflow values.

    """

    if (alpha <= 0) or (alpha >= 1.):
        raise ValueError(
            'Parameter alpha should be between range 0 < alpha < 1.')

    if (beta <= 0) or (beta >= 1.):
        raise ValueError(
            'Parameter beta should be between range 0 < beta < 1.')

    # Filter parameters are refactored to fit the scipy linear filter
    # parameters

    # input coefficient for the multiple backwards operators
    b = np.array([(1 - alpha) * beta / (1 - alpha * beta)])
    # output coefficient for the multiple backwards operators
    a = np.array([1, -(1 - beta) * alpha / (1 - alpha * beta)])

    # setting initial value

    if init_value is not None:
        qb0 = lfiltic(b, a, [init_value])
    else:
        locmin = min(q[:window])
        qb0 = lfiltic(b, a, [locmin])

    qb, _ = lfilter(b, a, q, zi=qb0)

    # find where qb is higher than q

    invalid = np.where(qb > q)

    # replace qb > q values by q

    qb[invalid] = q.iloc[invalid]

    return qb


def brutsaert(df):
    df['dqdt'] = df.diff()
    df['neg'] = df['dqdt'] < 0
    df['neg'] = df['neg'] & df['q'] > 0
    df['pos'] = df['dqdt'] > 0
    pos = np.nonzero(df['pos'].values)[0]
    buf_pos = list(set([i + a for a in range(-2, 4) for i in pos]))
    df['buf_pos'] = df['pos']
    df['buf_pos'][buf_pos] = True
    df['neg'][df['buf_pos'].values] = False
    df = df[df['neg']]
    df['dqdt_left'] = df['dqdt'].shift()
    df = df.dropna()
    mod = QuantReg(df['dqdt'], df['dqdt_left'])
    res = mod.fit(q=.5)
    # print(res.summary())
    return res.history['params'][-1][0]


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
