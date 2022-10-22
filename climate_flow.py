import os
import json
from datetime import date
from calendar import monthrange
from dateutil.relativedelta import relativedelta as rdlt

import numpy as np
from scipy.stats.stats import linregress

from gage_data import hydrograph


def climate_flow_correlation(climate_dir, month, in_json, out_json):
    """Find linear relationship between climate and flow in an expanding time window"""

    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])

    windows = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)

    for csv in l:
        sid = os.path.basename(csv).strip('.csv')

        try:
            s_meta = metadata[sid]
        except KeyError:
            continue

        df = hydrograph(csv)

        years = np.arange(1991, 2021)

        response_d = {}
        offsets = np.arange(1, 61)

        corr = (0, 0.0)
        q_dates = [(date(y, month, 1), date(y, month, monthrange(y, month)[1])) for y in years]
        q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])

        # get concurrent (w/ flow) data to be used in single-month trends analysis
        cc = np.array([df['cc'][d[0]: d[1]].sum() for d in q_dates])
        etr = np.array([df['etr'][d[0]: d[1]].sum() for d in q_dates])
        ppt = np.array([df['ppt'][d[0]: d[1]].sum() for d in q_dates])
        ai_m = etr - ppt
        irr = np.array([df['irr'][d[0]: d[1]].sum() for d in q_dates])

        for lag in offsets:

            dates = [(date(y, month, monthrange(y, month)[1]) + rdlt(months=-lag, days=1),
                      date(y, month, monthrange(y, month)[1])) for y in years]

            etr = np.array([df['etr'][d[0]: d[1]].sum() for d in dates])
            ppt = np.array([df['ppt'][d[0]: d[1]].sum() for d in dates])
            ai = etr - ppt
            lr = linregress(ai, q)
            b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
            qres = q - (b * ai + inter)

            if abs(r) > corr[1] and p < 0.05:
                corr = (lag, abs(r))

                response_d[month] = {'inter': inter,
                                     'b': b,
                                     'lag': lag,
                                     'r': r,
                                     'p': p,
                                     'q': list(q),
                                     'ai': list(ai),
                                     'qres': list(qres),
                                     'ai_month': list(ai_m),
                                     'ppt_month': list(ppt),
                                     'etr_month': list(etr),
                                     'cc_month': list(cc),
                                     'irr': list(irr),
                                     }

        windows[sid] = {**response_d, **s_meta}

    with open(out_json, 'w') as f:
        json.dump(windows, f, indent=4)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
