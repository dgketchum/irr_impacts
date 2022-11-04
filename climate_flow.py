import os
import json
from datetime import date
from calendar import monthrange
from dateutil.relativedelta import relativedelta as rdlt

import numpy as np
from scipy.stats.stats import linregress

from gage_data import hydrograph
from utils.gage_lists import EXCLUDE_STATIONS


def climate_flow_correlation(q_flow_dir, month, in_json, out_json, start_yr=1987, end_yr=2021):
    """Find linear relationship between climate and flow in an expanding time window"""

    l = sorted([os.path.join(q_flow_dir, x) for x in os.listdir(q_flow_dir)])
    excluded, short_q, no_relation = [], [], []
    windows = {}

    with open(in_json, 'r') as f:
        metadata = json.load(f)

    for csv in l:
        sig_relationship = False
        sid = os.path.basename(csv).strip('.csv')

        if sid in EXCLUDE_STATIONS:
            excluded.append(sid)
            continue

        try:
            s_meta = metadata[sid]
        except KeyError:
            continue

        df = hydrograph(csv)

        years = range(start_yr, end_yr + 1)

        response_d = {}
        offsets = range(1, 61)

        corr = (0, 0.0)
        q_dates = np.array([(date(y, month, 1), date(y, month, monthrange(y, month)[1])) for y in years])
        q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])

        mask = q > 0.

        if np.count_nonzero(mask) < 20:
            print('\nonly {} q records month {}, {}, {}'.format(np.count_nonzero(q > 1),
                                                                month, sid, s_meta['STANAME']))
            short_q.append(sid)
            continue

        if np.count_nonzero(~mask) > 0:
            q_dates = q_dates[mask]
            q = q[mask]
            years = [d[0].year for d in q_dates]

        # get concurrent (w/ flow) data to be used in single-month trends analysis
        cc = np.array([df['cc'][d[0]: d[1]].sum() for d in q_dates])
        etr_m = np.array([df['gm_etr'][d[0]: d[1]].sum() for d in q_dates])
        ppt_m = np.array([df['gm_ppt'][d[0]: d[1]].sum() for d in q_dates])
        ai_m = etr_m - ppt_m
        irr = np.array([df['irr'][d[0]: d[1]].sum() for d in q_dates])
        lr = linregress(ai_m, cc)
        b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
        cc_res = cc - (b * ai_m + inter)

        for lag in offsets:

            dates = [(date(y, month, monthrange(y, month)[1]) + rdlt(months=-lag, days=1),
                      date(y, month, monthrange(y, month)[1])) for y in years]

            etr = np.array([df['gm_etr'][d[0]: d[1]].sum() for d in dates])
            ppt = np.array([df['gm_ppt'][d[0]: d[1]].sum() for d in dates])
            ai = etr - ppt
            lr = linregress(ai, q)
            b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
            qres = q - (b * ai + inter)
            if abs(r) > corr[1] and p < 0.05:
                sig_relationship = True
                corr = (lag, abs(r))

                response_d = {'inter': inter,
                              'b': b,
                              'lag': lag,
                              'r': r,
                              'p': p,
                              'q': list(q),
                              'q_mo': month,
                              'ai': list(ai),
                              'etr': list(etr),
                              'ppt': list(ppt),
                              'qres': list(qres),
                              'ai_month': list(ai_m),
                              'ppt_month': list(ppt_m),
                              'etr_month': list(etr_m),
                              'cc_month': list(cc),
                              'ccres_month': list(cc_res),
                              'irr': list(irr),
                              'years': list(years),
                              }
        if sig_relationship:
            s_meta.update(response_d)
            windows[sid] = s_meta
        else:
            no_relation.append(sid)
            print('\nno significant relation {}, r = {:.3f}, month {}, {}'.format(sid, r, month, s_meta['STANAME']))

    with open(out_json, 'w') as f:
        json.dump(windows, f, indent=4)

    print('{} short discharge records'.format(len(short_q)))
    print('{} excluded gages'.format(len(excluded)))
    print('{} no sig relationship gages'.format(len(no_relation)))
    print('{} gages included'.format(len(windows.keys())))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
