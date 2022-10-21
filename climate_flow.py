import os
import json
from datetime import date
from calendar import monthrange
from dateutil.relativedelta import relativedelta as rdlt

import numpy as np
from scipy.stats.stats import linregress

from hydrograph import hydrograph
from station_lists import VERIFIED_IRRIGATED_HYDROGRAPHS


def climate_flow_correlation(climate_dir, in_json, out_json, spec_time=None):
    """Find linear relationship between climate and flow in an expanding time window"""

    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])

    offsets = [x for x in range(1, 61)]
    windows = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)

    if abs(spec_time[1] - spec_time[0]) > 0:
        spec_time = tuple([x for x in range(spec_time[0], spec_time[1] + 1)])

    for csv in l:
        sid = os.path.basename(csv).strip('.csv')

        if sid not in VERIFIED_IRRIGATED_HYDROGRAPHS:
            continue
        print(sid)
        s_meta = metadata[sid]

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_frac = mean_irr * (1. / 1e6) / s_meta['AREA']

        years = [x for x in range(1991, 2021)]

        flow_periods = []
        max_len = 12
        rr = [x for x in range(1, 13)]
        for n in range(1, max_len + 1):
            for i in range(max_len):
                per = rr[i: i + n]
                if len(per) == 1:
                    per = [per[0], per[0]]
                per = tuple(per)
                flow_periods.append(per)

        response_d = {}
        r_dct = {}

        for q_win in flow_periods:

            if spec_time:
                if q_win != spec_time:
                    continue

            corr = (0, 0.0)
            key_ = '{}-{}'.format(q_win[0], q_win[-1])
            r_dct[key_] = []
            q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
            q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])

            for lag in offsets:
                if lag < q_win[-1] - q_win[0] + 1:
                    r_dct[key_].append(np.nan)
                    continue
                dates = [(date(y, q_win[-1], monthrange(y, q_win[-1])[1]) + rdlt(months=-lag),
                          date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]

                etr = np.array([df['gm_etr'][d[0]: d[1]].sum() for d in dates])
                ppt = np.array([df['gm_ppt'][d[0]: d[1]].sum() for d in dates])
                ai = etr - ppt
                lr = linregress(ai, q)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                qres = q - (b * ai + inter)
                r_dct[key_].append(r)
                if abs(r) > corr[1] and p < 0.05:
                    corr = (lag, abs(r))

                    response_d[key_] = {'q_window': q_win,
                                        'inter': inter,
                                        'b': b,
                                        'lag': lag,
                                        'r': r,
                                        'p': p,
                                        'irr_frac': irr_frac,
                                        'q_data': list(q),
                                        'ai_data': list(ai),
                                        'qres_data': list(qres)}

        windows[sid] = {**response_d, **s_meta}

    with open(out_json, 'w') as f:
        json.dump(windows, f, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/impacts'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/impacts'

    clim_dir = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')
    i_json = os.path.join(root, 'station_metadata', 'station_metadata.json')

    analysis_d = os.path.join(root, 'gridmet_analysis', 'analysis')

    for m in range(1, 13):
        print('month', m)
        clim_resp = os.path.join(root, 'gridmet_analysis', 'analysis',
                                 'climate_q_{}.json'.format(m))
        if not os.path.exists(clim_resp):
            climate_flow_correlation(in_json=i_json, climate_dir=clim_dir,
                                     out_json=clim_resp, spec_time=(m, m))

# ========================= EOF ====================================================================
