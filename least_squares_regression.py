import os
import json
from datetime import date
from pprint import pprint
from calendar import monthrange
from dateutil.relativedelta import relativedelta as rdlt

import numpy as np
from scipy.stats.stats import linregress
import matplotlib.pyplot as plt

from hydrograph import hydrograph
from gage_analysis import EXCLUDE_STATIONS


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

        if sid in EXCLUDE_STATIONS:
            continue

        # if sid != '06036650':
        #     continue

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


def test_cc_qres_lr(clim_q_data, ee_series, out_jsn=None):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}

    with open(clim_q_data, 'r') as f:
        clim_q_data = json.load(f)

    for sid, v in clim_q_data.items():

        if sid != '06016000':
            continue

        s_meta = clim_q_data[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))

        try:
            cdf = hydrograph(_file)
        except FileNotFoundError as e:
            print(sid, e)

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        print('\n', sid, v['STANAME'])

        for k, v in s_meta.items():
            if not isinstance(v, dict):
                continue

            clim_ai_p, lag, q_window, q, ai, qres = (v[s] for s in ['p', 'lag', 'q_window', 'q_data',
                                                                    'ai_data', 'qres_data'])

            if clim_ai_p > 0.05:
                continue

            if q_window[1] < 5:
                lookback = True
                q, ai, qres = q[1:], ai[1:], qres[1:]
            else:
                lookback = False

            if v['irr_frac'] < 0.001:
                continue
            else:
                irr_ct += 1

            cc_periods = []
            max_len = 7
            rr = [x for x in range(4, 11)]
            for n in range(1, max_len + 1):
                for i in range(max_len - n):
                    per = rr[i: i + n]
                    if len(per) == 1:
                        per = [per[0], per[0]]
                    per = tuple(per)
                    cc_periods.append(per)

            for cc_period in cc_periods:
                cc_start, cc_end = cc_period[0], cc_period[-1]

                if lookback:
                    cc_dates = [(date(y - 1, cc_start, 1), date(y - 1, cc_end, monthrange(y - 1, cc_end)[1]))
                                for y in years[1:]]
                else:
                    cc_dates = [(date(y, cc_start, 1), date(y, cc_end, monthrange(y, cc_end)[1])) for y in years]

                if cc_dates[-1][1].month > q_window[0] and not lookback:
                    continue
                if cc_dates[-1][0].month > q_window[0] and not lookback:
                    continue

                cdf['cci'] = cdf['cc'] / cdf['irr']
                cc = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])

                lr = linregress(cc, qres)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                b_norm = b * np.std(cc) / np.std(qres)

                if p < 0.05:

                    irr_sig_ct += 1
                    print(cc_period, q_window, 'b', '{:.3f}'.format(b_norm), 'p', '{:.3f}'.format(p))

                    if sid not in sig_stations.keys():
                        sig_stations[sid] = {k: v for k, v in s_meta.items() if not isinstance(v, dict)}

                    sig_stations[sid].update({'{}-{}'.format(cc_start, cc_end): {'p': p,
                                                                                 'b': b_norm,
                                                                                 'r': r,
                                                                                 'lag': lag,
                                                                                 'q_window': k,
                                                                                 'q_data': list(q),
                                                                                 'ai_data': list(ai),
                                                                                 'cc_data': list(cc),
                                                                                 'qres_data': list(qres),
                                                                                 }})

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)

    print('{} climate-sig, {} irrigated, {} irr imapacted periods, {} total basins'
          ''.format(ct, irr_ct, irr_sig_ct, ct_tot))


def get_monthly_trends_data(metadata, ee_series, out_jsn=None):
    stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():

        s_meta = metadata[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))

        try:
            cdf = hydrograph(_file)
        except FileNotFoundError as e:
            print(sid, e)

        years = [x for x in range(1991, 2021)]

        print('\n', sid, v['STANAME'])

        for k, v in s_meta.items():

            if not isinstance(v, dict):
                continue

            clim_ai_p, lag, q_window, q, ai, qres = (v[s] for s in ['p', 'lag', 'q_window', 'q_data',
                                                                    'ai_data', 'qres_data'])

            q_start, q_end = q_window[0], q_window[-1]

            if v['irr_frac'] < 0.001:
                continue

            if clim_ai_p > 0.05:
                continue

            cc_dates = [(date(y, q_start, 1), date(y, q_end, monthrange(y, q_end)[1])) for y in years]
            cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])

            stations[sid] = {'{}'.format(q_start): {'q_window': k,
                                                    'q_data': list(q),
                                                    'ai_data': list(ai),
                                                    'cc_data': list(cc),
                                                    'qres_data': list(qres)}}

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(stations, f, indent=4, sort_keys=False)


def test_lr_trends(in_json, out_json, include_data=False):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    regressions = {}

    trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_q']
    counts = {k: [0, 0] for k in trc_subdirs}

    for station, period, records in diter:

        try:
            q = np.array(records['q_data'])
            qres = np.array(records['qres_data'])
            cc = np.array(records['cc_data'])
            ai = np.array(records['ai_data'])

            qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
            years = (np.linspace(0, 1, len(qres)) + 0.001) + 0.001

            regression_combs = [(years, cc),
                                (years, qres),
                                (years, ai),
                                (years, q)]

            if include_data:
                regressions[station] = records
            else:
                regressions[station] = {k: {} for k in trc_subdirs}

            for (x, y), subdir in zip(regression_combs, trc_subdirs):

                lr = linregress(x, y)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                b = b * np.std(x) / np.std(y)

                if p < 0.05:
                    if b > 0:
                        counts[subdir][1] += 1
                    else:
                        counts[subdir][0] += 1

                if not include_data:
                    regressions[station][subdir].update({'b': b, 'p': p, 'rsq': r})
                else:
                    regressions[station][subdir] = {'b': b, 'p': p, 'rsq': r}

        except Exception as e:
            print(e, station, period)

    print('\n {}'.format(in_json))
    pprint(counts)

    with open(out_json, 'w') as f:
        json.dump(regressions, f, indent=4, sort_keys=False)


def summarize_cc_qres(_dir, out_json, glob=None):
    dct = {}
    _files = [os.path.join(_dir, '{}_{}.json'.format(glob, m)) for m in range(1, 13)]

    for m, f in enumerate(_files, start=1):
        insig, sig = 0, 0
        with open(f, 'r') as fp:
            d_obj = json.load(fp)

        diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in d_obj.items()]
        diter = [i for ll in diter for i in ll]
        for k, cc, d in diter:
            if d['p'] > 0.05:
                insig += 1
                continue
            sig += 1
            if k not in dct.keys():
                dct[k] = {m: {cc: d['b']}}
            elif m not in dct[k].keys():
                dct[k][m] = {cc: d['b']}
            else:
                dct[k][m].update({cc: d['b']})

    with open(out_json, 'w') as fp:
        json.dump(dct, fp, indent=4)


def summarize_trends(_dir):
    _files = [os.path.join(_dir, 'linear_regressions_{}.json'.format(m)) for m in range(1, 13)]

    for var in ['time_cc', 'time_qres', 'time_ai', 'time_q']:
        dct = {}
        for m, f in enumerate(_files, start=1):

            if var == 'time_cc':
                if m > 10 or m < 4:
                    continue

            insig, sig = 0, 0
            with open(f, 'r') as fp:
                d_obj = json.load(fp)

            diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in d_obj.items()]
            diter = [i for ll in diter for i in ll]

            for k, v, d in diter:
                if v != var:
                    continue
                if d['p'] > 0.05:
                    insig += 1
                    continue
                sig += 1
                slope = d['b']
                if np.isnan(slope):
                    continue
                if k not in dct.keys():
                    dct[k] = {m: {v: slope}}
                elif m not in dct[k].keys():
                    dct[k][m] = {v: slope}
                else:
                    dct[k][m].update({v: slope})

            print(var, 'month ', m, 'sig', sig, 'insig', insig)

        out_json = os.path.join(_dir, '{}_summary.json'.format(var))
        with open(out_json, 'w') as fp:
            json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')

    clim_dir = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')
    i_json = os.path.join(root, 'station_metadata/station_metadata.json')
    fig_dir_ = os.path.join(root, 'figures/clim_q_correlations')

    analysis_d = os.path.join(root, 'gridmet_analysis', 'analysis')

    for m in range(1, 13):
        clim_resp = os.path.join(root, 'gridmet_analysis', 'analysis',
                                 'climate_q_{}.json'.format(m))
        if not os.path.exists(clim_resp):
            climate_flow_correlation(in_json=i_json, climate_dir=clim_dir,
                                     out_json=clim_resp, spec_time=(m, m))

        f_json = os.path.join(analysis_d, 'qres_cc_{}.json'.format(m))
        if not os.path.join(f_json):
            test_cc_qres_lr(clim_resp, ee_data, out_jsn=f_json)

        monthly_json = os.path.join(analysis_d, 'monthly_{}.json'.format(m))
        if not os.path.exists(monthly_json):
            get_monthly_trends_data(clim_resp, ee_data, out_jsn=monthly_json)

        trends_json = os.path.join(analysis_d, 'trends_{}.json'.format(m))
        # if not os.path.exists(trends_json):
        test_lr_trends(monthly_json, trends_json, include_data=True)

        cc_qres_summ = os.path.join(analysis_d, 'cc_qres_summary_.json')
        # summarize_cc_qres(analysis_d, cc_qres_summ, glob='qres_cc')

# ========================= EOF ====================================================================
