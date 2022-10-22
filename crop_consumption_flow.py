import os
import json
from datetime import date
from calendar import monthrange
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import LinearModel
import numpy as np
from scipy.stats.stats import linregress

from gage_data import hydrograph
from utils.gage_lists import VERIFIED_IRRIGATED_HYDROGRAPHS
from utils.gage_lists import STATION_BASINS

from utils.uncertainty import BASIN_CC_ERR, BASIN_F1, QRES_ERR

SAMPLE_KWARGS = {'draws': 1000,
                 'tune': 5000,
                 'cores': None,
                 'chains': 4,
                 'init': 'advi+adapt_diag',
                 'progressbar': False,
                 'return_inferencedata': False}


def initial_impacts_test(clim_q_data, ee_series, out_jsn=None, cc_res=False):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}

    with open(clim_q_data, 'r') as f:
        clim_q_data = json.load(f)

    for sid, v in clim_q_data.items():

        if sid not in VERIFIED_IRRIGATED_HYDROGRAPHS:
            continue

        s_meta = clim_q_data[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))

        cdf = hydrograph(_file)

        ct_tot += 1
        resid = None
        years = np.arange(1991, 2021)
        first = True
        staname = v['STANAME']

        for k, v in s_meta.items():

            if not isinstance(v, dict):
                continue

            clim_ai_p, lag, q_window, q, ai, qres = (v[s] for s in ['p', 'lag', 'q_window', 'q_data',
                                                                    'ai_data', 'qres_data'])

            if q_window[1] < 5:
                lookback = True
                q, ai, qres = q[1:], ai[1:], qres[1:]
            else:
                lookback = False

            cc_periods = []
            max_len = 7
            rr = np.arange(4, 11)
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

                cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])

                if cc_res:
                    etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in cc_dates])
                    ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in cc_dates])
                    aim = etr - ppt

                    lr = linregress(aim, cc)
                    b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                    resid = cc - (b * cc + inter)

                    lr = linregress(resid, qres)
                    b_norm = b * np.std(resid) / np.std(qres)
                    b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue

                else:
                    lr = linregress(cc, qres)
                    b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                    b_norm = b * np.std(cc) / np.std(qres)

                if p < 0.05:

                    if first:
                        print('\n', sid, staname)
                        first = False

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
                    if cc_res:
                        sig_stations[sid]['{}-{}'.format(cc_start, cc_end)]['cc_data'] = list(resid)

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)

    print('{} climate-sig, {} irrigated, {} irr imapacted periods, {} total basins'
          ''.format(ct, irr_ct, irr_sig_ct, ct_tot))
    return irr_sig_ct


def run_bayes_regression_cc_qres(traces_dir, stations, multiproc=0, overwrite=False):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    if multiproc > 0:
        pool = Pool(processes=multiproc)

    covered = []
    for sid, per, rec in diter:

        if sid in covered:
            continue

        if not (sid == '09058000' and per == '7-9'):
            continue

        rmse = BASIN_CC_ERR[STATION_BASINS[sid]]['rmse']
        bias = BASIN_CC_ERR[STATION_BASINS[sid]]['bias']
        irr_f1 = 1 - BASIN_F1[STATION_BASINS[sid]]
        cc_err = (irr_f1 + rmse - abs(bias))

        if rec['p'] > 0.05:
            continue

        covered.append(sid)

        if not multiproc:
            bayes_linear_regression_cc_qres(sid, rec, per, cc_err, bias, traces_dir, 4, overwrite)
        else:
            pool.apply_async(bayes_linear_regression_cc_qres, args=(sid, rec, per,
                                                                    cc_err, bias, traces_dir, 1, overwrite))
        exit()

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_linear_regression_cc_qres(station, records, period, cc_err, bias, trc_dir, cores, overwrite):
    try:
        cc = np.array(records['cc_data']) * (1 + bias)
        qres = np.array(records['qres_data'])

        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001

        qres_err = QRES_ERR * np.ones_like(qres)
        cc_err = cc_err * np.ones_like(cc)

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        model_dir = os.path.join(trc_dir, 'cc_qres')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        save_model = os.path.join(model_dir, '{}_cc_{}_q_{}.model'.format(station,
                                                                          period,
                                                                          records['q_window']))
        save_data = save_model.replace('.model', '.data')
        dct = {'x': list(cc),
               'y': list(qres),
               'x_err': list(cc_err),
               'y_err': list(qres_err),
               'xvar': 'cc',
               'yvar': 'qres',
               }
        with open(save_data, 'w') as fp:
            json.dump(dct, fp, indent=4)

        if os.path.isfile(save_model) and not overwrite:
            print('{} exists'.format(save_model))
            return None

        else:
            start = datetime.now()
            print('\n=== sampling qres {} at {}, p = {:.3f}, err: {:.3f}, bias: {} ======='.format(
                records['q_window'], station, records['p'], cc_err[0], bias))

        model = LinearModel()

        model.fit(cc, qres, qres_err, cc_err,
                  save_model=save_model,
                  sample_kwargs=sample_kwargs)
        delta = (datetime.now() - start).seconds / 60.
        print('sampled in {} minutes'.format(delta))

    except Exception as e:
        print(e, station, period)


def bayes_write_significant_cc_qres(metadata, trc_dir, out_json, update=False):
    with open(metadata, 'r') as f:
        stations = json.load(f)

    if update:
        with open(out_json, 'r') as f:
            out_meta = json.load(f)
    else:
        out_meta = {}

    for i, (station, data) in enumerate(stations.items()):

        if not update:
            out_meta[station] = data

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:
            records = data[period]

            saved_model = os.path.join(trc_dir, 'cc_qres',
                                       '{}_cc_{}_q_{}.model'.format(station, period, records['q_window']))

            if os.path.exists(saved_model):
                try:
                    with open(saved_model, 'rb') as buff:
                        mdata = pickle.load(buff)
                        model, trace = mdata['model'], mdata['trace']
                except Exception as e:
                    print(e)
                    continue
            else:
                continue

            try:
                vars_ = ['slope', 'inter']
                summary = az.summary(trace, hdi_prob=0.95, var_names=vars_)
                d = {'mean': summary['mean'],
                     'hdi_2.5%': summary['hdi_2.5%'],
                     'hdi_97.5%': summary['hdi_97.5%'],
                     'model': saved_model}

                if not update:
                    out_meta[station][period] = data[period]

                out_meta[station][period]['cc_qres'] = d
                if np.sign(d['hdi_2.5%']) == np.sign(d['hdi_97.5%']):
                    print('{}, {}, {}, {} mean: {:.2f}; hdi {:.2f} to {:.2f}'.format(station,
                                                                                     data['STANAME'],
                                                                                     period,
                                                                                     'cc_qres',
                                                                                     d['mean'],
                                                                                     d['hdi_2.5%'],
                                                                                     d['hdi_97.5%']))
                else:
                    print(station, data['STANAME'], period)

            except ValueError as e:
                print(station, e)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
