import os
import json
from datetime import datetime, date
from calendar import monthrange
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import BiVarLinearModel
import numpy as np
from scipy.stats.stats import linregress
import warnings

from gage_data import hydrograph

from utils.error_estimates import BASIN_CC_ERR, BASIN_IRRMAPPER_F1, BASIN_PRECIP_RMSE, ETR_ERR

warnings.simplefilter(action='ignore', category=FutureWarning)


def initial_impacts_test(clim_q_data, ee_series, out_jsn, month, cc_res=False):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}

    with open(clim_q_data, 'r') as f:
        clim_q_data = json.load(f)

    for sid, v in clim_q_data.items():

        s_meta = clim_q_data[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))

        cdf = hydrograph(_file)

        ct_tot += 1
        resid = None
        first = True
        staname = v['STANAME']

        corr = (0, 0.0)

        q, ai, qres, basin, years = (v[s] for s in ['q', 'ai', 'qres', 'basin', 'years'])

        if month < 5:
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

            if cc_dates[-1][1].month > month and not lookback:
                continue
            if cc_dates[-1][0].month > month and not lookback:
                continue

            cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
            etr = np.array([cdf['gm_etr'][d[0]: d[1]].sum() for d in cc_dates])
            ppt = np.array([cdf['gm_ppt'][d[0]: d[1]].sum() for d in cc_dates])
            aim = etr - ppt

            if cc_res:
                lr = linregress(aim, cc)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                resid = cc - (b * aim + inter)

                lr = linregress(resid, qres)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                b_norm = b * np.std(resid) / np.std(qres)

            else:
                lr = linregress(cc, qres)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                b_norm = b * np.std(cc) / np.std(qres)

            if p < 0.05 and abs(r) > abs(corr[1]):

                corr = (cc_period, r)

                if first:
                    print('\n', sid, staname)
                    first = False

                irr_sig_ct += 1
                print('{} {} b: {:.3f}, r: {:.3f}, p: {:.3f}'.format(cc_period, month, b_norm, r, p))

                if sid not in sig_stations.keys():
                    sig_stations[sid] = {k: v for k, v in s_meta.items() if not isinstance(v, list)}

                sig_stations[sid].update({'{}-{}'.format(cc_start, cc_end): {'p': p,
                                                                             'b': b_norm,
                                                                             'r': r,
                                                                             'q_mo': month,
                                                                             'basin': basin,
                                                                             'cc': list(cc),
                                                                             'q': list(q),
                                                                             'ai': list(ai),
                                                                             'aim': list(aim),
                                                                             }})
                if cc_res:
                    sig_stations[sid]['{}-{}'.format(cc_start, cc_end)]['cc_data'] = list(resid)

    with open(out_jsn, 'w') as f:
        json.dump(sig_stations, f, indent=4, sort_keys=False)


def run_bayes_regression_cc_qres(traces_dir, stations, multiproc=0, overwrite=False, station=None):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    if multiproc > 0:
        pool = Pool(processes=multiproc)

    for sid, period, rec in diter:

        if station and sid != station:
            continue

        if not multiproc:
            bayes_linear_regression_cc_qres(sid, period, rec, traces_dir, 4, overwrite)
        else:
            pool.apply_async(bayes_linear_regression_cc_qres, args=(sid, period, rec, traces_dir, 1, overwrite))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_linear_regression_cc_qres(station, period, records, trc_dir, cores, overwrite):
    try:

        basin = records['basin']
        rmse = BASIN_CC_ERR[basin]['rmse']
        bias = BASIN_CC_ERR[basin]['bias']
        irr_f1 = 1 - BASIN_IRRMAPPER_F1[basin]
        cc_err = (irr_f1 + rmse - abs(bias))
        ppt_err, etr_err = BASIN_PRECIP_RMSE[basin], ETR_ERR
        qres_err = np.sqrt(ppt_err ** 2 + etr_err ** 2)

        month = records['q_mo']
        cc = np.array(records['cc']) * (1 + bias)
        q = np.array(records['q'])
        ai = np.array(records['ai'])

        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        q = (q - q.min()) / (q.max() - q.min()) + 0.001
        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001

        ai_err = qres_err * np.ones_like(ai)
        cc_err = cc_err * np.ones_like(cc)
        q_err = np.ones_like(ai_err) * 0.08

        if not os.path.isdir(trc_dir):
            os.makedirs(os.path.join(trc_dir, 'model'))
            os.makedirs(os.path.join(trc_dir, 'data'))
            os.makedirs(os.path.join(trc_dir, 'trace'))

        save_model = os.path.join(trc_dir, 'model', '{}_cc_{}_q_{}.model'.format(station, period, month))
        save_data = os.path.join(trc_dir, 'data', '{}_cc_{}_q_{}.data'.format(station, period, month))
        save_tracefig = os.path.join(trc_dir, 'trace', '{}_cc_{}_q_{}.png'.format(station, period, month))

        dct = {'x1': list(ai),
               'x2': list(cc),
               'y': list(q),
               'x1_err': list(cc_err),
               'x2_err': list(ai_err),
               'y_err': list(np.zeros_like(ai_err)),
               'x1_var': 'cc',
               'x2_var': 'cwd',
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
                month, station, records['p'], cc_err[0], bias))

        variable_names = {'x1_name': 'cwd_coeff',
                          'x2_name': 'iwu_coeff'}

        model = BiVarLinearModel()

        model.fit(ai, ai_err, cc, cc_err, q, q_err,
                  save_model=save_model,
                  var_names=variable_names,
                  figure=save_tracefig)
        delta = (datetime.now() - start).seconds / 60.
        print('sampled in {} minutes'.format(delta))

    except Exception as e:
        print(e, station)


def bayes_write_significant_cc_qres(metadata, trc_dir, out_json, month):
    with open(metadata, 'r') as f:
        stations = json.load(f)

        out_meta = {}

    for i, (station, data) in enumerate(stations.items()):

        out_meta[station] = data

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:

            saved_model = os.path.join(trc_dir, 'model', '{}_cc_{}_q_{}.model'.format(station, period, month))

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
                vars_ = ['cwd_coeff', 'iwu_coeff', 'inter']
                summary = az.summary(trace, hdi_prob=0.95, var_names=vars_)
                d = {'mean': summary['mean'].iwu_coeff,
                     'hdi_2.5%': summary['hdi_2.5%'].iwu_coeff,
                     'hdi_97.5%': summary['hdi_97.5%'].iwu_coeff,
                     'r_hat': summary['r_hat'].iwu_coeff,
                     'model': saved_model}

                out_meta[station][period] = data[period]

                out_meta[station][period]['cc_q'] = d
                if np.sign(d['hdi_2.5%']) == np.sign(d['hdi_97.5%']) and d['r_hat'] <= 1.2:
                    print('{}, {}, {}, {} mean: {:.2f}; hdi {:.2f} to {:.2f}   {:.3f}'.format(station,
                                                                                              data['STANAME'],
                                                                                              period,
                                                                                              'cc_q',
                                                                                              d['mean'],
                                                                                              d['hdi_2.5%'],
                                                                                              d['hdi_97.5%'],
                                                                                              d['r_hat']))
                else:
                    print(station, data['STANAME'], period)

            except ValueError as e:
                print(station, e)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
