import os
import json
from pprint import pprint
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import LinearModel
import numpy as np
from scipy.stats.stats import linregress
import pymannkendall as mk

from utils.gage_lists import VERIFIED_IRRIGATED_HYDROGRAPHS
from utils.gage_lists import STATION_BASINS

from utils.uncertainty import BASIN_CC_ERR, BASIN_F1, QRES_ERR, PPT_ERR, ETR_ERR

SAMPLE_KWARGS = {'draws': 1000,
                 'tune': 5000,
                 'cores': None,
                 'chains': 4,
                 'init': 'advi+adapt_diag',
                 'progressbar': False,
                 'return_inferencedata': False}


def initial_trends_test(in_json, out_json):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    regressions = {}

    trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_aim', 'time_q', 'time_etr', 'time_ppt', 'time_irr']
    counts = {k: [0, 0] for k in trc_subdirs}

    deltas = {'time_ai': [], 'time_q': [], 'time_cc': []}

    for station, period, records in diter:
        if station not in VERIFIED_IRRIGATED_HYDROGRAPHS:
            continue

        q = np.array(records['q'])
        qres = np.array(records['qres'])
        ai = np.array(records['ai'])

        cc = np.array(records['cc_month'])
        aim = np.array(records['ai_month'])
        etr = np.array(records['etr_month'])
        ppt = np.array(records['ppt_month'])
        irr = np.array(records['irr'])

        years = (np.linspace(0, 1, len(qres)) + 0.001) + 0.001

        regression_combs = [(years, cc),
                            (years, qres),  # deceptive as this is only looking at same-month cc and q
                            (years, ai),
                            (years, aim),
                            (years, q),
                            (years, etr),
                            (years, ppt),
                            (years, irr)]

        regressions[station] = records

        for (x, y), subdir in zip(regression_combs, trc_subdirs):

            if subdir == 'time_q':
                mk_test = mk.hamed_rao_modification_test(y)
                mk_slope_std = mk_test.slope * np.std(x) / np.std(y)
                regressions[station][subdir] = {'test': 'mk',
                                                'b': mk_slope_std,
                                                'p': mk_test.p, 'rsq': r}

            else:
                lr = linregress(x, y)
                b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue
                b = b * np.std(x) / np.std(y)
                regressions[station][subdir] = {'test': 'ols',
                                                'b': b, 'p': p, 'rsq': r}

            if subdir in deltas.keys():
                deltas[subdir].append(b)

    print('\n {}'.format(in_json))
    pprint(counts)
    print(sum([np.array(v).sum() for k, v in counts.items()]))

    with open(out_json, 'w') as f:
        json.dump(regressions, f, indent=4, sort_keys=False)


def run_bayes_regression_trends(traces_dir, stations, multiproc=0, overwrite=False):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    if multiproc > 0:
        pool = Pool(processes=multiproc)

    covered = []
    for sid, rec in stations.items():

        if sid in covered:
            continue

        rmse = BASIN_CC_ERR[STATION_BASINS[sid]]['rmse']
        bias = BASIN_CC_ERR[STATION_BASINS[sid]]['bias']
        irr_f1 = 1 - BASIN_F1[STATION_BASINS[sid]]
        cc_err = (irr_f1 + rmse - abs(bias))

        covered.append(sid)

        if not multiproc:
            bayes_linear_regression_trends(sid, rec, float(cc_err), bias, traces_dir, 4, overwrite)
        else:
            pool.apply_async(bayes_linear_regression_trends, args=(sid, rec,
                                                                   float(cc_err), bias, traces_dir, 1, overwrite))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_linear_regression_trends(station, records, cc_err, bias, trc_dir, cores, overwrite):
    try:
        cc = np.array(records['cc_data']) * (1 + bias)
        qres = np.array(records['qres_data'])
        ai = np.array(records['ai_data'])
        ppt = np.array(records['ppt_data'])
        etr = np.array(records['etr_data'])

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        cc_err = np.ones_like(cc) * cc_err
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001

        qres_err = np.ones_like(qres) * QRES_ERR
        ppt_err = np.ones_like(qres) * PPT_ERR
        etr_err = np.ones_like(qres) * ETR_ERR

        years = np.linspace(0, 1, len(qres)) + 0.001

        regression_combs = [(years, cc, None, cc_err),
                            (years, qres, None, qres_err),
                            (years, ai, None, qres_err),
                            (years, ppt, None, ppt_err),
                            (years, etr, None, etr_err)]

        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_ppt', 'time_etr']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, y, x_err, y_err), subdir in zip(regression_combs, trc_subdirs):

            if int(records['q_window'][0]) not in range(4, 11) and subdir == 'time_cc':
                continue
            if records[subdir]['p'] > 0.05:
                continue

            save_model = os.path.join(trc_dir, subdir, '{}_q_{}.model'.format(station, records['q_window']))
            save_data = save_model.replace('.model', '.data')
            dct = {'x': list(x),
                   'y': list(y),
                   'x_err': None,
                   'y_err': list(y_err),
                   'xvar': 'time',
                   'yvar': subdir.split('_')[1],
                   }

            with open(save_data, 'w') as fp:
                json.dump(dct, fp, indent=4)

            if os.path.isfile(save_model) and not overwrite:
                print('{} exists'.format(save_model))
                continue

            else:
                print('\n=== sampling {} {} at {}, p = {:.3f}, err: {:.3f}, bias: {} ===='.format(subdir,
                                                                                                  records['q_window'],
                                                                                                  station,
                                                                                                  records[subdir]['p'],
                                                                                                  cc_err[0],
                                                                                                  bias))

            model = LinearModel()

            model.fit(x, y, y_err, x_err,
                      save_model=save_model,
                      sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, records['q_window'])


def bayes_write_significant_trends(metadata, trc_dir, out_json, update=False):
    with open(metadata, 'r') as f:
        stations = json.load(f)

    out_meta = {}

    trc_subdirs = ['time_cc', 'time_qres', 'time_ai']

    for i, (station, data) in enumerate(stations.items()):

        period = data['q_window']

        out_meta[station] = {period: data}

        for subdir in trc_subdirs:

            if update and subdir in out_meta[station][period].keys():
                continue

            saved_model = os.path.join(trc_dir, subdir,
                                       '{}_q_{}.model'.format(station, period))

            if os.path.exists(saved_model):
                try:
                    with open(saved_model, 'rb') as buff:
                        mdata = pickle.load(buff)
                        model, trace = mdata['model'], mdata['trace']
                except Exception as e:
                    print(e)
                    continue
            else:
                out_meta[station][subdir] = None
                continue

            chain_idx = [i for i in range(4)]

            try:
                summary = az.summary(trace, hdi_prob=0.95, var_names=['slope'])
                d = {'mean': summary.iloc[0]['mean'],
                     'hdi_2.5%': summary.iloc[0]['hdi_2.5%'],
                     'hdi_97.5%': summary.iloc[0]['hdi_97.5%'],
                     'model': saved_model}

                if not update:
                    out_meta[station][subdir] = {}

                out_meta[station][subdir] = d
                if np.sign(d['hdi_2.5%']) == np.sign(d['hdi_97.5%']):
                    print('{}, {}, {} mean: {:.2f}; hdi {:.2f} to {:.2f}'.format(station,
                                                                                 period,
                                                                                 subdir,
                                                                                 d['mean'],
                                                                                 d['hdi_2.5%'],
                                                                                 d['hdi_97.5%']))

            except ValueError as e:
                print(station, e)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
