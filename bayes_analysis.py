import os
import sys
import json
import pickle
from multiprocessing import Pool

import numpy as np
import arviz as az
from bayes_models import LinearRegressionwithErrors, LinearModel
import bayes_models
from station_lists import STATION_BASINS
# suppress pymc3 FutureWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# temporary hack to open pickled model with renamed module
sys.modules['linear_regression_errors'] = bayes_models

BASIN_CC_ERR = {'missouri': {'rmse': 0.19, 'bias': 0.06},
                'colorado': {'rmse': 0.28, 'bias': -0.14},
                'columbia': {'rmse': 0.23, 'bias': -0.02}}

BASIN_F1 = {'missouri': 0.8985,
            'colorado': 0.8345,
            'columbia': 0.8649}

SAMPLE_KWARGS = {'draws': 500,
                 'tune': 5000,
                 'target_accept': 0.9,
                 'cores': None,
                 'init': 'advi+adapt_diag',
                 'chains': 4,
                 'n_init': 50000,
                 'progressbar': False,
                 'return_inferencedata': False}


def standardize(arr):
    arr = (arr - arr.mean()) / arr.std()
    return arr


def magnitude(arr):
    mag = np.ceil(np.log10(abs(arr)))
    return 10 ** mag


def run_bayes_regression_trends(traces_dir, stations, multiproc=False):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    if multiproc:
        pool = Pool(processes=30)

    covered = []
    for sid, rec in stations.items():

        if sid in covered:
            continue

        covered.append(sid)

        if not multiproc:
            bayes_linear_regression_trends(sid, rec, float(qres_err_),
                                           float(cc_err_), traces_dir, 4)
        else:
            pool.apply_async(bayes_linear_regression_trends, args=(sid, rec, float(qres_err_),
                                                                   float(cc_err_), traces_dir, 1))

    if multiproc:
        pool.close()
        pool.join()


def bayes_linear_regression_trends(station, records, qres_err, cc_err, trc_dir, cores):
    try:
        cc = np.array(records['cc_data'])
        qres = np.array(records['qres_data'])
        ai = np.array(records['ai_data'])
        q = np.array(records['q_data'])

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        q = (q - q.min()) / (q.max() - q.min()) + 0.001
        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
        years = (np.linspace(0, 1, len(qres)) + 0.001).reshape(1, -1) + 0.001

        qres_err = qres_err * np.ones_like(qres) * 0.5
        cc_err *= 0.5

        regression_combs = [(years, cc, None, cc_err),
                            (years, qres, None, qres_err),
                            (years, ai, None, qres_err),
                            (years, q, None, qres_err * 0.00001)]

        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_q']

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
            if os.path.isfile(save_model):
                print('{} exists'.format(save_model))
                continue
            else:
                print('\n==================== sampling {} q {} at {}, p = {:.3f} ======================='.format(
                    subdir, records['q_window'], station, records[subdir]['p']))

            model = LinearModel()

            model.fit(x, y, y_err, x_err,
                      save_model=save_model,
                      sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, records['q_window'])


def run_bayes_regression_cc_qres(traces_dir, stations, accuracy, multiproc=False):
    qres_err_ = 0.17 / 2.
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    with open(accuracy, 'r') as f:
        accuracy = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    if multiproc:
        pool = Pool(processes=30)

    covered = []
    for sid, per, rec in diter:

        if sid in covered:
            continue

        rmse = BASIN_CC_ERR[STATION_BASINS[sid]]['rmse']
        bias = BASIN_CC_ERR[STATION_BASINS[sid]]['bias']
        irr_f1 = 1 - BASIN_F1[STATION_BASINS[sid]]
        cc_err = (irr_f1 + rmse - abs(bias)) / 2

        if rec['p'] > 0.05:
            continue

        covered.append(sid)

        if not multiproc:
            bayes_linear_regression_cc_qres(sid, rec, per, qres_err_,
                                            cc_err, bias, traces_dir, 4)
        else:
            pool.apply_async(bayes_linear_regression_cc_qres, args=(sid, rec, per, qres_err_,
                                                                    cc_err, bias, traces_dir, 1))

    if multiproc:
        pool.close()
        pool.join()


def bayes_linear_regression_cc_qres(station, records, period, qres_err, cc_err, bias, trc_dir, cores):
    try:
        cc = np.array(records['cc_data']) * (1 + bias)
        qres = np.array(records['qres_data'])

        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001

        qres_err = qres_err * np.ones_like(qres)

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        model_dir = os.path.join(trc_dir, 'cc_qres')
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        save_model = os.path.join(model_dir, '{}_cc_{}_q_{}.model'.format(station,
                                                                          period,
                                                                          records['q_window']))
        if os.path.isfile(save_model):
            print('{} exists'.format(save_model))
            return None

        else:
            print('\n=== sampling q {} at {}, p = {:.3f}, err: {:.3f}, bias: {} ======='.format(
                records['q_window'], station, records['p'], cc_err * 2, bias))

        model = LinearRegressionwithErrors()

        model.fit(cc, qres, cc_err, qres_err,
                  save_model=save_model,
                  sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, period)


def bayes_write_significant_trends(metadata, trc_dir, out_json, update=False):
    with open(metadata, 'r') as f:
        stations = json.load(f)

    if update:
        with open(out_json, 'r') as f:
            out_meta = json.load(f)
    else:
        out_meta = {}

    # trc_subdirs = ['time_cc', 'time_q', 'time_qres', 'time_ai']
    trc_subdirs = ['time_q', 'time_qres', 'time_ai']

    for i, (station, data) in enumerate(stations.items()):

        period = data['q_window']

        if not update:
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
                summary = az.summary(trace, hdi_prob=0.95, var_names=['slope'], coords={'chain': chain_idx})
                d = {'mean': summary.iloc[0]['mean'],
                     'hdi_2.5%': summary.iloc[0]['hdi_2.5%'],
                     'hdi_97.5%': summary.iloc[0]['hdi_97.5%'],
                     'model': saved_model}

                if not update:
                    out_meta[station][subdir] = {}

                out_meta[station][subdir] = d

                print('{} mean: {:.2f}; hdi {:.2f} to {:.2f}'.format(subdir,
                                                                     d['mean'],
                                                                     d['hdi_2.5%'],
                                                                     d['hdi_97.5%']))
            except ValueError as e:
                print(station, e)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)


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

            # individual chains may diverge; drop them?
            diverge = trace.diverging.reshape(4, -1)
            diverge_sum = diverge.sum(axis=1)
            div_chain = np.array(diverge_sum, dtype=float) / (np.ones_like(diverge_sum) * diverge.shape[1])
            drop_chain = div_chain < 0.1
            chain_idx = [i for i, x in enumerate(drop_chain) if x]

            try:
                summary = az.summary(trace, hdi_prob=0.95, var_names=['slope'], coords={'chain': chain_idx})
                d = {'mean': summary.iloc[0]['mean'],
                     'hdi_2.5%': summary.iloc[0]['hdi_2.5%'],
                     'hdi_97.5%': summary.iloc[0]['hdi_97.5%'],
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


def count_bayes_tests(_dir):
    cc_err_ = '0.233'
    qres_err_ = '0.17'

    target_trends, target_cc = 0, 0
    trends_models, cc_models = 0, 0
    missing_trends, missing_cc = {}, {}
    insig_trends, insig_cc = 0, 0

    for m in range(1, 13):
        _state = 'm_{}_cc_{}_qreserr_{}'.format(m, str(cc_err_), str(qres_err_))

        _trace_dir = os.path.join(_dir, 'traces', _state)
        _analysis_d = os.path.join(_dir, 'analysis')
        _monthly_json = os.path.join(_analysis_d, 'trends_{}.json'.format(m))

        with open(_monthly_json, 'r') as f:
            stations = json.load(f)

        for sid, rec in stations.items():
            trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_q']

            for subdir in trc_subdirs:
                if rec[subdir]['p'] > 0.05:
                    insig_trends += 1
                    continue
                else:
                    target_trends += 1
                    model_path = os.path.join(_trace_dir, subdir, '{}_q_{}.model'.format(sid, rec['q_window']))

                    if os.path.exists(model_path):
                        trends_models += 1

                    else:
                        if m in missing_trends.keys():
                            missing_trends[m].append(model_path)
                        else:
                            missing_trends[m] = [model_path]

        _cc_json = os.path.join(_analysis_d, 'qres_cc_{}.json'.format(m))

        with open(_cc_json, 'r') as f:
            stations = json.load(f)

        diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
        diter = [i for ll in diter for i in ll]

        for sid, per, rec in diter:

            if rec['p'] > 0.05:
                insig_cc += 1
                continue

            target_cc += 1
            model_path = os.path.join(_trace_dir, 'cc_qres',
                                      '{}_cc_{}_q_{}.model'.format(sid, per, rec['q_window']))
            if os.path.exists(model_path):
                cc_models += 1
            else:
                if m in missing_cc.keys():
                    missing_cc[m].append(model_path)
                else:
                    missing_cc[m] = [model_path]
    pass


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS/gages/gridmet_analysis'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages/gridmet_analysis'

    acc_json = os.path.join(root, 'watershed_accuracy.json')
    var_ = 'cc'

    # count_bayes_tests(root)
    ct = 0
    for m in range(1, 13):

        print('\n\n month {} \n\n'.format(m))
        state = 'm_{}_cc_qres'.format(m)

        # trace_dir = os.path.join(root, 'traces', var_, state)
        trace_dir = os.path.join(root, 'traces_basin_acc', var_, state)
        analysis_d = os.path.join(root, 'analysis')

        monthly_json = os.path.join(analysis_d, 'trends_{}.json'.format(m))

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        # run_bayes_regression_trends(trace_dir, monthly_json, multiproc=True)

        cc_res = False
        if cc_res:
            f_json = os.path.join(analysis_d, 'qres_ccres_{}.json'.format(m))
        else:
            f_json = os.path.join(analysis_d, 'qres_cc_{}.json'.format(m))

        run_bayes_regression_cc_qres(trace_dir, f_json, acc_json, multiproc=True)
        # monthly_data = os.path.join(analysis_d, 'trends_{}.json'.format(m))
        # o_json = os.path.join(analysis_d, 'bayes_trend_{}.json'.format(m))
        # bayes_write_significant_trends(monthly_data, trace_dir, o_json, update=False)

        o_json = os.path.join(analysis_d, 'bayes_{}_qres_{}_acc.json'.format(var_, m))
        # bayes_write_significant_cc_qres(f_json, trace_dir, o_json, update=False)
    pass
# ========================= EOF ====================================================================
