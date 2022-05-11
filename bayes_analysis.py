import os
import sys
import json
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from bayes_models import LinearRegressionwithErrors, LinearModel
import bayes_models

# suppress pymc3 FutureWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# temporary hack to open pickled model with renamed module
sys.modules['linear_regression_errors'] = bayes_models


def standardize(arr):
    arr = (arr - arr.mean()) / arr.std()
    return arr


def magnitude(arr):
    mag = np.ceil(np.log10(abs(arr)))
    return 10**mag


def run_bayes_regression(traces_dir, stations, multiproc=False):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    if multiproc:
        pool = Pool(processes=30)

    for sid, per, rec in diter:

        if sid != '06018500' or per != '7-9':
            continue

        if not multiproc:
            bayes_linear_regression(sid, rec, per, float(qres_err_),
                                    float(cc_err_), trace_dir, 4)
        else:
            pool.apply_async(bayes_linear_regression, args=(sid, rec, per, float(qres_err_),
                                                            float(cc_err_), traces_dir, 1))

    if multiproc:
        pool.close()
        pool.join()


def bayes_linear_regression(station, records, period, qres_err, cc_err, trc_dir, cores):
    try:
        print('\n{}, p = {:.3f}'.format(station, records['res_sig']))

        cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
        qres = np.array(records['q_resid'])

        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
        years = (np.linspace(0, 1, len(qres)) + 0.001).reshape(1, -1) + 0.001

        qres_err = qres_err * np.ones_like(qres) * 0.5
        cc_err = cc_err * np.ones_like(cc) * 0.5

        sample_kwargs = {'draws': 500,
                         'tune': 5000,
                         'target_accept': 0.9,
                         'cores': cores,
                         'init': 'advi+adapt_diag',
                         'chains': 4,
                         'n_init': 50000,
                         'progressbar': False,
                         'return_inferencedata': False}

        regression_combs = [(cc, qres, cc_err, qres_err),
                            (years, cc, None, cc_err),
                            (years, qres, None, qres_err)]

        trc_subdirs = ['cc_qres', 'time_cc', 'time_qres']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, y, x_err, y_err), subdir in zip(regression_combs[:1], trc_subdirs[:1]):
            if subdir == 'time_cc':
                y = y[0, :]

            save_model = os.path.join(trc_dir, subdir, '{}_cc_{}_q_{}.model'.format(station,
                                                                                    period,
                                                                                    records['q_window']))
            if os.path.isfile(save_model):
                print('{} exists'.format(subdir))
                continue
            else:
                print('\nsampling {}'.format(subdir))

            if subdir == 'cc_qres':
                model = LinearRegressionwithErrors()
            else:
                model = LinearModel()

            model.fit(x, y, y_err, x_err,
                      save_model=save_model,
                      sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, period)


def bayes_sig_irr_impact(metadata, trc_dir, out_json, update=False):
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
            print('\n', station, period)

            trc_subdirs = ['cc_qres', 'time_cc', 'time_qres']

            for subdir in trc_subdirs:

                if update and subdir in out_meta[station][period].keys():
                    continue

                saved_model = os.path.join(trc_dir, subdir,
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

                if subdir == 'cc_qres':
                    # individual chains may diverge; drop them?
                    diverge = trace.diverging.reshape(4, -1)
                    diverge_sum = diverge.sum(axis=1)
                    div_chain = np.array(diverge_sum, dtype=float) / (np.ones_like(diverge_sum) * diverge.shape[1])
                    drop_chain = div_chain < 0.1
                    chain_idx = [i for i, x in enumerate(drop_chain) if x]
                else:
                    chain_idx = [i for i in range(4)]

                try:
                    summary = az.summary(trace, hdi_prob=0.95, var_names=['slope'], coords={'chain': chain_idx})
                    d = {'mean': summary.iloc[0]['mean'],
                         'hdi_2.5%': summary.iloc[0]['hdi_2.5%'],
                         'hdi_97.5%': summary.iloc[0]['hdi_97.5%'],
                         'model': saved_model}

                    if not update:
                        out_meta[station][period] = data[period]

                    out_meta[station][period][subdir] = d

                    print('{} mean: {:.2f}; hdi {:.2f} to {:.2f}'.format(subdir,
                                                                     d['mean'],
                                                                     d['hdi_2.5%'],
                                                                     d['hdi_97.5%']))
                except ValueError as e:
                    print(station, e)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    cc_err_ = '0.233'
    qres_err_ = '0.17'
    state = 'ccerr_{}_qreserr_{}'.format(str(cc_err_), str(qres_err_))
    trace_dir = os.path.join(root, 'bayes', 'traces', state)
    f_json = os.path.join(root, 'station_metadata', 'cci_impacted.json')

    if not os.path.exists(trace_dir):
        os.makedirs(trace_dir)

    # run_bayes_regression(trace_dir, f_json, multiproc=False)

    var = 'cci'
    o_fig = os.path.join(root, 'figures', 'slope_trace_{}'.format(var), state)
    if not os.path.exists(o_fig):
        os.makedirs(o_fig)
    o_json = os.path.join(root, 'station_metadata', 'cci_impacted_bayes_{}.json'.format(state))
    bayes_sig_irr_impact(f_json, trace_dir, o_json, update=True)

# ========================= EOF ====================================================================
