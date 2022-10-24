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

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from utils.uncertainty import BASIN_CC_ERR, BASIN_F1, QRES_ERR, PPT_ERR, ETR_ERR

SAMPLE_KWARGS = {'draws': 1000,
                 'tune': 5000,
                 'cores': None,
                 'chains': 4,
                 'init': 'advi+adapt_diag',
                 'progressbar': False,
                 'return_inferencedata': False}


def initial_trends_test(in_json, out_json, month, plot_dir=None):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    regressions, counts = {}, None

    for station, records in stations.items():

        try:
            q = np.array(records['q'])
        except KeyError:
            continue

        qres = np.array(records['qres'])
        ai = np.array(records['ai'])

        cc = np.array(records['cc_month'])
        aim = np.array(records['ai_month'])
        etr_m = np.array(records['etr_month'])
        etr = np.array(records['etr'])
        ppt_m = np.array(records['ppt_month'])
        ppt = np.array(records['ppt'])
        irr = np.array(records['irr'])

        years = np.array(records['years'])

        regression_combs = [(years, cc, 'time_cc'),
                            (years, qres, 'time_qres'),
                            (years, ai, 'time_ai'),
                            (years, aim, 'time_aim'),
                            (years, q, 'time_q'),
                            (years, etr, 'time_etr'),
                            (years, ppt, 'time_ppt'),
                            (years, etr_m, 'time_etrm'),
                            (years, ppt_m, 'time_pptm'),
                            (years, irr, 'time_irr')]

        if not counts:
            counts = {k[2]: [0, 0] for k in regression_combs}

        regressions[station] = records

        for x, y, subdir in regression_combs:

            if month not in range(4, 11) and subdir in ['time_irr', 'time_cc']:
                continue

            if subdir == 'time_q':
                mk_test = mk.hamed_rao_modification_test(y)
                y_pred = x * mk_test.slope + mk_test.intercept
                mk_slope_std = mk_test.slope * np.std(x) / np.std(y)
                p = mk_test.p
                if p < 0.05:
                    if mk_slope_std > 0:
                        counts[subdir][1] += 1
                    else:
                        counts[subdir][0] += 1
                    regressions[station][subdir] = {'test': 'mk',
                                                    'b': mk_slope_std,
                                                    'p': p, 'rsq': r}

            else:
                lr = linregress(x, y)
                b, inter, r, p = lr.slope, lr.intercept.item(), lr.rvalue, lr.pvalue
                y_pred = x * b + inter
                b_norm = b * np.std(x) / np.std(y)
                if p < 0.05:
                    if b_norm > 0:
                        counts[subdir][1] += 1
                    else:
                        counts[subdir][0] += 1
                    regressions[station][subdir] = {'test': 'ols',
                                                    'b': b,
                                                    'inter': inter,
                                                    'p': p,
                                                    'rsq': r,
                                                    'b_norm': b_norm}

            if plot_dir:
                d = os.path.join(plot_dir, str(month), subdir)

                if not os.path.exists(d):
                    os.makedirs(d)

                plt.scatter(x, y)
                yvar = subdir.split('_')[1]
                if yvar != 'q':
                    plt.plot(x, y_pred)
                plt.xlabel('time')
                plt.ylabel(yvar)
                desc = '{} {} {} {} yrs {}'.format(month, station, yvar, len(x), records['STANAME'])
                plt.suptitle(desc)
                f = os.path.join(d, '{}.png'.format(desc))
                plt.savefig(f)
                plt.close()

    print('\n {}'.format(in_json))
    pprint(counts)
    print(sum([np.array(v).sum() for k, v in counts.items()]))

    with open(out_json, 'w') as f:
        json.dump(regressions, f, indent=4, sort_keys=False)


def run_bayes_regression_trends(traces_dir, stations, month, multiproc=0, overwrite=False):
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

        rmse = BASIN_CC_ERR[rec['basin']]['rmse']
        bias = BASIN_CC_ERR[rec['basin']]['bias']
        irr_f1 = 1 - BASIN_F1[rec['basin']]
        cc_err = (irr_f1 + rmse - abs(bias))

        covered.append(sid)

        if not multiproc:
            bayes_linear_regression_trends(sid, rec, float(cc_err), bias, traces_dir, month, 4, overwrite)
        else:
            pool.apply_async(bayes_linear_regression_trends, args=(sid, rec,
                                                                   float(cc_err), bias, traces_dir, month, 1,
                                                                   overwrite))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_linear_regression_trends(station, records, cc_err, bias, trc_dir, month, cores, overwrite):
    try:
        cc = np.array(records['cc_month']) * (1 + bias)
        qres = np.array(records['qres'])
        ai = np.array(records['ai'])
        irr = np.array(records['irr'])
        years = np.array(records['years'])

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        cc_err = np.ones_like(cc) * cc_err
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001

        qres_err = np.ones_like(qres) * QRES_ERR

        years = (years - years.min()) / (years.max() - years.min()) + 0.001

        regression_combs = [(years, cc, None, cc_err),
                            (years, qres, None, qres_err),
                            (years, ai, None, qres_err),
                            (years, irr, None, err)]

        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_irr']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, y, x_err, y_err), subdir in zip(regression_combs, trc_subdirs):

            if subdir not in records.keys():
                continue
            if month not in range(4, 11) and subdir == 'time_cc':
                continue
            if records[subdir]['p'] > 0.05:
                continue

            model_dir = os.path.join(trc_dir, subdir)
            if len(os.listdir(model_dir)) > 3:
                continue

            save_model = os.path.join(model_dir, '{}_q_{}.model'.format(station, month))
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
                print('\n=== sampling {} len {} {} at {}, p = {:.3f}, err: {:.3f}, bias: {} ===='.format(subdir,
                                                                                                         len(x),
                                                                                                         month,
                                                                                                         station,
                                                                                                         records[
                                                                                                             subdir][
                                                                                                             'p'],
                                                                                                         cc_err[0],
                                                                                                         bias))

                model = LinearModel()

                model.fit(x, y, y_err, x_err,
                          save_model=save_model,
                          sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, month)


def bayes_write_significant_trends(metadata, trc_dir, out_json, month, update=False):
    with open(metadata, 'r') as f:
        stations = json.load(f)

    out_meta = {}

    trc_subdirs = ['time_cc', 'time_qres', 'time_ai']

    for i, (station, data) in enumerate(stations.items()):

        out_meta[station] = {month: data}

        for subdir in trc_subdirs:

            saved_model = os.path.join(trc_dir, subdir,
                                       '{}_q_{}.model'.format(station, month))

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
                                                                                 month,
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
