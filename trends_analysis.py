import os
import json
from pprint import pprint
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import LinearModel
import numpy as np
from scipy.stats.stats import linregress
from scipy.stats import anderson
import pymannkendall as mk
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

from utils.error_estimates import BASIN_CC_ERR, BASIN_IRRMAPPER_F1, BASIN_PRECIP_RMSE, ETR_ERR

SAMPLE_KWARGS = {'draws': 1000,
                 'tune': 5000,
                 'cores': None,
                 'chains': 4,
                 'init': 'advi+adapt_diag',
                 'progressbar': False,
                 'return_inferencedata': False}


def initial_trends_test(in_json, out_json, plot_dir=None, selectors=None):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    regressions, counts, ct = {}, None, 0

    for enu, (station, records) in enumerate(stations.items(), start=1):

        try:
            q = np.array(records['q'])
        except KeyError:
            continue

        month = records['q_mo']
        qres = np.array(records['qres'])
        ai = np.array(records['ai'])

        cc = np.array(records['cc_month'])
        try:
            ccres = np.array(records['ccres_month'])
        except KeyError:
            ccres = cc * np.zeros_like(cc)
        aim = np.array(records['ai_month'])
        etr_m = np.array(records['etr_month'])
        etr = np.array(records['etr'])
        ppt_m = np.array(records['ppt_month'])
        ppt = np.array(records['ppt'])
        irr = np.array(records['irr'])
        cci = cc / irr

        years = np.array(records['years'])

        regression_combs = [(years, cc, 'time_cc'),
                            (years, ccres, 'time_ccres'),
                            (years, qres, 'time_qres'),
                            (years, ai, 'time_ai'),
                            (years, aim, 'time_aim'),
                            (years, q, 'time_q'),
                            (years, etr, 'time_etr'),
                            (years, ppt, 'time_ppt'),
                            (years, etr_m, 'time_etrm'),
                            (years, ppt_m, 'time_pptm'),
                            (years, irr, 'time_irr'),
                            (years, cci, 'time_cci')]

        if not counts:
            counts = {k[2]: [0, 0] for k in regression_combs}

        regressions[station] = records

        for x, y, subdir in regression_combs:

            if selectors and subdir not in selectors:
                continue

            if month not in range(4, 11) and subdir in ['time_irr', 'time_cc', 'time_ccres', 'time_cci']:
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
                                                    'p': p}

            else:
                lr = linregress(x, y)
                b, inter, r, p = lr.slope, lr.intercept.item(), lr.rvalue, lr.pvalue
                y_pred = x * b + inter
                b_norm = b * np.std(x) / np.std(y)
                resid = y - (b * x + inter)
                ad_test = anderson(resid, 'norm').statistic.item()
                if ad_test < 0.05:
                    print('{} month {} failed normality test'.format(station, month))
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
                                                    'b_norm': b_norm,
                                                    'anderson': ad_test}

            if plot_dir:
                d = os.path.join(plot_dir, str(month), subdir)

                if not os.path.exists(d):
                    os.makedirs(d)

                yvar = subdir.split('_')[1]
                plt.scatter(x, y)
                if yvar == 'q':
                    lr = linregress(x, y)
                    b, inter, r, p = lr.slope, lr.intercept.item(), lr.rvalue, lr.pvalue
                    y_pred = x * b + inter

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


def run_bayes_regression_trends(traces_dir, stations, multiproc=0, overwrite=False, selectors=None):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations, 'r') as f:
        stations = json.load(f)

    if multiproc > 0:
        pool = Pool(processes=multiproc)

    for sid, rec in stations.items():

        if not multiproc:
            bayes_linear_regression_trends(sid, rec, traces_dir, 4, overwrite, selectors)
        else:
            pool.apply_async(bayes_linear_regression_trends, args=(sid, rec, traces_dir, 1, overwrite, selectors))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_linear_regression_trends(station, records, trc_dir, cores, overwrite, selectors=None):
    try:
        basin = records['basin']
        rmse = BASIN_CC_ERR[basin]['rmse']
        bias = BASIN_CC_ERR[basin]['bias']
        irr_f1 = 1 - BASIN_IRRMAPPER_F1[basin]
        cci_err = rmse - abs(bias)
        cc_err = irr_f1 + rmse - abs(bias)
        ppt_err, etr_err = BASIN_PRECIP_RMSE[basin], ETR_ERR
        qres_err = np.sqrt(ppt_err ** 2 + etr_err ** 2)

        month = records['q_mo']
        cc = np.array(records['cc_month']) * (1 + bias)
        ccres = np.array(records['ccres_month']) * (1 + bias)
        qres = np.array(records['qres'])
        ai = np.array(records['ai'])
        irr = np.array(records['irr'])
        cci = cc / irr
        years = np.array(records['years'])

        sample_kwargs = SAMPLE_KWARGS
        sample_kwargs['cores'] = cores

        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        cci = (cci - cci.min()) / (cci.max() - cci.min()) + 0.001
        ccres = (ccres - ccres.min()) / (ccres.max() - ccres.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
        irr = (irr - irr.min()) / (irr.max() - irr.min()) + 0.001

        cc_err = np.ones_like(cc) * cc_err
        cci_err = np.ones_like(cci) * cci_err
        qres_err = np.ones_like(qres) * qres_err
        irr_err = np.ones_like(qres) * irr_f1

        years = (years - years.min()) / (years.max() - years.min()) + 0.001

        regression_combs = [(years, cc, None, cc_err),
                            (years, qres, None, qres_err),
                            (years, ai, None, qres_err),
                            (years, irr, None, irr_err),
                            (years, ccres, None, cc_err),
                            (years, cci, None, cci_err)]

        trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_irr', 'time_ccres', 'time_cci']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, y, x_err, y_err), subdir in zip(regression_combs, trc_subdirs):

            if selectors and subdir not in selectors:
                continue
            if subdir == 'time_irr' and month != 8:
                continue
            if subdir not in records.keys():
                continue
            if month not in range(4, 11) and subdir == 'time_cc':
                continue
            if records[subdir]['p'] > 0.05:
                continue

            model_dir = os.path.join(trc_dir, subdir)

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
                print('\n=== sampling {} len {}, m {} at {}, '
                      'p = {:.3f}, err: {:.3f}, bias: {} ===='.format(subdir,
                                                                      len(x),
                                                                      month,
                                                                      station,
                                                                      records[subdir]['p'],
                                                                      cc_err[0],
                                                                      bias))

                model = LinearModel()

                model.fit(x, y, y_err, x_err,
                          save_model=save_model,
                          sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station)


def bayes_write_significant_trends(metadata, trc_dir, out_json, month, update_selectors=None):
    with open(metadata, 'r') as f:
        stations = json.load(f)

    out_meta = {}
    if update_selectors:
        with open(out_json, 'r') as f:
            out_meta = json.load(f)

    trc_subdirs = ['time_cci', 'time_ccres', 'time_cc', 'time_qres', 'time_ai', 'time_irr']

    for i, (station, data) in enumerate(stations.items()):

        if not update_selectors:
            out_meta[station] = {month: data}

        for subdir in trc_subdirs:

            if update_selectors and subdir not in update_selectors:
                if subdir not in out_meta[station].keys():
                    out_meta[station][subdir] = None
                continue

            model_dir = os.path.join(trc_dir, subdir)
            saved_model = os.path.join(model_dir, '{}_q_{}.model'.format(station, month))

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
                summary = az.summary(trace, hdi_prob=0.95, var_names=['slope', 'inter'])
                d = {'mean': summary['mean'].slope,
                     'hdi_2.5%': summary['hdi_2.5%'].slope,
                     'hdi_97.5%': summary['hdi_97.5%'].slope,
                     'model': saved_model}

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


def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
