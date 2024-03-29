import os
import json
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import BiVarLinearModel
import numpy as np
import warnings
import seaborn as sns

sns.set_style("dark")
sns.set_theme()
sns.despine()

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

from utils.error_estimates import BASIN_CC_ERR, BASIN_IRRMAPPER_F1, BASIN_PRECIP_RMSE, ETR_ERR, STUDY_EPT_ERROR


def run_bayes_multivariate_trends(traces_dir, stations_meta, multiproc=0, overwrite=False, stations=None, selector=None):
    if not os.path.exists(traces_dir):
        os.makedirs(traces_dir)

    with open(stations_meta, 'r') as f:
        stations_meta = json.load(f)

    if multiproc > 0:
        pool = Pool(processes=multiproc)

    for sid, rec in stations_meta.items():

        if stations and sid not in stations:
            continue

        if not multiproc:
            bayes_multivariate_trends(sid, rec, traces_dir, overwrite, selector)
        else:
            pool.apply_async(bayes_multivariate_trends, args=(sid, rec, traces_dir, overwrite, selector))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_multivariate_trends(station, records, trc_dir, overwrite, selector=None):
    try:
        basin = records['basin']
        rmse = BASIN_CC_ERR[basin]['rmse']
        bias = BASIN_CC_ERR[basin]['bias']
        ppt_err, etr_err = BASIN_PRECIP_RMSE[basin], ETR_ERR
        ai_err = np.sqrt(ppt_err ** 2 + etr_err ** 2)
        irr_f1 = 1 - BASIN_IRRMAPPER_F1[basin]
        cc_err = np.sqrt(irr_f1**2 + rmse**2 + STUDY_EPT_ERROR**2)

        month = records['q_mo']
        q = np.array(records['q'])
        cc = np.array(records['cc_month']) * (1 + bias)
        aim = np.array(records['ai_month'])
        ai = np.array(records['ai'])
        years = np.array(records['years'])

        q = (q - q.min()) / (q.max() - q.min()) + 0.001
        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        aim = (aim - aim.min()) / (aim.max() - aim.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        years_norm = (years - years.min()) / (years.max() - years.min()) + 0.001

        q_err = np.ones_like(q) * 0.08
        ai_err = np.ones_like(ai) * ai_err
        cc_err = np.ones_like(cc) * cc_err
        time_err = np.ones_like(ai) * 0.0001

        regression_combs = [(ai, ai_err, q, q_err),
                            (aim, ai_err, cc, cc_err)]

        trc_subdirs = ['time_q', 'time_cc']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, x_err, y, y_err), subdir in zip(regression_combs, trc_subdirs):

            if selector and selector != subdir:
                continue

            if subdir == 'time_cc' and month not in list(range(4, 11)):
                continue

            model_dir = os.path.join(trc_dir, subdir)
            save_model = os.path.join(model_dir, 'model', '{}_q_{}.model'.format(station, month))
            save_figure = os.path.join(model_dir, 'trace', '{}_q_{}.png'.format(station, month))
            save_data = os.path.join(model_dir, 'data', '{}_q_{}.data'.format(station, month))

            if os.path.exists(save_model) and not overwrite:
                print(os.path.basename(save_model), 'exists, skipping')
                continue

            else:
                dct = {'x': list(x),
                       'y': list(y),
                       'years': list(years_norm),
                       'x_err': list(x_err),
                       'y_err': list(y_err),
                       'xvar': 'cwd',
                       'xvar2': 'time',
                       'yvar': subdir.split('_')[1],
                       'model': save_model}

                with open(save_data, 'w') as fp:
                    json.dump(dct, fp, indent=4)

                print('\n=== sampling {} len {}, m {} at {}, '
                      ' ===='.format(subdir, len(x), month, station))

                variable_names = {'x1_name': 'time_coeff',
                                  'x2_name': 'cwd_coeff'}

                model = BiVarLinearModel()

                model.fit(years_norm, time_err, x, x_err, y, y_err, save_model=save_model,
                          figure=save_figure, var_names=variable_names)

    except Exception as e:
        print(e, station)


def summarize_multivariate_trends(metadata, trc_dir, out_json, month, update_selectors=None):

    div_, conv_ = 0, 0
    sp, sn = 0, 0

    with open(metadata, 'r') as f:
        stations = json.load(f)

    out_meta = {}
    if update_selectors:
        with open(out_json, 'r') as f:
            out_meta = json.load(f)

    for i, (station, data) in enumerate(stations.items()):

        if station not in out_meta.keys():
            out_meta[station] = {month: data}

        if not update_selectors:
            out_meta[station] = {month: data}

        trc_subdirs = ['time_q', 'time_cc']
        for subdir in trc_subdirs:

            if subdir == 'time_cc' and month not in range(4, 11):
                continue

            if update_selectors and subdir not in update_selectors:

                if subdir not in out_meta[station].keys():
                    out_meta[station][subdir] = None
                continue

            model_dir = os.path.join(trc_dir, subdir)
            saved_model = os.path.join(model_dir, 'model', '{}_q_{}.model'.format(station, month))

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
                summary = az.summary(trace, hdi_prob=0.95)
                beta = 'time_coeff' if 'time_coeff' in summary.index else 'slope'
                d = {'mean': summary['mean'][beta],
                     'hdi_2.5%': summary['hdi_2.5%'][beta],
                     'hdi_97.5%': summary['hdi_97.5%'][beta],
                     'r_hat': summary['r_hat'][beta],
                     'model': saved_model}

                if d['r_hat'] > 1.1:
                    div_ += 1

                out_meta[station][subdir] = d
                if np.sign(d['hdi_2.5%']) == np.sign(d['hdi_97.5%']) and d['r_hat'] <= 1.1:
                    conv_ += 1
                    if d['mean'] > 0:
                        sp += 1
                    else:
                        sn += 1
                    print('{}, {}, {} mean: {:.2f}; hdi {:.2f} to {:.2f}    rhat: {:.3f}'.format(station,
                                                                                                 month,
                                                                                                 subdir,
                                                                                                 d['mean'],
                                                                                                 d['hdi_2.5%'],
                                                                                                 d['hdi_97.5%'],
                                                                                                 d['r_hat']))

            except (ValueError, KeyError) as e:
                print(station, e, saved_model)

    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)

    return conv_, div_, sp, sn


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
