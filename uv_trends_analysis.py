import os
import json
import pickle
from multiprocessing import Pool

import arviz as az
from utils.bayes_models import TimeTrendModel
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore')

from utils.error_estimates import BASIN_CC_ERR, BASIN_IRRMAPPER_F1, BASIN_PRECIP_RMSE, ETR_ERR, STUDY_EPT_ERROR


def run_bayes_univariate_trends(traces_dir, stations_meta, multiproc=0, overwrite=False, selectors=None, stations=None):
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
            bayes_univariate_trends(sid, rec, traces_dir, overwrite, selectors)
        else:
            pool.apply_async(bayes_univariate_trends, args=(sid, rec, traces_dir, overwrite, selectors))

    if multiproc > 0:
        pool.close()
        pool.join()


def bayes_univariate_trends(station, records, trc_dir, overwrite, selectors=None):
    try:
        basin = records['basin']
        rmse = BASIN_CC_ERR[basin]['rmse']
        bias = BASIN_CC_ERR[basin]['bias']
        irr_f1 = 1 - BASIN_IRRMAPPER_F1[basin]
        cc_err = np.sqrt(irr_f1 ** 2 + rmse ** 2 + STUDY_EPT_ERROR ** 2)

        if 'static' in trc_dir:
            cc_err = np.sqrt(rmse ** 2 + STUDY_EPT_ERROR ** 2)

        ppt_err, etr_err = BASIN_PRECIP_RMSE[basin], ETR_ERR
        ai_err = np.sqrt(ppt_err ** 2 + etr_err ** 2)

        month = records['q_mo']
        cc = np.array(records['cc_month']) * (1 + bias)
        qres = np.array(records['qres'])
        q = np.array(records['q'])
        ai = np.array(records['ai'])
        aim = np.array(records['ai_month'])
        irr = np.array(records['irr'])
        years = np.array(records['years'])

        q = (q - q.min()) / (q.max() - q.min()) + 0.001
        ai = (ai - ai.min()) / (ai.max() - ai.min()) + 0.001
        aim = (aim - aim.min()) / (aim.max() - aim.min()) + 0.001
        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
        irr = (irr - irr.min()) / (irr.max() - irr.min()) + 0.001

        cc_err = np.ones_like(cc) * cc_err
        q_err = np.ones_like(q) * 0.08
        ai_err = np.ones_like(qres) * ai_err
        irr_err = np.ones_like(qres) * irr_f1

        years = (years - years.min()) / (years.max() - years.min()) + 0.001

        regression_combs = [(years, cc, None, cc_err),
                            (years, ai, None, ai_err),
                            (years, aim, None, ai_err),
                            (years, irr, None, irr_err),
                            (years, q, None, q_err)]

        trc_subdirs = ['time_cc', 'time_ai', 'time_aim', 'time_irr', 'time_q']

        for subdir in trc_subdirs:
            model_dir = os.path.join(trc_dir, subdir)
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

        for (x, y, x_err, y_err), subdir in zip(regression_combs, trc_subdirs):

            if selectors and subdir not in selectors:
                continue
            if subdir not in records.keys():
                continue
            if month not in range(4, 11) and subdir == 'time_cc':
                continue

            model_dir = os.path.join(trc_dir, subdir)

            save_model = os.path.join(model_dir, 'model', '{}_q_{}.model'.format(station, month))
            save_data = os.path.join(model_dir, 'data', '{}_q_{}.data'.format(station, month))
            save_png = os.path.join(model_dir, 'trace', '{}_q_{}.png'.format(station, month))

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
                      'err: {:.3f}, bias: {} ===='.format(subdir, len(x), month, station, y_err[0], bias))

                model = TimeTrendModel()

                model.fit(x, y, y_err, save_model=save_model, figure=save_png)

    except Exception as e:
        print(e, station)


def summarize_univariate_trends(metadata, trc_dir, out_json, month, update_selectors=None):
    div_, conv_ = 0, 0
    sp, sn = 0, 0

    with open(metadata, 'r') as f:
        stations = json.load(f)

    out_meta = {}
    if update_selectors:
        with open(out_json, 'r') as f:
            out_meta = json.load(f)

    trc_subdirs = ['time_cc', 'time_ai', 'time_aim', 'time_irr', 'time_q']
    ct = 0

    for i, (station, data) in enumerate(stations.items()):

        if station not in out_meta.keys():
            out_meta[station] = {month: data}

        if not update_selectors:
            out_meta[station] = {month: data}

        for subdir in trc_subdirs:

            if subdir == 'time_cc' and month not in list(range(4, 11)):
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
                        ct += 1
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
                     'r_hat': summary['r_hat'].slope,
                     'model': saved_model}

                if d['r_hat'] > 1.1:
                    div_ += 1

                if station not in out_meta.keys():
                    out_meta[station] = {subdir: None}

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

            except ValueError as e:
                print(station, e)

    print('{} models opened'.format(ct))
    with open(out_json, 'w') as f:
        json.dump(out_meta, f, indent=4, sort_keys=False)

    return ct, conv_, div_, sp, sn


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
