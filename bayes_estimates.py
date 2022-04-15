import os
import json
from multiprocessing import Pool, cpu_count

import numpy as np
from pandas import read_csv
from scipy.stats.stats import linregress
from linear_regression_errors import LinearRegressionwithErrors, LinearModel
# from astroML.linear_model import LinearRegressionwithErrors

from figs.regression_figs import plot_trace


def ssebop_error(csv):
    df = read_csv(csv)
    et_var = 'ET'
    df['fill'] = np.isnan(df['ET'])

    # drop MT dryland sites
    df['spec_drop'] = [x not in ['US-Mj1', 'US-Mj2'] for x in list(df['site'])]

    df = df[df['spec_drop']]
    # df = df[df['site'] == 'S2']
    df['ET'].loc[df['fill']] = df['ET_fill']

    # initial data
    df = df[['et_ssebop', et_var, 'site', 'date', 'ET_gap']]
    # df.dropna(how='any', inplace=True)
    df['diff_f'] = (df['et_ssebop'] - df[et_var]) / df[et_var]
    df['diff_abs'] = df['et_ssebop'] - df[et_var]
    abs_err = np.nanmean(df['diff_abs'])
    print('intial mean pct error: {:.3f}'.format(np.nanmean(df['diff_f'] * 100.)))
    print('intial mean abs error: {:.3f}'.format(abs_err))

    lr = linregress(df[et_var], df['et_ssebop'])
    line_ = lr.slope * df[et_var].values + lr.intercept

    # plt.xlabel('EC ET')
    # plt.ylabel('SSEBOP ET')
    # plt.plot(df[et_var].values, line_)
    # plt.scatter(df[et_var], df['et_ssebop'])
    # plt.show()
    # plt.close()
    pass


def irrmapper_error(csv):
    df = read_csv(csv)
    pass


def regression_errors(station, records, period, qres_err, cc_err, trc_dir, cores):
    try:
        print('\n{}, p = {:.3f}'.format(station, records['res_sig']))

        cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
        qres = np.array(records['q_resid'])

        cc = (cc - cc.min()) / (cc.max() - cc.min()) + 0.001
        qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
        years = (np.linspace(0, 1, len(qres)) + 0.001).reshape(1, -1) + 0.001

        qres_err = qres_err * np.ones_like(qres)
        cc_err = cc_err * np.ones_like(cc)

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

        for (x, y, x_err, y_err), subdir in zip(regression_combs, trc_subdirs):
            if subdir == 'time_cc':
                y = y[0, :]

            save_model = os.path.join(trc_dir, subdir, '{}_cc_{}_q_{}.model'.format(station,
                                                                                    period,
                                                                                    records['q_window']))
            if os.path.isfile(save_model):
                print('{} exists'.format(subdir))
                continue
            else:
                print('sampling {}'.format(subdir))

            if subdir == 'cc_qres':
                model = LinearRegressionwithErrors()
            else:
                model = LinearModel()

            model.fit(x, y, y_err, x_err,
                      save_model=save_model,
                      sample_kwargs=sample_kwargs)

    except Exception as e:
        print(e, station, period)


if __name__ == '__main__':

    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    flux_ = os.path.join(root, 'ameriflux', 'ec_data', 'ec_ssebop_comp.csv')
    # ssebop_error(flux_)

    irrmap = os.path.join(root, 'climate', 'irrmapper', 'pixel_metric_climate_clip.csv')
    # irrmapper_error(irrmap)

    cc_err_ = '0.196'
    qres_err_ = '0.17'
    mproc = 30

    for var in ['cci']:
        state = 'ccerr_{}_qreserr_{}'.format(str(cc_err_), str(qres_err_))
        trace_dir = os.path.join(root, 'gages', 'bayes', 'traces', state)
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        _json = os.path.join(root, 'gages', 'station_metadata',
                             '{}_impacted.json'.format(var))

        o_fig = os.path.join(root, 'gages', 'figures',
                             'slope_trace_{}'.format(var), state)

        if not os.path.exists(o_fig):
            os.makedirs(o_fig)

        with open(_json, 'r') as f:
            stations = json.load(f)

        diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
        diter = [i for ll in diter for i in ll]

        pool = Pool(processes=mproc)

        for sid, per, rec in diter:
            # if sid != '09486500' or per != '5-6':
            #     continue
            # if sid != '06025500' or per != '10-10':
            #     continue
            # regression_errors(sid, rec, per, float(qres_err_),
            #                   float(cc_err_), trace_dir, 4)
            pool.apply_async(regression_errors, args=(sid, rec, per, float(qres_err_),
                                                      float(cc_err_), trace_dir, 1))

        pool.close()
        pool.join()

# ========================= EOF ====================================================================
