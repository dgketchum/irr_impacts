import os
import json
from multiprocessing import Pool, cpu_count

import numpy as np
from pandas import read_csv
from scipy.stats.stats import linregress
from linear_regression_errors import LinearRegressionwithErrors
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


def regression_errors(station, records, period, qres_err, cc_err, trc_dir):
    try:
        print('\n{} {:.3f}'.format(station, records['res_sig']))

        cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
        qres = np.array(records['q_resid'])

        cc = (cc - cc.min()) / (cc.max() - cc.min())
        qres = (qres - qres.min()) / (qres.max() - qres.min())
        years = np.array([x for x in range(1991, 2021)])
        print('mean cc: {}, mean q res: {}'.format(cc.mean(), qres.mean()))

        qres_err = qres_err * np.ones_like(qres)
        cc_err = cc_err * np.ones_like(cc)
        years_err = 0.0 * np.ones_like(cc)

        sample_kwargs = {'draws': 1000,
                         'target_accept': 0.9,
                         'cores': 1,
                         'chains': 4}

        regression_combs = [(cc, qres, cc_err, qres_err),
                            (years, cc, years_err, cc_err),
                            (years, qres, years_err, qres_err)]

        trc_subdirs = ['cc_qres', 'time_cc', 'time_qres']

        for (x, y, x_err, y_err), subdir in zip(regression_combs[1:], trc_subdirs[1:]):
            save_model = os.path.join(trc_dir, subdir, '{}_cc_{}_q_{}.model'.format(station,
                                                                                    period,
                                                                                    records['q_window']))
            if os.path.isfile(save_model):
                print('{} exists, skipping'.format(os.path.basename(save_model)))
                continue

            model = LinearRegressionwithErrors()
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

    # cc_err=0.32, qres_err=0.174
    cc_err = '0.305'
    qres_err = '0.17'
    mproc = 1

    for var in ['cci']:
        state = 'ccerr_{}_qreserr_{}'.format(str(cc_err), str(qres_err))
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
            pool.apply_async(regression_errors, args=(sid, rec, per, float(qres_err),
                                                      float(cc_err), trace_dir))

        pool.close()
        pool.join()

# ========================= EOF ====================================================================
