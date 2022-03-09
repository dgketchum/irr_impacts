import os
import json

from tqdm import tqdm
import numpy as np
from pandas import read_csv
from scipy.stats.stats import linregress
from astroML.linear_model import LinearRegressionwithErrors

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


def estimate_parameter_distributions(impacts_json, trc_dir, qres_err=0.1, cc_err=0.1, fig_dir=None,
                                     cores=4, plot=False):
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    for station, data in tqdm(stations.items(), total=len(stations)):

        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]

        for period in impact_keys:

            records = data[period]
            try:
                print('\n{} {:.3f} {}'.format(station, records['res_sig'], data['STANAME']))

                cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
                qres = np.array(records['q_resid'])

                cc = (cc - cc.min()) / (cc.max() - cc.min())
                qres = (qres - qres.min()) / (qres.max() - qres.min())
                print('mean cc: {}, mean q res: {}'.format(cc.mean(), qres.mean()))

                qres_cc_lr = linregress(qres, cc)

                qres_err = qres_err * np.ones_like(qres)
                cc_err = cc_err * np.ones_like(cc)

                save_model = os.path.join(trc_dir, '{}_cc{}_q{}.model'.format(station,
                                                                              period,
                                                                              records['q_window']))

                sample_kwargs = {'draws': 1000, 'target_accept': 0.9, 'cores': cores, 'chains': 4}
                model = LinearRegressionwithErrors()
                model.fit(cc, qres, qres_err, cc_err, save_model=save_model, sample_kwargs=sample_kwargs)

                if plot:
                    desc = [station, data['STANAME'], 'cc', period, 'q', records['q_window']]
                    plot_trace(cc, qres, cc_err, qres_err, model, qres_cc_lr, fig_dir, desc)

            except Exception as e:
                print(e, station, period)
                continue


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    flux_ = os.path.join(root, 'ameriflux', 'ec_data', 'ec_ssebop_comp.csv')
    # ssebop_error(flux_)

    irrmap = os.path.join(root, 'climate', 'irrmapper', 'pixel_metric_climate_clip.csv')
    # irrmapper_error(irrmap)

    for var in ['cci']:
        state = 'ccerr_0.20_qreserr_0.17'
        trace_dir = os.path.join(root, 'gages', 'bayes', 'traces', state)
        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        _json = os.path.join(root, 'gages', 'station_metadata', '{}_impacted.json'.format(var))

        o_fig = os.path.join(root, 'gages', 'figures', 'slope_trace_{}'.format(var), state)
        if not os.path.exists(o_fig):
            os.makedirs(o_fig)

        # cc_err=0.32, qres_err=0.174
        estimate_parameter_distributions(_json, trc_dir=trace_dir, cc_err=0.20,
                                         qres_err=0.17, fig_dir=o_fig, cores=10,
                                         plot=True)

# ========================= EOF ====================================================================
