import os
import json

import numpy as np
import pymc3 as pm
from pandas import read_csv

from sklearn.preprocessing import MinMaxScaler
from scipy.stats.stats import linregress
from matplotlib import pyplot as plt
import arviz as az

from astroML.datasets import simulation_kelly
from astroML.linear_model import LinearRegressionwithErrors
from astroML.plotting import plot_regressions, plot_regression_from_trace


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


def estimate_parameter_distributions(impacts_json, fig_dir, qres_err=0.1, cc_err=0.1):
    scaler = MinMaxScaler()
    with open(impacts_json, 'r') as f:
        stations = json.load(f)

    for station, data in stations.items():
        impact_keys = [p for p, v in data.items() if isinstance(v, dict)]
        for period in impact_keys:
            if station != '06126500' or period != '7-9':
                continue
            records = data[period]
            try:
                print('{} {:.3f} {}'.format(station, records['res_sig'], data['STANAME']))
                # if records['res_sig'] > 0.02:
                #     continue

                # ai, q = records['ai_data'], records['q_data']
                # q = np.log10(q)
                # q_clim_sigma_slope = np.sqrt((max(ai) - min(ai)) / (max(q) - min(q)))
                # q_clim_lr = linregress(q, ai)
                # q_clim = linear_regression(ai, q, q_clim_lr, q_clim_sigma_slope)
                # with q_clim:
                #     q_clim_fit = pm.sample(return_inferencedata=True)

                cc = np.array(records['cc_data']).reshape(1, len(records['cc_data']))
                qres = np.array(records['q_resid'])

                # e.g., simulated data as in
                # https://www.astroml.org/notebooks/chapter8/astroml_chapter8_Regression_with_Errors_on_Dependent_and_Independent_Variables.html
                # ksi, eta, xi, yi, xi_error, yi_error, alpha_in, beta_in = simulation_kelly(size=100,
                #                                                                            scalex=0.2,
                #                                                                            scaley=0.2,
                #                                                                            alpha=2,
                #                                                                            beta=1,
                #                                                                            epsilon=(0, 0.75))

                # qres, cc = np.array(qres) / 1e8, np.array(cc) / 1e8
                cc = (cc - cc.min()) / (cc.max() - cc.min())
                qres = (qres - qres.min()) / (qres.max() - qres.min())
                print('mean cc: {}, mean q res: {}'.format(cc.mean(), qres.mean()))

                qres_cc_lr = linregress(qres, cc)

                model = LinearRegressionwithErrors()
                qres_err = qres_err * np.ones_like(qres)
                cc_err = cc_err * np.ones_like(cc)
                model.fit(cc, qres, qres_err, cc_err)

                plot_regressions(0.0, 0.0, cc[0], qres,
                                 cc_err[0], qres_err,
                                 add_regression_lines=True,
                                 alpha_in=qres_cc_lr.intercept, beta_in=qres_cc_lr.slope)

                # plt.scatter(cc, qres)
                # plt.xlim([cc.min(), cc.max()])
                # plt.ylim([qres.min(), qres.max()])

                plot_regression_from_trace(model, (cc, qres, cc_err, qres_err),
                                           ax=plt.gca(), chains=50)

                # fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex='col')
                # az.plot_posterior(q_clim_fit, var_names=["slope"], ref_val=q_clim_lr.slope, ax=ax[0])
                # ax[0].set(title="Flow - Climate", xlabel="slope")
                #
                # az.plot_posterior(model, var_names=["slope"], ref_val=qres_cc_lr.slope, ax=ax[1])
                # ax[1].set(title="Residual Flow - Crop Consumption", xlabel="slope")
                #
                desc_str = '{} {}\n crop consumption: {}\n flow: {}'.format(station, data['STANAME'], period,
                                                                            records['q_window'])
                plt.suptitle(desc_str)
                fig_file = os.path.join(fig_dir, '{}_cc{}_q{}.png'.format(station, records['q_window'],
                                                                          period))
                # plt.savefig(fig_file)
                plt.show()
                # plt.close()

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
        _json = os.path.join(root, 'gages', 'station_metadata', '{}_impacted_06126500.json'.format(var))
        o_fig = os.path.join(root, 'gages', 'figures', 'slope_trace_{}'.format(var))

        # cc_err=0.32, qres_err=0.174
        estimate_parameter_distributions(_json, fig_dir=o_fig, cc_err=0.50, qres_err=0.17)

# ========================= EOF ====================================================================
