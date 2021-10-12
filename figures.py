import os
from pprint import pprint
import json
from datetime import date
from dateutil.relativedelta import relativedelta as rdlt

from pandas import read_csv, DataFrame
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pylab import rcParams
import statsmodels.api as sm
from hydrograph import hydrograph
from gage_list import CLMB_STATIONS, UMRB_STATIONS

STATIONS = CLMB_STATIONS + UMRB_STATIONS


def plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d):
    resid_line = fit_resid.params[1] * cc + fit_resid.params[0]
    clim_line = fit_clim.params[1] * ai + fit_clim.params[0]
    rcParams['figure.figsize'] = 16, 10
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.scatter(ai, q)
    ax1.plot(ai, clim_line)
    ax1.set(xlabel='ETr / PPT [-]')
    ax1.set(ylabel='q [m^3]')
    for i, y in enumerate(years):
        ax1.annotate(y, (ai[i], q[i]))
    plt.suptitle(desc_str)
    ax2.set(xlabel='cc [m]')
    ax2.set(ylabel='q epsilon [m^3]')
    ax2.scatter(cc, resid)
    ax2.plot(cc, resid_line)
    for i, y in enumerate(years):
        ax2.annotate(y, (cc[i], resid[i]))
    fig_name = os.path.join(fig_d, '{}.png'.format(desc_str.strip().split(' ')[0]))
    plt.savefig(fig_name)
    plt.close('all')


def filter_by_significance(metadata, ee_series, fig_d, out_jsn, plot=False):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        if sid not in STATIONS:
            continue
        r, p, lag, months, q_start = (v[s] for s in ['r', 'pval', 'lag', 'recession_months', 'q_start'])
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if not os.path.exists(_file):
            continue
        cdf = hydrograph(_file)

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        tot_area = metadata[sid]['AREA_SQKM'] * 1e6
        irr_area = np.nanmean(cdf['irr'].values) / tot_area
        if irr_area < 0.005:
            continue

        cdf['ai'] = (cdf['etr'] - cdf['ppt']) / (cdf['etr'] + cdf['ppt'])
        cdf['cci'] = cdf['cc'] / cdf['irr']

        q_dates = [(date(y, q_start, 1), date(y, 11, 1)) for y in years]
        clim_dates = [(date(y, months[-1], 1) + rdlt(months=-lag), date(y, 10, 31)) for y in years]
        cc_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]

        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in q_dates])
        ai = np.array([cdf['ai'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])

        ai_c = sm.add_constant(ai)

        try:
            ols = sm.OLS(q, ai_c)
        except Exception as e:
            print(sid, e, cdf['q'].dropna().index[0], cdf['qb'].dropna().index[-1])
            continue

        fit_clim = ols.fit()

        irr_ct += 1

        if fit_clim.pvalues[1] < 0.05:
            ct += 1
            resid = fit_clim.resid
            _cc_c = sm.add_constant(cc)
            ols = sm.OLS(resid, _cc_c)
            fit_resid = ols.fit()
            resid_p = fit_resid.pvalues[1]
            print(sid, '{:.2f}'.format(resid_p))
            if resid_p < 0.05:
                if fit_resid.params[1] > 0.0:
                    slp_pos += 1
                else:
                    slp_neg += 1
                sig_stations[sid] = fit_resid.params[1]
                desc_str = '\n{} {}\nlag = {} p = {:.3f}, ' \
                           'irr = {:.3f}, m = {:.2f}'.format(sid,
                                                             metadata[sid]['STANAME'], lag,
                                                             fit_resid.pvalues[1],
                                                             irr_area, fit_resid.params[1])
                if plot:
                    plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d)
                print(desc_str)
                irr_sig_ct += 1
                sig_stations[sid] = {'STANAME': metadata[sid]['STANAME'],
                                     'SIG': fit_resid.pvalues[1],
                                     'IRR_AREA': irr_area,
                                     'SLOPE': fit_resid.params[1],
                                     'lag': metadata[sid]['lag']}

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f)
    pprint(list(sig_stations.keys()))
    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct, ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))


def trend_analysis(meta, ee_data, climate_dir, fig_d, out_jsn=None):
    idf = read_csv(ee_data)
    idx = [str(c).rjust(8, '0') for c in idf['STAID'].values]
    idf.index = idx
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    with open(meta, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        r, p, lag, months = (v[s] for s in ['r', 'pval', 'lag', 'recession_months'])
        # months = [x for x in range(7, 10)]
        lag_yrs = int(np.ceil(float(lag) / 12))
        ct_tot += 1
        h_file = os.path.join(climate_dir, '{}.csv'.format(sid))
        if not os.path.exists(h_file):
            continue
        try:
            cdf = hydrograph(h_file)
            years = metadata[sid]['qb_years']
            irr = np.array([idf.loc[sid]['irr_{}'.format(y)] for y in years]) / (metadata[sid]['AREA_SQKM'] * 1e6)
            irr_area = irr.mean()
            if sid not in ['06309000', '06327500', '06329500', '06295000', '06214500']:
                continue
            if irr_area < 0.005:
                continue
            cdf = cdf[cdf.index.year.isin(years)]
            qb = cdf[cdf.index.month.isin(months)]['q'].dropna().resample('A').agg(DataFrame.sum,
                                                                                   skipna=False)
            ind_var = years
            _ind_c = sm.add_constant(ind_var)
            dep_var = qb.values

            try:
                ols = sm.OLS(dep_var, _ind_c)
            except Exception as e:
                print(sid, e, cdf['qb'].dropna().index[0], cdf['qb'].dropna().index[-1])
                continue

            trend_fit = ols.fit()
            trend_line = trend_fit.params[1] * np.array(ind_var) + trend_fit.params[0]
            irr_ct += 1
            desc_str = '\n{} {}\nlag = {} p = {:.3f}, ' \
                       'irr = {:.3f}, m = {:.2f}'.format(sid,
                                                         metadata[sid]['STANAME'], lag,
                                                         trend_fit.pvalues[1],
                                                         irr_area, trend_fit.params[1])
            print(desc_str)

            rcParams['figure.figsize'] = 16, 10
            plt.scatter(ind_var, qb.values)
            plt.plot(ind_var, trend_line)
            plt.xlabel('Year')
            plt.ylabel('qb')
            fig_name = os.path.join(fig_d, '{}.png'.format(sid))
            plt.savefig(fig_name)
            plt.close('all')

        except KeyError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass
        except IndexError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass

        if out_jsn:
            with open(out_jsn, 'w') as f:
                json.dump(sig_stations, f)

    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct, ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    ee_data = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q'

    _json = '/media/research/IrrigationGIS/gages/station_metadata/basin_lag_recession_11OCT2021.json'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_11OCT2021.json'
    fig_dir = '/media/research/IrrigationGIS/gages/figures/sig_irr_qb_monthly_comp_scatter_10OCT2021'

    filter_by_significance(_json, ee_data, fig_dir, out_jsn=o_json, plot=True)
# ========================= EOF ====================================================================
