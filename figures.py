import os
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
from gage_analysis import EXCLUDE_STATIONS


def filter_by_significance(metadata, ee_series, climate_dir, fig_d, out_jsn):
    idf = read_csv(ee_series)
    idx = [str(c).rjust(8, '0') for c in idf['STAID'].values]
    idf.index = idx
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            print(metadata[sid]['STANAME'], 'excluded')
        r, p, lag = (v[s] for s in ['r', 'pval', 'lag'])
        lag_yrs = int(np.ceil(float(lag) / 12))
        try:
            ct_tot += 1
            h_file = os.path.join(climate_dir, '{}.csv'.format(sid))
            if not os.path.exists(h_file):
                continue
            cdf = hydrograph(h_file)
            years = [x for x in range(1986, 2021)]
            irr = np.array([idf.loc[sid]['irr_{}'.format(y)] for y in years]) / (metadata[sid]['AREA_SQKM'] * 1e6)
            irr_area = irr.mean()
            if irr_area < 0.005:
                continue
            cdf['ai'] = (cdf['etr'] - cdf['ppt']) / (cdf['etr'] + cdf['ppt'])
            qb = cdf[cdf.index.month.isin([8, 9, 10, 11])]['qb'].dropna().resample('A').agg(DataFrame.sum,
                                                                                               skipna=False)
            ai = cdf['ai']
            dates = [(date(y, 11, 1), date(y, 11, 1) + rdlt(months=-lag)) for y in
                     years]
            ind_var = np.array([ai[d[1]: d[0]].sum() for d in dates])

            _ind_c = sm.add_constant(ind_var)
            dep_var = qb.values

            try:
                ols = sm.OLS(dep_var, _ind_c)
            except Exception as e:
                print(e, cdf['qb'].dropna().index[0], cdf['qb'].dropna().index[-1])
                continue

            fit_clim = ols.fit()
            clim_line = fit_clim.params[1] * ind_var + fit_clim.params[0]
            irr_ct += 1

            if fit_clim.pvalues[1] < 0.05:
                resid = fit_clim.resid

                if 3 > lag_yrs > 1:
                    irr = np.array([irr[i] + irr[i + 1] for i, _ in enumerate(years[1:])])
                    resid = resid[1:]
                    alt_years = years[1:]
                else:
                    alt_years = years

                _irr_c = sm.add_constant(irr)
                ols = sm.OLS(resid, _irr_c)
                fit_resid = ols.fit()
                if fit_resid.pvalues[1] < 0.05:
                    if fit_resid.params[1] > 0.0:
                        slp_pos += 1
                    else:
                        slp_neg += 1
                    resid_line = fit_resid.params[1] * irr + fit_resid.params[0]
                    sig_stations[sid] = fit_resid.params[1]
                    desc_str = '\n{} {}\nlag = {} p = {:.3f}, ' \
                               'irr = {:.3f}, m = {:.2f}'.format(sid,
                                                                 metadata[sid]['STANAME'], lag,
                                                                 fit_resid.pvalues[1],
                                                                 irr_area, fit_resid.params[1])
                    print(desc_str)

                    rcParams['figure.figsize'] = 16, 10
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.scatter(ind_var, dep_var)
                    ax1.plot(ind_var, clim_line)
                    ax1.set(xlabel='ETr / PPT')
                    ax1.set(ylabel='qb')
                    for i, y in enumerate(years):
                        ax1.annotate(y, (ind_var[i], dep_var[i]))
                    plt.suptitle(desc_str)
                    ax2.set(xlabel='irr')
                    ax2.set(ylabel='qb epsilon')
                    ax2.scatter(irr, resid)
                    ax2.plot(irr, resid_line)
                    for i, y in enumerate(alt_years):
                        ax2.annotate(y, (irr[i], resid[i]))
                    fig_name = os.path.join(fig_d, '{}.png'.format(sid))
                    plt.savefig(fig_name)
                    plt.close('all')
                    irr_sig_ct += 1

                    ct += 1
                    sig_stations[sid] = {'STANAME': metadata[sid]['STANAME'],
                                         'SIG': fit_resid.pvalues[1],
                                         'IRR_AREA': irr_area,
                                         'SLOPE': fit_resid.params[1],
                                         'lag': metadata[sid]['lag']}

        except KeyError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f)

    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct, ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    # c = '/media/research/IrrigationGIS/gages/hydrographs/group_stations/stations_annual.csv'
    # fig = '/media/research/IrrigationGIS/gages/figures/bfi_vs_irr.png'
    # src = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'

    clim_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/q_terraclim'
    ee_data = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_comp_25AUG2021.csv'
    _json = '/media/research/IrrigationGIS/gages/station_metadata/climresponse_qbJASO_fromOct1.json'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_25AUG2021.json'
    fig_dir = '/media/research/IrrigationGIS/gages/figures/sig_irr_qb_monthly_comp_scatter_18AUG2021'
    filter_by_significance(_json, ee_data, clim_dir, fig_dir, out_jsn=o_json)
    # two_variable_relationship('qb', 'irr', jsn, data, fig_dir)
# ========================= EOF ====================================================================
