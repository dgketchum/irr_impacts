import os
import json
from collections import OrderedDict, Counter
from pprint import pprint
from itertools import combinations
from calendar import monthrange

import numpy as np
from scipy.stats.stats import linregress
from datetime import date
from dateutil.relativedelta import relativedelta as rdlt
import fiona
import statsmodels.api as sm

from figures import MAJOR_IMPORTS, plot_clim_q_resid
from hydrograph import hydrograph

os.environ['R_HOME'] = '/home/dgketchum/miniconda3/envs/renv/lib/R'

from gage_list import CLMB_STATIONS, UMRB_STATIONS

STATIONS = CLMB_STATIONS + UMRB_STATIONS


def write_json_to_shapefile(in_shp, out_shp, meta):
    with open(meta, 'r') as f:
        meta = json.load(f)

    features = []
    gages = list(meta.keys())
    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            if sid in gages:
                features.append(f)

    shape_meta['schema']['properties']['lag'] = 'float:19.11'
    shape_meta['schema']['properties']['IRR_AREA'] = 'float:19.11'
    shape_meta['schema']['properties']['SLOPE'] = 'float:19.11'

    ct = 0
    with fiona.open(out_shp, 'w', **shape_meta) as dst:
        for f in features:
            ct += 1
            sid = f['properties']['STAID']
            feature = {'geometry': f['geometry'],
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('SQMI', f['properties']['SQMI']),

                                                  ('lag', meta[sid]['lag']),
                                                  ('IRR_AREA', meta[sid]['IRR_AREA']),
                                                  ('SLOPE', meta[sid]['SLOPE']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'], f['properties']['STANAME'])
            except TypeError:
                pass


def climate_flow_correlation(climate_dir, in_json, out_json):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    print(len(l), 'station files')
    offsets = [x for x in range(1, 36)]
    windows = {}
    lags = []
    month_series = [x for x in range(1, 13)] + [x for x in range(1, 13)]
    date_sample = None
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        try:
            sid = os.path.basename(csv).strip('.csv')
            # if sid != '09209400':
            #     continue
            s_meta = metadata[sid]
            max_q_mo, min_q_mo = s_meta['max_mo'], s_meta['min_mo']
            if min_q_mo < max_q_mo:
                recession_months = month_series[max_q_mo: min_q_mo + 12]
            else:
                recession_months = month_series[max_q_mo: min_q_mo + 1]
            irr = s_meta['irr_mean']
            if irr == 0.0:
                continue
            print('\n{}'.format(sid))
            df = hydrograph(csv)

            years = [x for x in range(1991, 2021)]
            base_l = [x for x in range(7, 11)]
            flow_periods = [x for n in range(2, len(base_l)) for x in combinations(base_l, n)]
            corr = (0, 0.0)
            found_sig = False
            d = None
            ind = None
            for q_win in flow_periods:
                q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
                q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
                for lag in offsets:
                    dates = [(date(y, 10, 31) + rdlt(months=-lag), date(y, 10, 31)) for y in years]
                    etr = np.array([df['etr'][d[0]: d[1]].sum() for d in dates])
                    ppt = np.array([df['ppt'][d[0]: d[1]].sum() for d in dates])
                    ind = etr / ppt
                    lr = linregress(ind, q)
                    r, p = lr.rvalue, lr.pvalue
                    if abs(r) > corr[1] and p < 0.05:
                        corr = (lag, abs(r))
                        d = {'STANAME': s_meta['STANAME'], 'q_window': q_win,
                             'lag': lag, 'r': r, 'pval': p, 'recession_months': recession_months}
                        date_sample = dates[0][1], dates[0][0]
                        found_sig = True

            if not found_sig:
                print('no significant r at {}: {} ({} nan)'.format(sid, s_meta['STANAME'],
                                                                   np.count_nonzero(np.isnan(ind))))
                continue

            print('month {} to {}'.format(d['q_window'][0], d['q_window'][-1]),
                  'lag {}'.format(d['lag']), 'r {:.2f}'.format(d['r']),
                  '\np {:.2f}'.format(d['pval']), d['STANAME'], '{:.3f}'.format(irr), date_sample)

            windows[sid] = {**d, **s_meta}
            if irr > 0.01:
                lags.append(d['lag'])
        except Exception as e:
            print(csv, e)

    hist = sorted(Counter(lags).items())
    pprint(hist)
    print(sum([x[1] for x in hist]), ' stations')

    with open(out_json, 'w') as f:
        json.dump(windows, f)


def filter_by_significance(metadata, ee_series, out_jsn, fig_d=None):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    impacted_gages = []
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        # if sid != '09209400':
        #     continue
        r, p, lag, months, q_window = (v[s] for s in ['r', 'pval', 'lag',
                                                      'recession_months', 'q_window'])
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if p > 0.05:
            continue
        if not os.path.exists(_file):
            continue
        if sid in MAJOR_IMPORTS:
            continue

        cdf = hydrograph(_file)

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        tot_area = metadata[sid]['AREA_SQKM'] * 1e6
        irr_area = np.nanmean(cdf['irr'].values) / tot_area
        if irr_area < 0.005:
            continue

        cdf['cci'] = cdf['cc'] / cdf['irr']
        q_dates = [(date(y, q_window[0], 1), date(y, q_window[-1], monthrange(y, q_window[-1])[1])) for y in years]
        clim_dates = [(date(y, months[-1], 1) + rdlt(months=-lag), date(y, 11, 1)) for y in years]

        for start_cci in range(5, 11):
            cc_dates = [(date(y, start_cci, 1), date(y, 10, 31)) for y in years]

            q = np.array([cdf['q'][d[0]: d[1]].sum() for d in q_dates])
            ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
            etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
            ai = etr / ppt
            cc = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])

            ai_c = sm.add_constant(ai)

            try:
                ols = sm.OLS(q, ai_c)
            except Exception as e:
                print(sid, e, cdf['q'].dropna().index[0])
                continue

            fit_clim = ols.fit()

            irr_ct += 1
            clim_p = fit_clim.pvalues[1]
            if clim_p > p and clim_p > 0.05:
                print('\n', sid, v['STANAME'], '{:.3f} p {:.3f} clim p'.format(p, clim_p), '\n')
            # if clim_p < 0.05:
            ct += 1
            resid = fit_clim.resid
            _cc_c = sm.add_constant(cc)
            ols = sm.OLS(resid, _cc_c)
            fit_resid = ols.fit()
            resid_p = fit_resid.pvalues[1]
            # print(sid, '{:.2f}'.format(resid_p))
            if resid_p < 0.05:
                impacted_gages.append(sid)
                if fit_resid.params[1] > 0.0:
                    slp_pos += 1
                else:
                    slp_neg += 1
                desc_str = '\n{} {}\nlag = {} p = {:.3f}, ' \
                           'irr = {:.3f}, m = {:.2f}'.format(sid,
                                                             metadata[sid]['STANAME'], lag,
                                                             fit_resid.pvalues[1],
                                                             irr_area, fit_resid.params[1])
                if fig_d:
                    plot_clim_q_resid(q, ai, fit_clim, desc_str, years, cc, resid, fit_resid, fig_d)
                print(desc_str)
                irr_sig_ct += 1
                if sid not in sig_stations.keys():
                    sig_stations[sid] = {'STANAME': metadata[sid]['STANAME'],
                                         '{}_{}'.format(start_cci, 10): {'sig': fit_resid.pvalues[1],
                                                                         'irr_area': irr_area,
                                                                         'slope': fit_resid.params[1],
                                                                         'lag': metadata[sid]['lag']}}
                else:
                    sig_stations[sid]['{}_{}'.format(start_cci, 10)] = {'sig': fit_resid.pvalues[1],
                                                                        'irr_area': irr_area,
                                                                        'slope': fit_resid.params[1],
                                                                        'lag': metadata[sid]['lag']}

    impacted_gages = list(set(impacted_gages))
    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f)
    pprint(list(sig_stations.keys()))
    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct,
                                                                            ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))
    print('total impacted gages: {}'.format(len(impacted_gages)))
    pprint(impacted_gages)


if __name__ == '__main__':
    dq = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    # i_json = '/media/research/IrrigationGIS/gages/station_metadata/metadata.json'
    # o_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_irr_lf.json'
    # mq = '/media/research/IrrigationGIS/gages/hydrographs/q_bf_monthly'
    # m = 1
    # irr_data = '/media/research/IrrigationGIS/gages/ee_exports/series/extracts_comp_25AUG2021.csv'
    # get_basin_metadata(i_json, mq, dq, irr_data, o_json, m)

    clim_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021'
    i_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_irr_lf.json'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_lag_recession_ai_17NOV2021.json'
    # climate_flow_correlation(clim_dir, i_json, o_json)

    # i_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_31AUG2021.json'
    # i_shp = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds_meta.shp'
    # o_shp = '/media/research/IrrigationGIS/gages/watersheds/response_31AUG2021.shp'
    # write_json_to_shapefile(i_shp, o_shp, i_json)

    _json = '/media/research/IrrigationGIS/gages/station_metadata/basin_lag_recession_ai_17NOV2021.json'
    o_json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_metadata_17NOV2021.json'

    figs = '/media/research/IrrigationGIS/gages/figures'
    fig_dir = os.path.join(figs, 'sig_irr_qb_monthly_comp_scatter_17NOV2021')
    ee_data = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021'

    filter_by_significance(_json, ee_data, out_jsn=o_json, fig_d=None)

# ========================= EOF ====================================================================
