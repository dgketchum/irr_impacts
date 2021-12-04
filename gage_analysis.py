import os
import json
from collections import OrderedDict, Counter
from pprint import pprint
from itertools import combinations
from calendar import monthrange

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
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
    out_gages = []
    gages = list(meta.keys())
    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            if sid in gages:
                out_gages.append(sid)
                features.append(f)

    shape_meta['schema']['properties'].update({'STANAME': 'str:40',
                                               'cc_frac': 'float:19.11',
                                               'irr_pct': 'float:19.11',
                                               'slope': 'float:19.11'})

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

                                                  ('cc_frac', meta[sid]['cc_frac']),
                                                  ('irr_pct', meta[sid]['irr_pct']),
                                                  ('slope', meta[sid]['slope']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'], f['properties']['STANAME'])
            except TypeError:
                pass


def climate_flow_correlation(climate_dir, in_json, out_json, plot_r=None):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    print(len(l), 'station files')
    offsets = [x for x in range(1, 61)]
    windows = {}
    lags = []
    date_sample = None
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        # try:
        sid = os.path.basename(csv).strip('.csv')
        s_meta = metadata[sid]
        if s_meta['AREA'] < 8000.:
            continue

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_pct = mean_irr * (1. / 1e6) / s_meta['AREA']
        if irr_pct == 0.0:
            continue

        years = [x for x in range(1991, 2021)]

        flow_periods = []
        max_len = 5
        rr = [x for x in range(7, 11)]
        for n in range(1, max_len + 1):
            for i in range(max_len - n):
                per = rr[i: i + n]
                if len(per) == 1:
                    per = [per[0], per[0]]
                per = tuple(per)
                flow_periods.append(per)

        corr = (0, 0.0)
        found_sig = False
        d = None
        ind = None
        r_dct = {}
        for q_win in flow_periods:
            key_ = '{}-{}'.format(q_win[0], q_win[-1])
            r_dct[key_] = []
            q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
            q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
            for lag in offsets:
                if lag < q_win[-1] - q_win[0] + 1:
                    r_dct[key_].append(np.nan)
                    continue
                dates = [(date(y, 10, 31) + rdlt(months=-lag), date(y, 10, 31)) for y in years]
                etr = np.array([df['etr'][d[0]: d[1]].sum() for d in dates])
                ppt = np.array([df['ppt'][d[0]: d[1]].sum() for d in dates])
                ind = etr / ppt
                lr = linregress(ind, q)
                r, p = lr.rvalue ** 2, lr.pvalue
                r_dct[key_].append(r)
                if r > corr[1] and p < 0.05:
                    corr = (lag, abs(r))
                    d = {'STANAME': s_meta['STANAME'], 'q_window': q_win,
                         'lag': lag, 'r': r, 'pval': p, 'irr_pct': irr_pct}
                    date_sample = dates[0][1], dates[0][0]
                    found_sig = True
        df = DataFrame(data=r_dct)
        if plot_r:
            desc = '{} {}\n{:.1f} sq km, {:.1f}% irr'.format(sid, s_meta['STANAME'], s_meta['AREA'], irr_pct * 100)
            df.plot(title=desc)
            plt.savefig(os.path.join(plot_r, '{}_{}.png'.format(sid, s_meta['STANAME'])))
        if not found_sig:
            print('no significant r at {}: {} ({} nan)'.format(sid, s_meta['STANAME'],
                                                               np.count_nonzero(np.isnan(ind))))
            continue

        print('\nmonth {} to {}'.format(d['q_window'][0], d['q_window'][-1]),
              'lag {}'.format(d['lag']), 'r {:.2f}'.format(d['r']),
              '\np {:.2f}'.format(d['pval']), d['STANAME'], '{:.3f}'.format(irr_pct), date_sample)

        windows[sid] = {**d, **s_meta}
        if irr_pct > 0.01:
            lags.append(d['lag'])
        # except Exception as e:
        #     print(csv, e)

    hist = sorted(Counter(lags).items())
    pprint(hist)
    print(sum([x[1] for x in hist]), ' stations')

    with open(out_json, 'w') as f:
        json.dump(windows, f, indent=4)


def get_sig_irr_impact(metadata, ee_series, out_jsn=None, fig_dir=None):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    impacted_gages = []
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        s_meta = metadata[sid]
        r, p, lag, q_window = (v[s] for s in ['r', 'pval', 'lag', 'q_window'])
        q_start, q_end = q_window[0], q_window[-1]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if sid not in ['09209400']:
            continue
        if p > 0.05:
            continue
        if not os.path.exists(_file):
            continue
        if sid in MAJOR_IMPORTS:
            continue
        if s_meta['irr_pct'] < 0.001:
            continue

        cdf = hydrograph(_file)

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        cdf['cci'] = cdf['cc'] / cdf['irr']
        q_dates = [(date(y, q_start, 1), date(y, q_end, monthrange(y, q_end)[1])) for y in years]
        clim_dates = [(date(y, 11, 1) + rdlt(months=-lag), date(y, 11, 1)) for y in years]
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        ai = etr / ppt

        cc_periods = []
        max_len = 7
        rr = [x for x in range(5, 11)]
        for n in range(1, max_len + 1):
            for i in range(max_len - n):
                per = rr[i: i + n]
                if len(per) == 1:
                    per = [per[0], per[0]]
                per = tuple(per)
                cc_periods.append(per)

        for cc_period in cc_periods:
            cc_start, cc_end = cc_period[0], cc_period[-1]

            if cc_end > q_end:
                continue
            # if cc_start > q_start:
            #     continue

            month_end = monthrange(2000, cc_end)[1]
            cc_dates = [(date(y, cc_start, 1), date(y, cc_end, month_end)) for y in years]

            q = np.array([cdf['q'][d[0]: d[1]].sum() for d in q_dates])

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
            ct += 1
            resid = fit_clim.resid
            _cc_c = sm.add_constant(cc)
            ols = sm.OLS(resid, _cc_c)
            fit_resid = ols.fit()
            resid_p = fit_resid.pvalues[1]
            if resid_p < 0.05:
                impacted_gages.append(sid)
                if fit_resid.params[1] > 0.0:
                    slp_pos += 1
                else:
                    slp_neg += 1

                slope = fit_resid.params[1] * (np.std(cc) / np.std(resid))
                desc_str = '{} {}\n' \
                           '{} months climate, flow months {}-{}\n' \
                           'crop consumption {} to {}\n' \
                           'p = {:.3f}, irr = {:.3f}, m = {:.2f}\n        '.format(sid, s_meta['STANAME'],
                                                                                   lag, q_start, q_end,
                                                                                   cc_start, cc_end,
                                                                                   fit_resid.pvalues[1],
                                                                                   s_meta['irr_pct'],
                                                                                   slope)

                print(desc_str)
                irr_sig_ct += 1
                if fig_dir:
                    plot_clim_q_resid(q=q, ai=ai, fit_clim=fit_clim, desc_str=desc_str, years=years, cc=cc,
                                      resid=resid, fit_resid=fit_resid, fig_d=fig_dir, cci_per=cc_period)

                if sid not in sig_stations.keys():
                    sig_stations[sid] = s_meta
                    sig_stations[sid].update({'{}_{}'.format(cc_start, cc_end): {'sig': fit_resid.pvalues[1],
                                                                                 'slope': slope,
                                                                                 'lag': s_meta['lag']}})
                else:
                    sig_stations[sid].update({'{}_{}'.format(cc_start, cc_end): {'sig': fit_resid.pvalues[1],
                                                                                 'slope': slope,
                                                                                 'lag': s_meta['lag']}})

    impacted_gages = list(set(impacted_gages))
    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)
    pprint(list(sig_stations.keys()))
    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct,
                                                                            ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))
    print('total impacted gages: {}'.format(len(impacted_gages)))
    pprint(impacted_gages)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    clim_dir = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021')
    i_json = os.path.join(root, 'gages/station_metadata/station_metadata.json')
    clim_resp = os.path.join(root, 'gages/station_metadata/basin_climate_response_.json')
    fig_dir_ = os.path.join(root, 'gages/figures/clim_q_correlations')
    # climate_flow_correlation(clim_dir, i_json, clim_resp, plot_r=fig_dir_)

    fig_dir = os.path.join(root, 'gages/figures/irr_impact_q_clim_delQ_cci')
    irr_resp = os.path.join(root, 'gages/station_metadata/irr_impacted.json')
    ee_data = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021')
    get_sig_irr_impact(clim_resp, ee_data, out_jsn=None, fig_dir=fig_dir)

# ========================= EOF ====================================================================
