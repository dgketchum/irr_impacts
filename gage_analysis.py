import os
import json
from collections import OrderedDict
from pprint import pprint
from calendar import monthrange

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from scipy.stats.stats import linregress
from datetime import date
from dateutil.relativedelta import relativedelta as rdlt
import fiona
import statsmodels.api as sm

from figures import plot_clim_q_resid, plot_water_balance_trends
from hydrograph import hydrograph

os.environ['R_HOME'] = '/home/dgketchum/miniconda3/envs/renv/lib/R'

from gage_list import EXCLUDE_STATIONS


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


def water_balance_ratios(metadata, ee_series, watersheds=None, metadata_out=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    frac = []
    dct = {}
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        cdf = hydrograph(_file)
        cdf['cci'] = cdf['cc'] / cdf['irr']
        years = [x for x in range(1991, 2021)]
        cc_dates = [(date(y, 5, 1), date(y, 10, 31)) for y in years]
        clim_dates = [(date(y, 1, 1), date(y, 12, 31)) for y in years]
        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        irr = np.array([cdf['irr'][d[0]: d[1]].sum() for d in cc_dates])
        cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])
        if not np.all(irr > 0.0):
            continue
        print('cci: {:.3f}, {}'.format(np.mean(cci), v['STANAME']))

        dct[sid] = v
        dct[sid].update({'IAREA': np.mean(irr)})
        dct[sid].update({'cc_q': cc.sum() / q.sum()})
        dct[sid].update({'cci': np.mean(cci)})
        dct[sid].update({'q_ppt': q.sum() / ppt.sum()})
        dct[sid].update({'ai': etr.sum() / ppt.sum()})

    frac_dict = {k: v for k, v in frac}
    stations_ = [f[0] for f in frac]

    if metadata_out:
        with open(metadata_out, 'w') as fp:
            json.dump(dct, fp, indent=4, sort_keys=False)
    if watersheds:
        with fiona.open(watersheds, 'r') as src:
            features = [f for f in src]
            meta = src.meta
        meta['schema']['properties']['cc_f'] = 'float:19.11'
        out_shp = os.path.join(os.path.dirname(watersheds),
                               os.path.basename(metadata_out).replace('json', 'shp'))
        with fiona.open(out_shp, 'w', **meta) as dst:
            for f in features:
                sid = f['properties']['STAID']
                if sid in stations_:
                    f['properties']['cc_f'] = frac_dict[sid]
                    dst.write(f)


def climate_flow_correlation(climate_dir, in_json, out_json, plot_r=None):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    print(len(l), 'station files')
    offsets = [x for x in range(1, 61)]
    windows = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        # try:
        sid = os.path.basename(csv).strip('.csv')
        s_meta = metadata[sid]
        # if s_meta['AREA'] < 30000.:
        #     continue

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_pct = mean_irr * (1. / 1e6) / s_meta['AREA']
        # if irr_pct == 0.0:
        #     continue

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

        found_sig = False
        response_d = {}
        ind = None
        r_dct = {}
        for q_win in flow_periods:
            corr = (0, 0.0)
            key_ = '{}-{}'.format(q_win[0], q_win[-1])
            r_dct[key_] = []
            q_dates = [(date(y, q_win[0], 1), date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]
            q = np.array([df['q'][d[0]: d[1]].sum() for d in q_dates])
            for lag in offsets:
                if lag < q_win[-1] - q_win[0] + 1:
                    r_dct[key_].append(np.nan)
                    continue
                dates = [(date(y, q_win[-1], monthrange(y, q_win[-1])[1]) + rdlt(months=-lag),
                          date(y, q_win[-1], monthrange(y, q_win[-1])[1])) for y in years]

                etr = np.array([df['etr'][d[0]: d[1]].sum() for d in dates])
                ppt = np.array([df['ppt'][d[0]: d[1]].sum() for d in dates])
                ind = etr / ppt
                lr = linregress(ind, q)
                r, p = lr.rvalue ** 2, lr.pvalue
                r_dct[key_].append(r)
                if r > corr[1] and p < 0.05:
                    corr = (lag, abs(r))

                    response_d[key_] = {'q_window': q_win, 'lag': lag, 'r': r, 'pval': p, 'irr_pct': irr_pct,
                                        'c_dates': '{} to {}'.format(str(dates[0][0]), str(dates[0][1])),
                                        'q_dates': '{} to {}'.format(str(q_dates[0][0]), str(q_dates[0][1]))}
                    found_sig = True

        df = DataFrame(data=r_dct)
        if plot_r:
            desc = '{} {}\n{:.1f} sq km, {:.1f}% irr'.format(sid, s_meta['STANAME'], s_meta['AREA'], irr_pct * 100)
            df.plot(title=desc)
            plt.savefig(os.path.join(plot_r, '{}_{}.png'.format(sid, s_meta['STANAME'])))
            plt.close()

        if not found_sig:
            print('no significant r at {}: {} ({} nan)'.format(sid, s_meta['STANAME'],
                                                               np.count_nonzero(np.isnan(ind))))
            continue

        print('\n', sid, s_meta['STANAME'], 'irr {:.3f}'.format(irr_pct))
        desc_strs = [(k, v['lag'], 'r: {:.3f}, p: {:.3f} q {} climate {}'.format(v['r'], v['pval'],
                                                                                 v['q_dates'],
                                                                                 v['c_dates'])) for k, v in
                     response_d.items()]
        [print('{}'.format(x)) for x in desc_strs]

        windows[sid] = {**response_d, **s_meta}

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
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        cdf = hydrograph(_file)

        if not os.path.exists(_file):
            continue
        if sid in EXCLUDE_STATIONS:
            continue
        # if sid not in SELECTED_SYSTEMS:
        #     continue

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        # iterate over flow windows
        for k, v in s_meta.items():
            if not isinstance(v, dict):
                continue

            r, p, lag, q_window = (v[s] for s in ['r', 'pval', 'lag', 'q_window'])
            q_start, q_end = q_window[0], q_window[-1]

            if p > 0.05 or v['irr_pct'] < 0.001:
                continue

            cdf['cci'] = cdf['cc'] / cdf['irr']
            q_dates = [(date(y, q_start, 1), date(y, q_end, monthrange(y, q_end)[1])) for y in years]
            clim_dates = [(date(y, q_end, monthrange(y, q_end)[1]) + rdlt(months=-lag),
                           date(y, q_end, monthrange(y, q_end)[1])) for y in years]
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

            # iterate over crop consumption windows
            for cc_period in cc_periods:
                cc_start, cc_end = cc_period[0], cc_period[-1]

                if cc_end > q_end:
                    continue
                if cc_start > q_start:
                    continue

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
                                                                                       v['irr_pct'],
                                                                                       slope)

                    print(desc_str)
                    irr_sig_ct += 1
                    if fig_dir:
                        plot_clim_q_resid(q=q, ai=ai, fit_clim=fit_clim, desc_str=desc_str, years=years, cc=cc,
                                          resid=resid, fit_resid=fit_resid, fig_d=fig_dir, cci_per=cc_period,
                                          flow_per=(q_start, q_end))

                    if sid not in sig_stations.keys():
                        sig_stations[sid] = {k: v for k, v in s_meta.items() if not isinstance(v, dict)}
                        sig_stations[sid].update({'{}-{}'.format(cc_start, cc_end): {'sig': fit_resid.pvalues[1],
                                                                                     'slope': slope,
                                                                                     'lag': lag,
                                                                                     'q_window': k}})
                    else:
                        sig_stations[sid].update({'{}-{}'.format(cc_start, cc_end): {'sig': fit_resid.pvalues[1],
                                                                                     'slope': slope,
                                                                                     'lag': lag,
                                                                                     'q_window': k}})

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


def basin_trends(key, metadata, ee_series, start_mo, end_mo, out_jsn=None, fig_dir=None):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        # if sid not in ['06177000']:
        #     continue
        s_meta = metadata[sid]
        if s_meta['AREA'] < 7000:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        try:
            cdf = hydrograph(_file)
        except FileNotFoundError:
            continue

        cdf['cci'] = cdf['cc'] / cdf['irr']
        cdf['ai'] = cdf['etr'] / cdf['ppt']
        years = [x for x in range(1991, 2021)]

        dates = [(date(y, start_mo, 1), date(y, end_mo, 31)) for y in years]

        data = np.array([cdf[key][d[0]: d[1]].sum() for d in dates])

        ct_tot += 1
        years_c = sm.add_constant(years)

        _years = np.array(years)
        try:
            ols = sm.OLS(data, years_c)
        except Exception as e:
            print(sid, e, cdf['q'].dropna().index[0])
            continue

        ols_fit = ols.fit()

        # if ols_fit.pvalues[1] < 0.05:

        ols_line = ols_fit.params[1] * _years + ols_fit.params[0]

        irr_ct += 1

        desc_str = '{} {}\n' \
                   'crop consumption {} to {}\n' \
                   'irr = {:.3f}\n        '.format(sid, s_meta['STANAME'],
                                                   5, 10,
                                                   v['IAREA'] / 1e6 * v['AREA'])

        print(desc_str)
        irr_sig_ct += 1
        if fig_dir:
            plot_water_balance_trends(data=data, data_line=ols_line, data_str=key,
                                      years=years, desc_str=desc_str, fig_d=fig_dir)

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS'

    # watersheds_shp = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds.shp'
    # _json = '/media/research/IrrigationGIS/gages/station_metadata/irr_impacted_all.json'
    # ee_data = '/media/research/IrrigationGIS/gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021'
    # cc_frac_json = '/media/research/IrrigationGIS/gages/station_metadata/basin_cc_ratios.json'
    # water_balance_ratios(_json, ee_data, watersheds=None, metadata_out=cc_frac_json)

    # clim_dir = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021')
    # i_json = os.path.join(root, 'gages/station_metadata/station_metadata.json')
    # clim_resp = os.path.join(root, 'gages/station_metadata/basin_climate_response_all.json')
    # fig_dir_ = os.path.join(root, 'gages/figures/clim_q_correlations')
    # climate_flow_correlation(clim_dir, i_json, clim_resp, plot_r=None)

    # clim_resp = os.path.join(root, 'gages/station_metadata/basin_climate_response_irr.json')
    # fig_dir = os.path.join(root, 'gages/figures/irr_impact_q_clim_delQ_cci_gt30000')
    # irr_resp = os.path.join(root, 'gages/station_metadata/irr_impacted_gt30000.json')
    # ee_data = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_q_sw_17NOV2021')
    # get_sig_irr_impact(clim_resp, ee_data, out_jsn=None, fig_dir=fig_dir)

    irr_impacted = os.path.join(root, 'gages/station_metadata/basin_cc_ratios.json')
    ee_data = os.path.join(root, 'gages/merged_q_ee/monthly_ssebop_tc_q_Comp_14DEC2021')
    fig_dir = os.path.join(root, 'gages/figures/water_balance_time_series')
    # trend_json = os.path.join(root, 'gages/water_balance_time_series/cc_q_trends.json')
    for k in ['ppt', 'q', 'etr', 'ai', 'cc', 'cci', 'irr'][-1:]:
        fig_ = os.path.join(fig_dir, k)
        if not os.path.exists(fig_):
            os.mkdir(fig_)
        basin_trends(k, irr_impacted, ee_data, out_jsn=None, fig_dir=fig_,
                     start_mo=1, end_mo=12)

# ========================= EOF ====================================================================
