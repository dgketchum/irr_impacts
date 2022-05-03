import os
import json
from datetime import date
from pprint import pprint
from calendar import monthrange
from dateutil.relativedelta import relativedelta as rdlt

import numpy as np
from pandas import DataFrame
import statsmodels.api as sm
from scipy.stats.stats import linregress
import matplotlib.pyplot as plt

from figs.bulk_analysis_figs import plot_clim_q_resid, plot_water_balance_trends
from hydrograph import hydrograph
from gage_analysis import EXCLUDE_STATIONS


def climate_flow_correlation(climate_dir, in_json, out_json, plot_r=None):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    print(len(l), 'station files')
    offsets = [x for x in range(1, 61)]
    windows = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        sid = os.path.basename(csv).strip('.csv')
        s_meta = metadata[sid]

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_pct = mean_irr * (1. / 1e6) / s_meta['AREA']

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
                r, p, b = lr.rvalue, lr.pvalue, lr.slope
                norm_slope = (b * (np.std(ind) / np.std(q))).item()
                r_dct[key_].append(r)
                if abs(r) > corr[1] and p < 0.05:
                    corr = (lag, abs(r))

                    response_d[key_] = {'q_window': q_win, 'slope': b,
                                        'norm_slope': norm_slope,
                                        'lag': lag, 'r': r, 'pval': p, 'irr_pct': irr_pct,
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


def get_sig_irr_impact(metadata, ee_series, out_jsn=None, fig_dir=None, gage_example=None, climate_sig_only=False):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    sig_stations = {}
    impacted_gages = []
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():

        if gage_example and sid != gage_example:
            continue

        s_meta = metadata[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        cdf = hydrograph(_file)

        if not os.path.exists(_file):
            continue
        if sid in EXCLUDE_STATIONS:
            continue

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

                cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])

                ai_c = sm.add_constant(ai)

                try:
                    ols = sm.OLS(q, ai_c)
                except Exception as e:
                    print(sid, e, cdf['q'].dropna().index[0])
                    continue

                fit_clim = ols.fit()

                irr_ct += 1
                # obj.item() to python objects
                clim_p = (fit_clim.pvalues[1]).item()
                if clim_p > p and clim_p > 0.05:
                    print('\n', sid, v['STANAME'], '{:.3f} p {:.3f} clim p'.format(p, clim_p), '\n')
                ct += 1
                resid = fit_clim.resid
                _cc_c = sm.add_constant(cci)
                ols = sm.OLS(resid, _cc_c)
                fit_resid = ols.fit()
                resid_p = fit_resid.pvalues[1]
                if resid_p < 0.05 or climate_sig_only:
                    if sid not in impacted_gages:
                        impacted_gages.append(sid)
                    if fit_resid.params[1] > 0.0:
                        slp_pos += 1
                    else:
                        slp_neg += 1

                    lr = linregress(resid, cci)
                    res_r, res_p = (lr.rvalue).item(), (lr.pvalue).item()

                    slope_resid = (fit_resid.params[1] * (np.std(cci) / np.std(resid))).item()
                    resid_line = fit_resid.params[1] * cci + fit_resid.params[0]

                    clim_line = fit_clim.params[1] * ai + fit_clim.params[0]
                    slope_clime = (fit_clim.params[1] * (np.std(ai) / np.std(q))).item()

                    desc_str = '{} {}\n' \
                               '{} months climate, flow months {}-{}\n' \
                               'crop consumption {} to {}\n' \
                               'p = {:.3f}, irr = {:.3f}, m = {:.2f}\n        '.format(sid, s_meta['STANAME'],
                                                                                       lag, q_start, q_end,
                                                                                       cc_start, cc_end,
                                                                                       fit_resid.pvalues[1],
                                                                                       v['irr_pct'],
                                                                                       slope_resid)

                    print(desc_str)
                    irr_sig_ct += 1
                    if fig_dir:
                        plot_clim_q_resid(q=q, ai=ai, clim_line=clim_line, desc_str=desc_str, years=years, cc=cci,
                                          resid=resid, resid_line=resid_line, fig_d=fig_dir, cci_per=cc_period,
                                          flow_per=(q_start, q_end))

                    if sid not in sig_stations.keys():
                        sig_stations[sid] = {k: v for k, v in s_meta.items() if not isinstance(v, dict)}
                    sig_stations[sid].update({'{}-{}'.format(cc_start, cc_end): {'res_sig': resid_p.item(),
                                                                                 'resid_slope': slope_resid,
                                                                                 'clim_r': r,
                                                                                 'clim_sig': clim_p,
                                                                                 'clim_slope': slope_clime,
                                                                                 'resid_r': res_r,
                                                                                 'lag': lag,
                                                                                 'q_window': k,
                                                                                 'q_data': list(q),
                                                                                 'ai_data': list(ai),
                                                                                 'q_ai_line': list(clim_line),
                                                                                 'cc_data': list(cci),
                                                                                 'q_resid': list(resid),
                                                                                 'q_resid_line': list(resid_line)
                                                                                 }})
                    years_c = sm.add_constant(years)
                    ols = sm.OLS(resid, years_c)
                    fit_resid_q = ols.fit()
                    lr = linregress(resid, years)
                    flow_r = (lr.rvalue).item()
                    resid_p_q = (fit_resid_q.pvalues[1]).item()
                    resid_line_q = fit_resid_q.params[1] * np.array(years) + fit_resid_q.params[0]
                    sig_stations[sid]['{}-{}'.format(cc_start, cc_end)].update({'q_time_r': flow_r,
                                                                                'q_time_sig': resid_p_q,
                                                                                'resid_q_time_line': list(
                                                                                    resid_line_q)})

    if gage_example:
        return sig_stations

    impacted_gages = list(set(impacted_gages))
    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)
    pprint(list(sig_stations.keys()))
    print('{} climate-sig, {} irrigated, {} irr imapacted periods, {} total'.format(ct, irr_ct, irr_sig_ct,
                                                                                    ct_tot))
    print('{} positive slope, {} negative'.format(slp_pos, slp_neg))
    print('total impacted gages: {}'.format(len(impacted_gages)))
    pprint(impacted_gages)


def basin_trends(key, metadata, ee_series, start_mo, end_mo, out_jsn=None, fig_dir=None, significant=True):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            continue
        s_meta = metadata[sid]
        if s_meta['AREA'] < 2000:
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

        if ols_fit.pvalues[1] > 0.05 and significant:
            continue
        else:
            ols_line = ols_fit.params[1] * _years + ols_fit.params[0]

            irr_ct += 1

            desc_str = '{} {}\n' \
                       'crop consumption {} to {}\n' \
                       'irr = {:.3f}\n        '.format(sid, s_meta['STANAME'],
                                                       5, 10,
                                                       v['IAREA'] / 1e6 * v['AREA'])

            irr_sig_ct += 1
            if fig_dir:
                plot_water_balance_trends(data=data, data_line=ols_line, data_str=key,
                                          years=years, desc_str=desc_str, fig_d=fig_dir)
            sig_stations[sid] = {'{}_data'.format(key): list(data),
                                 '{}_line'.format(key): list(ols_line)}

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f, indent=4, sort_keys=False)
            print('write {} sig {} to file'.format(len(sig_stations.keys()), key))


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_q_Comp_16DEC2021')
    clim_resp = os.path.join(root, 'station_metadata/basin_climate_response_27APR2022.json')

    clim_dir = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_q_Comp_16DEC2021')
    i_json = os.path.join(root, 'station_metadata/station_metadata.json')
    fig_dir_ = os.path.join(root, 'figures/clim_q_correlations')
    # climate_flow_correlation(climate_dir=clim_dir, in_json=i_json,
    #                           out_json=clim_resp, plot_r=fig_dir_)
    f_json = os.path.join(root, 'station_metadata', 'cci_impacted_all_q.json')
    get_sig_irr_impact(clim_resp, ee_data, f_json)
# ========================= EOF ====================================================================
