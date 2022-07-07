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


def climate_flow_correlation(climate_dir, in_json, out_json, plot_r=None, spec_time=None):
    """Find linear relationship between climate and flow in an expanding time window"""
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    print(len(l), 'station files')
    offsets = [x for x in range(1, 61)]
    windows = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)

    if abs(spec_time[1] - spec_time[0]) > 0:
        spec_time = tuple([x for x in range(spec_time[0], spec_time[1] + 1)])

    for csv in l:
        sid = os.path.basename(csv).strip('.csv')
        s_meta = metadata[sid]

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_pct = mean_irr * (1. / 1e6) / s_meta['AREA']

        years = [x for x in range(1991, 2021)]

        flow_periods = []
        max_len = 12
        rr = [x for x in range(1, 13)]
        for n in range(1, max_len + 1):
            for i in range(max_len):
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

            if spec_time:
                if q_win != spec_time:
                    continue

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

                etr = np.array([df['gm_etr'][d[0]: d[1]].sum() for d in dates])
                ppt = np.array([df['gm_ppt'][d[0]: d[1]].sum() for d in dates])
                ind = etr - ppt
                lr = linregress(ind, q)
                r, p, b = lr.rvalue, lr.pvalue, lr.slope
                norm_slope = (b * (np.std(ind) / np.std(q))).item()
                r_dct[key_].append(r)
                if abs(r) > corr[1] and p < 0.05:
                    corr = (lag, abs(r))

                    response_d[key_] = {'q_window': q_win, 'slope': b,
                                        'norm_slope': norm_slope,
                                        'q_data': list(q),
                                        'ai_data': list(ind),
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


# TODO: break this up into cc vs. residual and trends extraction functions
def get_sig_irr_impact(metadata, ee_series, out_jsn=None, fig_dir=None, gage_example=None, climate_sig_only=False,
                       monthly_trends=False):
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    slp_pos, slp_neg = 0, 0
    q_window = None
    sig_stations = {}
    impacted_gages = []
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for sid, v in metadata.items():

        if gage_example and sid != gage_example:
            continue

        s_meta = metadata[sid]
        _file = os.path.join(ee_series, '{}.csv'.format(sid))

        try:
            cdf = hydrograph(_file)
        except FileNotFoundError as e:
            print(sid, e)

        if not os.path.exists(_file):
            continue
        if sid in EXCLUDE_STATIONS:
            continue

        ct_tot += 1
        years = [x for x in range(1991, 2021)]

        print('\n', sid, v['STANAME'])
        # iterate over flow windows
        for k, v in s_meta.items():
            if not isinstance(v, dict):
                continue

            r, p, lag, q_window = (v[s] for s in ['r', 'pval', 'lag', 'q_window'])

            if q_window[1] < 5:
                lookback = True
            else:
                lookback = False

            q_start, q_end = q_window[0], q_window[-1]

            if v['irr_pct'] < 0.001:
                continue
            else:
                irr_ct += 1

            if p > 0.05 and not climate_sig_only:
                continue

            if lookback:
                q_dates = [(date(y, q_start, 1), date(y, q_end, monthrange(y, q_end)[1])) for y in years[1:]]
                clim_dates = [(date(y, q_end, monthrange(y, q_end)[1]) + rdlt(months=-lag),
                               date(y, q_end, monthrange(y, q_end)[1])) for y in years[1:]]
            else:
                q_dates = [(date(y, q_start, 1), date(y, q_end, monthrange(y, q_end)[1])) for y in years]
                clim_dates = [(date(y, q_end, monthrange(y, q_end)[1]) + rdlt(months=-lag),
                               date(y, q_end, monthrange(y, q_end)[1])) for y in years]

            ppt = np.array([cdf['gm_ppt'][d[0]: d[1]].sum() for d in clim_dates])
            etr = np.array([cdf['gm_etr'][d[0]: d[1]].sum() for d in clim_dates])
            ai = etr - ppt

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
            if monthly_trends:
                cc_periods = [q_window]

            for cc_period in cc_periods:
                cc_start, cc_end = cc_period[0], cc_period[-1]

                month_end = monthrange(2000, cc_end)[1]

                if monthly_trends and lookback:
                    cc_dates = [(date(y, cc_start, 1), date(y, cc_end, monthrange(y, cc_end)[1])) for y in years[1:]]
                elif lookback:
                    cc_dates = [(date(y - 1, cc_start, 1), date(y - 1, cc_end, month_end)) for y in years[1:]]
                else:
                    cc_dates = [(date(y, cc_start, 1), date(y, cc_end, month_end)) for y in years]

                if cc_dates[-1][1] > q_dates[-1][1]:
                    continue
                if cc_dates[-1][0] > q_dates[-1][0]:
                    continue

                q = np.array([cdf['q'][d[0]: d[1]].sum() for d in q_dates])

                cci = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])

                ai_c = sm.add_constant(ai)

                try:
                    ols = sm.OLS(q, ai_c)
                except Exception as e:
                    print(sid, e, cdf['q'].dropna().index[0])
                    continue

                fit_clim = ols.fit()

                # obj.item() to python objects
                clim_p = (fit_clim.pvalues[1]).item()
                # if clim_p > p and clim_p > 0.05:
                #     print('\n', sid, v['STANAME'], '{:.3f} p {:.3f} clim p'.format(p, clim_p), '\n')
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
                    try:
                        res_r, res_p = (lr.rvalue).item(), (lr.pvalue).item()
                    except AttributeError:
                        res_r, res_p = lr.rvalue, lr.pvalue

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

                    # print(desc_str)
                    if resid_p < 0.05:
                        irr_sig_ct += 1
                        print(sid, cc_period, q_window, '{:.3f}'.format(slope_resid))

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
                    if lookback:
                        yrs = years[1:]
                    else:
                        yrs = years

                    years_c = sm.add_constant(yrs)
                    ols = sm.OLS(resid, years_c)
                    fit_resid_q = ols.fit()
                    lr = linregress(resid, yrs)
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
    # pprint(list(sig_stations.keys()))
    print('q window {}'.format(q_window))
    print('{} climate-sig, {} irrigated, {} irr imapacted periods, {} total'.format(ct, irr_ct, irr_sig_ct,
                                                                                    ct_tot))
    # print('{} positive slope, {} negative'.format(slp_pos, slp_neg))
    # print('total impacted gages: {}'.format(len(impacted_gages)))
    # pprint(impacted_gages)


def test_lr_trends(in_json, out_json):
    with open(in_json, 'r') as f:
        stations = json.load(f)

    diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in stations.items()]
    diter = [i for ll in diter for i in ll]

    regressions = {}

    trc_subdirs = ['time_cc', 'time_qres', 'time_ai', 'time_q']
    counts = {k: [0, 0] for k in trc_subdirs}

    for station, period, records in diter:

        try:
            q = np.array(records['q_data'])
            qres = np.array(records['q_resid'])
            cc = np.array(records['cc_data'])
            ai = np.array(records['ai_data'])

            qres = (qres - qres.min()) / (qres.max() - qres.min()) + 0.001
            years = (np.linspace(0, 1, len(qres)) + 0.001) + 0.001

            regression_combs = [(years, cc),
                                (years, qres),
                                (years, ai),
                                (years, q)]

            regressions[station] = {k: {} for k in trc_subdirs}

            for (x, y), subdir in zip(regression_combs, trc_subdirs):

                x_c = sm.add_constant(x)
                lr = sm.OLS(y, x_c)
                lr = lr.fit()
                r, p = lr.rsquared, lr.pvalues[1]
                b = (lr.params[1] * (np.std(x) / np.std(y))).item()
                if p < 0.05:
                    if b > 0:
                        counts[subdir][1] += 1
                    else:
                        counts[subdir][0] += 1
                regressions[station][subdir].update({'b': b, 'p': p, 'rsq': r})

        except Exception as e:
            print(e, station, period)

    print('\n {}'.format(in_json))
    pprint(counts)

    with open(out_json, 'w') as f:
        json.dump(regressions, f, indent=4, sort_keys=False)


def summarize_cc_qres(_dir, out_json):
    dct = {}
    _files = [os.path.join(_dir, 'impacts_{}_cc_test.json'.format(m)) for m in range(1, 13)]

    for m, f in enumerate(_files, start=1):
        insig, sig = 0, 0
        with open(f, 'r') as fp:
            d_obj = json.load(fp)

        diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in d_obj.items()]
        diter = [i for ll in diter for i in ll]
        for k, cc, d in diter:
            if d['res_sig'] > 0.05:
                insig += 1
                continue
            sig += 1
            if k not in dct.keys():
                dct[k] = {m: {cc: d['resid_slope']}}
            elif m not in dct[k].keys():
                dct[k][m] = {cc: d['resid_slope']}
            else:
                dct[k][m].update({cc: d['resid_slope']})

        print('month ', m, 'sig', sig, 'insig', insig)

    with open(out_json, 'w') as fp:
        json.dump(dct, fp, indent=4)


def summarize_trends(_dir):
    _files = [os.path.join(_dir, 'linear_regressions_{}.json'.format(m)) for m in range(1, 13)]

    for var in ['time_cc', 'time_qres', 'time_ai', 'time_q']:
        dct = {}
        for m, f in enumerate(_files, start=1):

            if var == 'time_cc':
                if m > 10 or m < 4:
                    continue

            insig, sig = 0, 0
            with open(f, 'r') as fp:
                d_obj = json.load(fp)

            diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in d_obj.items()]
            diter = [i for ll in diter for i in ll]

            for k, v, d in diter:
                if v != var:
                    continue
                if d['p'] > 0.05:
                    insig += 1
                    continue
                sig += 1
                slope = d['b']
                if np.isnan(slope):
                    continue
                if k not in dct.keys():
                    dct[k] = {m: {v: slope}}
                elif m not in dct[k].keys():
                    dct[k][m] = {v: slope}
                else:
                    dct[k][m].update({v: slope})

            print(var, 'month ', m, 'sig', sig, 'insig', insig)

        out_json = os.path.join(_dir, '{}_summary.json'.format(var))
        with open(out_json, 'w') as fp:
            json.dump(dct, fp, indent=4)


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')

    clim_dir = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021')
    i_json = os.path.join(root, 'station_metadata/station_metadata.json')
    fig_dir_ = os.path.join(root, 'figures/clim_q_correlations')

    for m in range(1, 13):
        clim_resp = os.path.join(root, 'gridmet_analysis', 'analysis',
                                 'basin_climate_response_{}.json'.format(m))

        if not os.path.exists(clim_resp):
            climate_flow_correlation(climate_dir=clim_dir, in_json=i_json,
                                     out_json=clim_resp, plot_r=None, spec_time=(m, m))

        f_json = os.path.join(root, 'gridmet_analysis', 'analysis', 'trend_data_{}.json'.format(m))
        # f_json = os.path.join(root, 'gridmet_analysis', 'analysis', 'impacts_{}_cc_test.json'.format(m))

        # if not os.path.exists(f_json):
        # get_sig_irr_impact(clim_resp, ee_data, f_json, climate_sig_only=True, monthly_trends=True)

        trends_summary_json = os.path.join(root, 'gridmet_analysis', 'analysis',
                                           'linear_regressions_{}.json'.format(m))
        # test_lr_trends(f_json, trends_summary_json)

    summary_cc_qres = os.path.join(root, 'gridmet_analysis', 'analysis', 'cc_qres_summary.json')
    results_d = f_json = os.path.join(root, 'gridmet_analysis', 'analysis')
    summarize_cc_qres(results_d, summary_cc_qres)

    # summarize_trends(results_d)

# ========================= EOF ====================================================================
