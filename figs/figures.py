import os
import sys
import json
import pickle
from itertools import product

import numpy as np
from scipy.stats.stats import linregress

import fiona
import geopandas as gpd
from shapely.geometry import shape
from scipy.interpolate import make_interp_spline

from matplotlib import cm, colors
from matplotlib import pyplot as plt
import seaborn as sns
import arviz as az

from gage_analysis import EXCLUDE_STATIONS
from figs.regression_figs import plot_regression_from_trace
import bayes_analysis

# temporary hack to open pickled model with renamed module
sys.modules['linear_regression_errors'] = bayes_analysis

SYSTEM_STATIONS = ['06109500', '06329500', '09180500', '09315000',
                   '09379500', '12396500', '13269000', '13317000']

SELECTED_SYSTEMS = ['06109500', '06329500', '09180500', '09315000',
                    '09379500', '09466500', '12389000', '12510500',
                    '13269000', '13317000', '14048000',
                    '14103000', '14211720']

large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large,
          'xtick.color': 'black',
          'ytick.color': 'black',
          'xtick.direction': 'out',
          'ytick.direction': 'out',
          'xtick.bottom': True,
          'xtick.top': False,
          'ytick.left': True,
          'ytick.right': False,
          }


def response_time_to_area(climate_resp, fig_dir):
    with open(climate_resp, 'r') as fp:
        c_dct = json.load(fp)

    irr_ids = []
    c_lags = []
    i_lags = []
    i_areas = []
    c_areas = []
    for k, v in c_dct.items():
        impact_keys = [v for p, v in v.items() if isinstance(v, dict)]
        irr_pct = impact_keys[0]['irr_pct']
        if irr_pct > 0.01:
            irr_ids.append(k)
            i_lags.append(np.median([d['lag'] for d in impact_keys]))
            i_areas.append(v['AREA'])
        else:
            c_lags.append(np.median([d['lag'] for d in impact_keys]))
            c_areas.append(v['AREA'])

    plt.hist(c_lags, bins=20, color='r', label='Unirrigated Basins')
    plt.hist(i_lags, bins=20, color='b', label='Irrigated Basins')
    plt.legend()
    plt.suptitle('Basin Median Response Time')
    plt.ylabel('Response Time [months]')
    plt.xlabel('Area [sq km]')
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'slope_irr.png'))
    plt.close()


def fraction_cc_water_balance_fig(metadata, ee_series, fig):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    frac = []
    suspect = []
    # TODO use json created in gage_analysis
    print(len(frac), 'irrigated basins')
    # print(min(frac), max(frac), np.mean(frac))
    lim_ = np.round(max(frac), decimals=2)
    bins = np.linspace(0, lim_, 10)
    plt.xlim(0, lim_)
    plt.hist(frac, bins=bins, histtype='barstacked', rwidth=0.95)
    plt.title('cc_ratio_q\n{} Basins'.format(len(frac)))
    plt.xlabel('Crop Consumption Fraction')
    plt.ylabel('Count')
    plt.savefig(fig)
    print(suspect)


def impact_time_series_bars(sig_stations, basin_designations, figures, min_area=None,
                            sort_key='cci', x_key='cci'):
    cmap = cm.get_cmap('RdYlGn')

    with open(sig_stations, 'r') as f:
        stations = json.load(f)

    with open(basin_designations, 'r') as f:
        basins = json.load(f)
        _ = {}
        for k, v in basins.items():
            l = [x for x in v if x in stations.keys()]
            _[k] = l
        basins = _

    if min_area:
        filtered_stations = [k for k, v in stations.items() if v['AREA'] > min_area]
    else:
        filtered_stations = SELECTED_SYSTEMS

    for basin, _stations in basins.items():

        sort_keys = sorted(_stations, key=lambda x: stations[x][sort_key], reverse=False)
        sort_keys = [s for s in sort_keys if s in filtered_stations]
        print('{} {} gages'.format(basin, len(sort_keys)))
        all_x = [stations[k][x_key] for k in sort_keys]
        all_slope = []

        for s, v in stations.items():
            if s not in sort_keys:
                continue
            impact_keys = [p for p, v in v.items() if isinstance(v, dict)]
            [all_slope.append(stations[s][k]['resid_slope']) for k in v.keys() if k in impact_keys]

        min_slope, max_slope = np.min(all_slope), np.max(all_slope)

        vert_increment = 1 / 7.
        v_position = 0.02
        fig, axes = plt.subplots(figsize=(12, 18))
        fig.subplots_adjust(left=0.4)
        ytick, ylab = [], []
        xmin, xmax, xstd, xmean = np.min(all_x), np.max(all_x), np.std(all_x), np.mean(all_x)
        xrange = (xmax + xstd * 2) - (xmin - xstd)
        h_increment = xrange / 5.
        width = h_increment / 6.

        for sid in sort_keys:
            if sid in EXCLUDE_STATIONS:
                continue
            dct = stations[sid]
            impact_keys = [p for p, v in dct.items() if isinstance(v, dict)]

            periods = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in impact_keys]
            periods = sorted(periods, key=lambda x: x[1] - x[0])

            slopes = [dct[k]['resid_slope'] for k in dct.keys() if k in impact_keys]
            [all_slope.append(s) for s in slopes]

            single_month_resp = [x[1] - x[0] == 0 for x in periods]
            periods = [p[0] for p, s in zip(periods, single_month_resp) if s]
            slopes = [sl for sl, s in zip(slopes, single_month_resp) if s]
            if np.all(slopes == 0.0) or not slopes:
                continue
            center = dct[x_key]
            months, locations = [x for x in range(5, 11)], list(np.linspace(h_increment * 0.5, -h_increment * 0.5, 6))
            for m, p in zip(months, locations):
                if m in periods:
                    s = slopes[periods.index(m)]
                    slope_scale = (s - min_slope) / (max_slope - min_slope)
                    color = cmap(slope_scale)
                    axes.barh(v_position, left=(center - p), width=width, height=vert_increment, color=color,
                              edgecolor=color, align='edge', alpha=0.5)
                else:
                    axes.barh(v_position, left=(center - p), width=width, height=vert_increment, color='none',
                              edgecolor='k', align='edge', alpha=0.2)

            tick_pos = v_position + (vert_increment * 0.5)
            ytick.append(tick_pos), ylab.append('{}\n {}: {:.3f}'.format(dct['STANAME'], sort_key,
                                                                         np.log10(dct[sort_key])))
            v_position += vert_increment + 0.02

        plt.yticks(ytick, ylab)
        # x_tick = list(np.linspace(5.5, 11.5, 6))
        # x_tick_lab = [x for x in range(5, 11)]
        # plt.yticks(ytick, ylab)
        # plt.xticks(x_tick, x_tick_lab)
        plt.xlabel(x_key)
        dir_ = os.path.join(figures, '{}_{}_system'.format(sort_key, x_key))
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
        fig_file = os.path.join(dir_, '{}_{}_{}_10DEC2021.png'.format(basin, sort_key, x_key))
        plt.xlim([(xmin - xstd), (xmax + xstd * 2)])
        plt.ylim([0, v_position])
        plt.suptitle('Irrigation Impact on Gages: {}\n sorted by {}'.format(basin, sort_key))
        # plt.show()
        plt.savefig(fig_file)
        plt.close()

        print('\nall slopes min {:.3f}, max {:.3f}'.format(min(all_slope), max(all_slope)))
        print(' slope used min {:.3f}, max {:.3f}'.format(min_slope, max_slope))


def trends_panel(irr_impact, clim_flow_d, png, x_err, y_err, filter_json=None):
    sns.set_theme()

    with open(irr_impact, 'r') as f:
        irr_impact_d = json.load(f)
    irr_stations = list(irr_impact_d.keys())

    with open(clim_flow_d, 'r') as f:
        clim_flow_d = json.load(f)

    if filter_json:
        with open(filter_json, 'r') as f:
            filter_json = json.load(f)

    # build data for slope, r histograms
    clime_slope, resid_slope, clim_r, resid_r = [], [], [], []
    t_cci_r_, t_cci_slope_ = [], []
    t_q_r_, t_q_slope_ = [], []
    years = np.array([x for x in range(1991, 2021)])

    clim_slope_insig, resid_slope_insig, t_cci_slope_insig, t_q_slope_insig = [], [], [], []
    for s, d in clim_flow_d.items():

        if s in EXCLUDE_STATIONS:
            continue

        impact_keys = [p for p, v in d.items() if isinstance(v, dict)]
        for k in d.keys():
            if k in impact_keys and s in irr_stations:
                if clim_flow_d[s][k]['pval'] < 0.05:
                    clime_slope.append(clim_flow_d[s][k]['slope'])
                    clim_r.append(clim_flow_d[s][k]['r'])
                else:
                    clim_slope_insig.append((s, k))

    for s, d in irr_impact_d.items():

        if s in EXCLUDE_STATIONS:
            continue

        impact_keys = [p for p, v in d.items() if isinstance(v, dict)]
        [resid_slope.append(irr_impact_d[s][k]['resid_slope']) for k in d.keys() if k in impact_keys]
        [resid_r.append(irr_impact_d[s][k]['resid_r']) for k in d.keys() if k in impact_keys]
        # TODO: write q_resid_time m/r/p and cci_time m/r/p to json from gage analysis

        for k, v in d.items():
            if k in impact_keys:
                bayes_est = d[k]['cc_qres']
                if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                    resid_slope.append(bayes_est['mean'])
                else:
                    resid_slope_insig.append(bayes_est['mean'])

                bayes_est = d[k]['time_cc']
                if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                    t_cci_slope_.append(bayes_est['mean'])
                else:
                    t_cci_slope_insig.append(bayes_est['mean'])

                bayes_est = d[k]['time_qres']
                if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                    t_q_slope_.append(bayes_est['mean'])
                else:
                    t_q_slope_insig.append(bayes_est['mean'])

    n_bins = 30

    bin_clime_r, bin_clime_slope = np.linspace(0.1, 0.9, n_bins), \
                                   np.linspace(-0.9, 0.75, n_bins)

    bin_resid_r, bin_resid_slope = np.linspace(-0.7, 0.7, n_bins), \
                                   np.linspace(-0.7, 0.7, n_bins)

    bin_t_cci_r, bin_t_cci_slope = np.linspace(-0.6, 0.8, n_bins), \
                                   np.linspace(-0.66, 0.95, n_bins)

    bin_t_q_res_r, bin_t_q_res_slope = np.linspace(-0.8, 0.8, n_bins), \
                                       np.linspace(-0.8, 0.8, n_bins)

    for s, d in irr_impact_d.items():

        impact_keys = [p for p, v in d.items() if isinstance(v, dict)]
        for k, v in d.items():
            if k in impact_keys:

                if filter_json:
                    if filter_json[s] == 'None':
                        continue
                    if k not in filter_json[s].keys():
                        continue
                # if s != '06016000' or k != '5-7':
                #     continue
                try:
                    figname = os.path.join(png, '{}_{}_cc_{}_q_{}_{}_mo_climate.png'.format(s, d['STANAME'],
                                                                                            k, v['q_window'],
                                                                                            v['lag']))

                    lr_cc = linregress(years, d[k]['cc_data'])
                    t_cci_r, t_cci_slope, t_cci_p = lr_cc.rvalue, lr_cc.slope, lr_cc.pvalue
                    lr_q_t = linregress(years, d[k]['q_resid_line'])
                    t_q_r, t_q_slope, t_q_p = lr_q_t.rvalue, lr_q_t.slope, lr_q_t.pvalue

                    # if t_cci_p < 0.05 and t_q_p < 0.05 and d['AREA'] > 2000.:
                    print(s, d['STANAME'], d['AREA'])

                    q, ai, clim_line = v['q_data'], v['ai_data'], v['q_ai_line']

                    fig, ax = plt.subplots(3, 1)
                    fig.set_figheight(16)
                    fig.set_figwidth(16)

                    # climate-flow data
                    ax[0].scatter(ai, q)
                    ax[0].plot(ai, clim_line)
                    ax[0].set(xlabel='ETr / PPT [-]')
                    ax[0].set(ylabel='q [m^3]')

                    ax[1].hist(clime_slope, bins=bin_clime_slope)
                    ax[1].set(xlabel='Climate-Flow Slope [-]')
                    ax[1].set(ylabel='count')
                    ax[1].text(0.8, 0.8, 'n={}'.format(len(clime_slope)))

                    ax[2].hist(clim_r, bins=bin_clime_r)
                    ax[2].set(xlabel='Climate-Flow Pearson r [-]')
                    ax[2].set(ylabel='count')
                    ax[2].text(0.8, 0.8, 'n={}'.format(len(clim_r)))
                    plt.savefig(figname.replace('.png', '_qclim.png'))
                    plt.close()
                    # plt.show()

                    ######################################################
                    # residual flow - crop consumption data
                    fig, ax = plt.subplots(3, 1)
                    fig.set_figheight(16)
                    fig.set_figwidth(24)

                    cci, resid, resid_line = np.array(v['cc_data']), np.array(v['q_resid']), v['q_resid_line']

                    ccin = (cci - cci.min()) / (cci.max() - cci.min())
                    qres = (resid - resid.min()) / (resid.max() - resid.min())
                    ax[0].set(xlabel='cci [m]')
                    ax[0].set(ylabel='q residual [m^3]')
                    ax[0].scatter(ccin, qres)
                    ax[0].errorbar(ccin, qres, xerr=x_err, yerr=y_err, alpha=0.1, ls='', color='b')
                    ax[0].set_ylim([0, 1])
                    ax[0].set_xlim([-0.1, 1.1])
                    cc_err = x_err * np.ones_like(ccin)
                    qres_err = y_err * np.ones_like(qres)

                    with open(d[k]['cc_qres']['model'], 'rb') as buff:
                        mdata = pickle.load(buff)
                        model, trace = mdata['model'], mdata['trace']

                    diverge = trace.diverging.reshape(4, -1)
                    diverge_sum = diverge.sum(axis=1)
                    div_chain = np.array(diverge_sum, dtype=float) / (np.ones_like(diverge_sum) * diverge.shape[1])
                    drop_chain = div_chain < 0.1
                    chain_idx = [i for i, x in enumerate(drop_chain) if x]

                    _ = plot_regression_from_trace(model, (ccin, qres, cc_err, qres_err),
                                                   ax=ax[0], chains=4, traces=trace, legend=False,
                                                   chain_idx=chain_idx)

                    az.plot_posterior(trace, var_names=['slope'], hdi_prob=0.95, ax=ax[1],
                                      coords={'chain': chain_idx})
                    ax[1].set(xlabel='Slope')

                    pos_, neg_ = [x for x in resid_slope if x > 0], [x for x in resid_slope if x < 0]
                    ax[2].hist([pos_, neg_, resid_slope_insig], color=['blue', 'red', 'white'],
                               stacked=True, bins=bin_resid_slope)
                    ax[2].set(xlabel='Residual Flow-Crop Consumption Slope [-]')
                    ax[2].set(ylabel='count')
                    ax[0].set_xlim([0, 1])
                    # plt.show()
                    plt.savefig(figname.replace('.png', '_qcc.png'))
                    plt.close()

                    ######################################################
                    # crop consumption trend data
                    fig, ax = plt.subplots(3, 1)
                    fig.set_figheight(16)
                    fig.set_figwidth(24)

                    years = (np.linspace(0, 1, len(qres)) + 0.001)
                    years_err = np.zeros_like(cc_err)
                    ax[0].set(xlabel='Time')
                    ax[0].set(ylabel='cci [m]')
                    ax[0].scatter(years, ccin)
                    ax[0].errorbar(years, ccin, xerr=0.0, yerr=y_err, alpha=0.1, ls='', color='b')

                    with open(d[k]['time_cc']['model'], 'rb') as buff:
                        mdata = pickle.load(buff)
                        model, trace = mdata['model'], mdata['trace']

                    _ = plot_regression_from_trace(model, (years, ccin, years_err, cc_err),
                                                   ax=ax[0], chains=4, traces=trace, legend=False)

                    az.plot_posterior(trace, var_names=['slope'], hdi_prob=0.95, ax=ax[1])
                    ax[1].set(xlabel='Slope')

                    ax[2].hist(t_cci_slope_, bins=bin_t_cci_slope)
                    ax[2].set(xlabel='Crop Consumption Trend Slope [-]')
                    ax[2].set(ylabel='count')
                    plt.savefig(figname.replace('.png', '_tcc.png'))
                    plt.close()

                    ######################################################
                    # residual flow trend data
                    fig, ax = plt.subplots(3, 1)
                    fig.set_figheight(16)
                    fig.set_figwidth(24)

                    ax[0].set(xlabel='Time')
                    ax[0].set(ylabel='qres [m]')
                    ax[0].scatter(years, qres)
                    ax[0].errorbar(years, qres, xerr=0.0, yerr=y_err, alpha=0.1, ls='', color='b')

                    with open(d[k]['time_qres']['model'], 'rb') as buff:
                        mdata = pickle.load(buff)
                        model, trace = mdata['model'], mdata['trace']

                    _ = plot_regression_from_trace(model, (years, qres, years_err, qres_err),
                                                   ax=ax[0], chains=4, traces=trace, legend=False)

                    az.plot_posterior(trace, var_names=['slope'], hdi_prob=0.95, ax=ax[1])
                    ax[1].set(xlabel='Slope')

                    ax[2].hist(t_q_slope_, bins=bin_t_q_res_slope)
                    ax[2].set(xlabel='Residual Flow Trend Slope [-]')
                    ax[2].set(ylabel='count')

                    plt.savefig(figname.replace('.png', '_tqres.png'))
                    plt.close()

                except ValueError as e:
                    print(e)


def delta_cn_q_map_annual(bayes_, in_shp, out_shp, polys, variable):
    study_area = gpd.read_file(polys)

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        feats = [f for f in src]

    geo = {f['properties']['STAID']: shape(f['geometry']) for f in feats}
    areas = {f['properties']['STAID']: f['properties']['AREA'] for f in feats}

    trends_dct = {}

    with open(bayes_, 'r') as qcc:
        cc_dct = json.load(qcc)

    for k, v in cc_dct.items():

        if k in EXCLUDE_STATIONS:
            continue

        if isinstance(v, dict):
            for kk, vv in v.items():

                if isinstance(vv, dict):
                    try:
                        bayes_est = vv[variable]
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                            sig_t_q_b_val = bayes_est['mean']
                            trends_dct.update({k: sig_t_q_b_val})
                        else:
                            trends_dct.update({k: np.nan})

                    except KeyError:
                        trends_dct.update({k: np.nan})

    marker_min = 1
    marker_max = 40

    data, columns = [x[1] for x in trends_dct.items()], [x[0] for x in trends_dct.items()]
    gdf = gpd.GeoDataFrame(data=data, index=columns, columns=['vals'])
    gdf['AREA'] = [areas[_id] for _id in gdf.index]
    gdf.geometry = [geo[_id] for _id in gdf.index]
    gdf.dropna(subset=['vals'], inplace=True)
    cmap = 'coolwarm_r'
    fig, ax = plt.subplots()

    areas = np.log(gdf['AREA'])
    areas = (areas - areas.min()) / (areas.max() - areas.min()) * marker_max + marker_min
    study_area.plot(ax=ax, **{'edgecolor': 'k', 'facecolor': (0, 0, 0, 0)})
    gdf.plot('vals', cmap=cmap, norm=colors.CenteredNorm(), s=areas.values, ax=ax)

    plt.savefig('/home/dgketchum/Downloads/{}.png'.format(os.path.basename(out_shp).replace('.shp', '.png')))
    plt.close()
    pass


def delta_cn_q_map_monthly(bayes_, in_shp, out_shp, polys, var='qnorm'):
    study_area = gpd.read_file(polys)

    with fiona.open(in_shp, 'r') as src:
        meta = src.meta
        feats = [f for f in src]

    geo = {f['properties']['STAID']: shape(f['geometry']) for f in feats}
    areas = {f['properties']['STAID']: f['properties']['AREA'] for f in feats}

    trends_dct = {}

    if var == 'time_qres':
        file_form = 'bayes_impacts_summerflow_{}_qnorm_qreserr_0.17.json'
    elif var == 'time_ai':
        file_form = 'bayes_trend_ai_{}_qreserr_0.17.json'

    for m in range(1, 13):

        trends = os.path.join(bayes_, file_form.format(m))

        with open(trends, 'r') as qcc:
            cc_dct = json.load(qcc)

        sig_t_q_gages, sig_t_q_ct, sig_t_q_b, = [], 0, []

        for k, v in cc_dct.items():

            if k in EXCLUDE_STATIONS:
                continue

            if isinstance(v, dict):
                for kk, vv in v.items():

                    if isinstance(vv, dict):
                        try:
                            bayes_est = vv[var]
                        except KeyError as e:
                            # print(e, k, m)
                            pass
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                            sig_t_q_gages.append(k)
                            sig_t_q_ct += 1
                            sig_t_q_b_val = bayes_est['mean']

                            if k not in trends_dct.keys():
                                trends_dct.update({k: {m: sig_t_q_b_val}})
                            else:
                                trends_dct[k].update({m: sig_t_q_b_val})

                            sig_t_q_b.append((sig_t_q_b_val, v['AREA']))
                        else:
                            if k not in trends_dct.keys():
                                trends_dct.update({k: {m: np.nan}})
                            else:
                                trends_dct[k].update({m: np.nan})

    for k, v in trends_dct.items():
        l = []
        for m in range(1, 13):
            if m not in v.keys():
                trends_dct[k].update({m: np.nan})
            l.append(trends_dct[k][m])
        trends_dct[k]['cntrnd_mn'] = np.nanmean(l)

    for k, v in trends_dct.items():
        for kk, vv in v.items():
            if kk == 'cntrnd_mn':
                continue
            if np.isnan(vv):
                trends_dct[k][kk] = 0.0

    marker_min = 1
    marker_max = 40

    gdf = gpd.GeoDataFrame(data=trends_dct).transpose()
    gdf['AREA'] = [areas[_id] for _id in gdf.index]
    gdf.geometry = [geo[_id] for _id in gdf.index]
    cmap = 'coolwarm_r'
    rows, cols = 4, 3
    fig, ax = plt.subplots(rows, cols, figsize=(20, 20))
    idxs = [i for i in product([i for i in range(rows)], [i for i in range(cols)])]

    for idx, m in zip(idxs, range(1, 13)):
        areas = np.log(gdf['AREA'])
        areas = (areas - areas.min()) / (areas.max() - areas.min()) * marker_max + marker_min
        study_area.plot(ax=ax[idx], **{'edgecolor': 'k', 'facecolor': (0, 0, 0, 0)})
        gdf.plot(m, cmap=cmap, norm=colors.CenteredNorm(), s=areas.values, ax=ax[idx])

    plt.savefig('/home/dgketchum/Downloads/{}_trend_1_12.png'.format(var))
    plt.close()
    pass


def delta_cn_q_kde(bayes_, climate_, out_png):
    plt.rcParams.update(params)
    plt.style.use('seaborn-whitegrid')
    sns.set_theme(style="white")

    for i, (bayes, climate) in enumerate(zip(bayes_, climate_)):

        with open(climate, 'r') as clm:
            clim_dct = json.load(clm)

        with open(bayes, 'r') as qcc:
            cc_dct = json.load(qcc)

        cc_gages = list(cc_dct)

        sig_clim_q_gages, sig_clim_q_ct, lag = [], 0, []
        area = []
        for k, v in clim_dct.items():
            if k not in cc_gages:
                continue
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        if vv['pval'] < 0.05:
                            sig_clim_q_ct += 1
                            sig_clim_q_gages.append(k)
                            lag.append(vv['lag'])
                            area.append(v['SQMI'])
        q_clim_set = set(sig_clim_q_gages)
        print('{} gages'.format(len(q_clim_set)))
        sig_t_q_gages, sig_t_q_ct, sig_t_q_b, = [], 0, []

        for k, v in cc_dct.items():

            if k in EXCLUDE_STATIONS:
                continue

            if isinstance(v, dict):
                for kk, vv in v.items():

                    if isinstance(vv, dict):
                        bayes_est = vv['time_qres']
                        if np.sign(bayes_est['hdi_2.5%']) == np.sign(bayes_est['hdi_97.5%']):
                            sig_t_q_gages.append(k)
                            sig_t_q_ct += 1
                            sig_t_q_b_val = bayes_est['mean']
                            sig_t_q_b.append((sig_t_q_b_val, v['AREA']))

        bins = np.linspace(-1, 1, 21)
        sums_ = np.array([0 for _ in bins])
        for v, a in sig_t_q_b:
            idx = np.argmin(abs(v - bins))
            sums_[idx] += a
        dens = (sums_ - min(sums_)) / (max(sums_) - min(sums_))
        spline = make_interp_spline(bins, dens)
        x = np.linspace(bins.min(), bins.max(), 500)
        y = spline(x)
        plt.plot(y + i, x)

    plt.hlines(0, 0, 12, linestyles='dashed')
    plt.savefig(out_png)
    plt.close()
    print()


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    figs = os.path.join(root, 'figures')
    climr = [os.path.join(root, 'station_metadata', 'flowtrends',
                          'basin_climate_response_{}_7JUN2022.json'.format(m)) for m in range(1, 13)]
    bayes_ = [os.path.join(root, 'station_metadata', 'flowtrends',
                           'bayes_impacts_summerflow_{}_qnorm_{}_qreserr_0.17.json'.format(m, m)) for m in range(1, 13)]

    out_ = os.path.join('/home/dgketchum/Downloads', 'basin_delQ_7JUN2022.png')
    # delta_cn_q_kde(bayes_, climr, out_)

    study_area_ = '/media/research/IrrigationGIS/gages/figures/fig_shapes/study_basins.shp'

    inshp = '/media/research/IrrigationGIS/gages/gage_loc_usgs/selected_gages.shp'
    aa = '/media/research/IrrigationGIS/gages/watersheds/bayes_trend_ai_1_12.shp'
    a = os.path.join(root, 'station_metadata', 'flowtrends')
    # vars = ['time_ai', 'time_qres']

    delta_cn_q_map_monthly(a, inshp, aa, polys=study_area_, var='time_qres')

    # climr = os.path.join(root, 'station_metadata', 'flowtrends', 'basin_climate_response_1_12_7JUN2022.json')
    # a = os.path.join(root, 'station_metadata', 'flowtrends', 'bayes_trend_ai_1_12_qreserr_0.17.json')
    # b = os.path.join(root, 'station_metadata', 'flowtrends', 'bayes_trend_qnorm_1_12_qreserr_0.17.json')
    #
    # bb = '/media/research/IrrigationGIS/gages/watersheds/bayes_trend_qnorm_1_12.shp'
    #
    #
    # for trend, outshp, var in zip([a, b], [aa, bb], vars):
    #     delta_cn_q_map_annual(trend, inshp, outshp, polys=study_area_, variable=var)

# ========================= EOF ====================================================================
