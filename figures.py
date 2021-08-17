import os
import json

from pandas import read_csv, to_datetime, DatetimeIndex, date_range
import numpy as np
import matplotlib
from matplotlib import colors, cm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from pylab import rcParams
import statsmodels.tools.sm_exceptions as sm_exceptions
import statsmodels.api as sm
from PyEMD import EMD
from hydrograph import hydrograph
from tables import CLMB_STATIONS


def plot_station_hydrographs(csv, fig_dir=None):
    df = read_csv(csv, index_col=0, parse_dates=True)
    df.index.freq = 'A'
    df[df == 0.0] = np.nan
    df = df.dropna(axis=1, how='any')
    linear = [x.year for x in df.index]
    means = df.mean(axis=0)
    z_totals = df.div(means, axis=1)
    z_totals.index = linear
    for i, r in df.iteritems():
        fig, ax = plt.subplots()
        r.index = linear
        cycle, trend = sm.tsa.filters.hpfilter(r.values, lamb=6.25)
        imf = EMD().emd(r.values, linear)
        ax = r.plot(ax=ax, kind='line', x=linear, y=r.values, alpha=0.6)
        ax = plt.plot(linear, trend)
        ax = plt.plot(linear, imf[-1, :])
        # plt.show()
        plt.savefig(os.path.join(fig_dir, '{}.png'.format(r.name[5:13])))
        print(r.name[5:13])


def plot_log_volumes(csv_dir, fig_dir, metadata):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    l.sort()
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for f in l:
        if '06192500' not in f:
            continue
        sid = os.path.basename(f).split('.')[0]
        df = read_csv(f, parse_dates=True, index_col='datetimeUTC')
        df = df.rename(columns={list(df.columns)[0]: 'q'})
        df[df <= 0.0] = 1.0
        df = np.log10(df)
        if not np.all(df['cc'].values) > 0:
            print(sid, ' unirrigated')
            continue
        df.plot(y=['cc', 'pr', 'etr', 'q', 'irr'])
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{}: 10^{:.2f} sq km'.format(meta[sid]['STANAME'], np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        print(figname)


def plot_irrigated_fraction(dir_):
    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]
    irr_idx = []
    for c in l:
        sid = os.path.basename(c).split('_')[0]
        df = read_csv(c)
        df['datetimeUTC'] = to_datetime(df['datetimeUTC'])
        df = df.set_index('datetimeUTC')
        df.rename(columns={list(df.columns)[0]: 'q'}, inplace=True)
        df['if'] = df['cc'] / df['q']
        frac = df['cc'].sum() / df['q'].sum()
        irr_idx.append(frac)
        print(sid, frac)


def plot_bf_time_series(daily_q_dir, fig_dir, metadata, start_month=None, end_month=None):
    s, e = '1984-01-01', '2020-12-31'
    idx = DatetimeIndex(date_range(s, e, freq='D'))
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)
        if start_month or end_month:
            idx = idx[idx.month.isin([x for x in range(start_month, end_month)])]
            df = df[df.index.month.isin([x for x in range(start_month, end_month)])]

        df.plot(y=['q', 'qb'])
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{} ({}): 10^{:.2f} sq km'.format(meta[sid]['STANAME'], sid, np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        # plt.show()
        print(figname)


def plot_gridded_time_series(daily_q_dir, fig_dir, metadata):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)
        irr = df['irr'].values
        irr_area = irr.mean()
        if irr_area < 0.001:
            continue
        df['intensity'] = (df['cc'] / df['irr']).values
        df.plot(y='intensity')
        figname = os.path.join(fig_dir, '{}.png'.format(sid))
        plt.title('{} ({}): 10^{:.2f} sq km'.format(meta[sid]['STANAME'], sid,
                                                    np.log10(meta[sid]['AREA_SQKM'])))
        plt.savefig(figname)
        plt.close('all')
        print(figname)


def plot_bf_linear_trend(daily_q_dir, fig_dir, metadata):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        try:
            sid = os.path.basename(c).split('.')[0]
            df = hydrograph(c)
            irr = df['irr'].mean()
            if irr > 0.01:
                color = 'b'
            else:
                color = 'r'
            _def = (df['etr'] / df['pr']).values
            qb = df['qb'].values
            idx = [x.year for x in list(df.index)]
            line = [x for x in range(df.shape[0])]
            mq, b = np.polyfit(line, qb, 1)
            fit = mq * np.array(line) + b
            plt.scatter(idx, qb, edgecolors=color, marker='o', alpha=0.6, facecolors='none')
            plt.plot(idx, fit)
            figname = os.path.join(fig_dir, '{}.png'.format(sid))
            plt.title('{} ({:.2f} irrigated)\n ({}): 10^{:.2f} sq km'.format(meta[sid]['STANAME'], irr,
                                                                             sid, np.log10(meta[sid]['AREA_SQKM'])))
            plt.savefig(figname)
            plt.close('all')

        except KeyError:
            print('{} has no qb'.format(sid))


def scatter_bfi_cc(metadata, csv_dir, fig):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir)]
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    irr_area, basin_area, ratio = [], [], []
    bfi = []
    exclude = []
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)
        m = metadata[sid]
        mean_area = df['irr'].mean()
        basin_sqm = m['AREA_SQKM'] * 1e6
        irr_area.append(mean_area)

        try:
            bfi_max = m['bfi_']
            rat = mean_area / basin_sqm
            if bfi_max < 0:
                exclude.append(sid)
                continue

            bfi.append(bfi_max)
            basin_area.append(basin_sqm)
            ratio.append(rat)
        except KeyError:
            exclude.append(sid)
            print('no bfi {} {}'.format(sid, m['STANAME']))

    plt.scatter(bfi, ratio)
    plt.savefig(fig)
    print(exclude)


def scatter_qb_cc(metadata, csv_dir, fig_d):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if 'lock' not in x]
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for c in l:
        if '13302500' not in c:
            pass
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)

        try:

            m = metadata[sid]
            _def = (df['etr'] / df['pr']).values
            irr = df['irr'].mean()
            qb = df['qb'].values
            cc = df['cc'].values
            m, b = np.polyfit(_def, qb, 1)
            fit = m * _def + b
            res = qb - fit

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.scatter(_def, qb)
            ax1.plot(_def, fit)
            ax1.set(xlabel='ETr / PPT')
            ax1.set(ylabel='qb')
            plt.title('{} {:.2f}'.format(sid, irr))
            ax2.set(ylabel='qb epsilon')
            ax2.set(xlabel='ETr / PPT')
            ax2.scatter(_def, res)
            fig_name = os.path.join(fig_d, '{}.png'.format(sid))
            # plt.show()
            # exit()
            plt.savefig(fig_name)
            plt.close('all')
            print(sid)
        except KeyError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass


def parameter_shift(jsn):
    with open(jsn, 'r') as f:
        meta = json.load(f)

    x_e, x_l = [], []
    y_e, y_l = [], []
    delt = []
    params = ['cc', 'qb', 'qb']
    for k, v in meta.items():
        try:
            if v['irr_late'] > 0.01:
                x_e.append(v['{}_early'.format(params[0])])
                x_l.append(v['{}_late'.format(params[0])])
                y_e.append(v['{}_early'.format(params[1])])
                y_l.append(v['{}_late'.format(params[1])])
                delt.append(v['{}_late'.format(params[2])] - v['{}_early'.format(params[2])])
        except KeyError:
            pass
    lines = [[(_xe, _ye), (_xl, _yl)] for _xe, _ye, _xl, _yl in zip(x_e, y_e, x_l, y_l)]
    slope = [(x[1][1] - x[0][1]) / (x[1][0] - x[0][0]) for x in lines]
    slope_bin = [1 if x > 0. else 0 for x in slope]
    _min, _max = min(slope), max(slope)
    slope_scale = np.interp(slope, (_min, _max), (0, 1))
    cmap = cm.get_cmap('RdYlGn')
    irr_clr = [cmap(i) for i in slope_bin]
    lc = LineCollection(lines, colors=irr_clr)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.set_title('Current')
    plt.show()


def scatter_delta_qb_delta_deficit(metadata, csv_dir, fig_d):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if 'lock' not in x]
    i, ni = [], []

    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for c in l:
        if '13302500' not in c:
            pass
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)
        try:
            irr = df['irr'].mean()
            _def = (df['etr'] / df['pr']).values
            qb = df['qb'].values
            idx = df.index.values
            mq, b = np.polyfit(idx, qb, 1)
            md, b = np.polyfit(idx, _def, 1)

            if irr > 0.01:
                i.append((mq, md, irr))
            else:
                ni.append((mq, md, irr))

        except KeyError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass

    iqb, idef, irr = [x[0] for x in i], [x[1] for x in i], [x[2] for x in i]
    niqb, nidef, nirr = [x[0] for x in ni], [x[1] for x in ni], [x[2] for x in ni]
    plt.scatter(idef, iqb, edgecolors='b', marker='o', alpha=0.6, facecolors='none')
    plt.scatter(nidef, niqb, edgecolors='r', marker='o', alpha=0.6, facecolors='none')
    plt.xlabel('ETr/PPT')
    plt.ylabel('delta qb')
    plt.show()


def filter_by_significance(metadata, csv_dir, fig_d, out_jsn):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if 'lock' not in x]
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for c in l:
        ct_tot += 1
        sid = os.path.basename(c).split('.')[0]
        df = hydrograph(c)
        try:
            years = [x for x in range(1991, 2021)]
            irr = df['irr'].values
            ind_var = df['irr'].values
            irr_area = irr.mean()
            if irr_area > 0.01:
                try:
                    _def = (df['etr'] / df['ppt']).values
                except KeyError:
                    _def = (df['etr'] / df['pr']).values

                _def_c = sm.add_constant(_def)
                dep_var = df['q'].values
                try:
                    ols = sm.OLS(dep_var, _def_c)
                except sm_exceptions.MissingDataError:
                    pass
                fit_clim = ols.fit()
                clim_line = fit_clim.params[1] * _def + fit_clim.params[0]
                irr_ct += 1
                if fit_clim.pvalues[1] < 0.05:
                    resid = fit_clim.resid
                    _irr_c = sm.add_constant(ind_var)
                    ols = sm.OLS(resid, _irr_c)
                    fit_resid = ols.fit()
                    if fit_resid.pvalues[1] < 0.05:
                        resid_line = fit_resid.params[1] * ind_var + fit_resid.params[0]
                        sig_stations[sid] = fit_resid.params[1]
                        desc_str = '{} {}\np = {:.3f}, irr = {:.3f}, m = {:.2f}'.format(sid,
                                                                                        metadata[sid]['STANAME'],
                                                                                        fit_resid.pvalues[1],
                                                                                        irr_area, fit_resid.params[1])
                        print(desc_str)
                        sig_stations[sid] = {'STANAME': metadata[sid]['STANAME'],
                                             'SIG': fit_resid.pvalues[1],
                                             'IRR_AREA': irr_area,
                                             'SLOPE': fit_resid.params[1]}

                        rcParams['figure.figsize'] = 16, 10
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        ax1.scatter(_def, dep_var)
                        ax1.plot(_def, clim_line)
                        ax1.set(xlabel='ETr / PPT')
                        ax1.set(ylabel='qb')
                        for i, y in enumerate(years):
                            ax1.annotate(y, (_def[i], dep_var[i]))
                        plt.suptitle(desc_str)
                        ax2.set(xlabel='cc')
                        ax2.set(ylabel='qb epsilon')
                        ax2.scatter(ind_var, resid)
                        ax2.plot(ind_var, resid_line)
                        for i, y in enumerate(years):
                            ax2.annotate(y, (ind_var[i], resid[i]))
                        fig_name = os.path.join(fig_d, '{}.png'.format(sid))
                        plt.savefig(fig_name)
                        plt.close('all')
                        irr_sig_ct += 1

                        ct += 1

        except KeyError as e:
            print('{} has no {}'.format(sid, e.args[0]))
            pass

    if out_jsn:
        with open(out_jsn, 'w') as f:
            json.dump(sig_stations, f)

    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct, ct_tot))
    print(sig_stations)


def two_variable_relationship(dep_var, ind_var, metadata, csv_dir, fig_d):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if 'lock' not in x]
    ct, irr_ct, irr_sig_ct, ct_tot = 0, 0, 0, 0
    sig_stations = {}
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    for c in l:
        ct_tot += 1
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)
        try:
            irr = df['irr'].values
            irr_area = irr.mean()
            x = df[ind_var].values
            x_c = sm.add_constant(x)
            y = df[dep_var].values
            ols = sm.OLS(y, x_c)
            fit_clim = ols.fit()
            clim_line = fit_clim.params[1] * x + fit_clim.params[0]
            if fit_clim.pvalues[1] < 0.05:
                if irr_area > 0.005:
                    desc_str = '{} {}\np = {:.3f}, irr = {:.3f}, m = {:.2f}'.format(sid,
                                                                                    metadata[sid]['STANAME'],
                                                                                    fit_clim.pvalues[1],
                                                                                    irr_area, fit_clim.params[1])
                    print(desc_str)

                    plt.scatter(x, y)
                    plt.plot(x, clim_line)
                    plt.xlabel(ind_var)
                    plt.ylabel(dep_var)
                    plt.suptitle(desc_str)
                    fig_name = os.path.join(fig_d, '{}.png'.format(sid))
                    plt.savefig(fig_name)
                    plt.close('all')
                    irr_sig_ct += 1

            ct += 1

        except KeyError as e:
            # print('{} has no {}'.format(sid, e.args[0]))
            pass

    print('{} climate-sig, {} irrigated, {} irr imapacted, {} total'.format(ct, irr_ct, irr_sig_ct, ct_tot))
    print(sig_stations)


def climate_vs_irrigation(csv_dir, metadata, fig_d):
    l = [os.path.join(csv_dir, x) for x in os.listdir(csv_dir) if 'lock' not in x]
    irr_impact, non_impact = 0, 0
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    for c in l:
        sid = os.path.basename(c).split('.')[0]
        df = read_csv(c)
        irr = df['irr'].values
        irr_area = irr.mean()
        if irr_area < 0.001:
            continue
        _def = (df['etr'] / df['pr']).values
        _def_c = sm.add_constant(_def)
        ols = sm.OLS(irr, _def_c)
        fit_clim = ols.fit()
        clim_line = fit_clim.params[1] * _def + fit_clim.params[0]

        desc_str = '{} {}\np = {:.3f}, irr = {:.3f}, m = {:.2f}\n'.format(sid,
                                                                          metadata[sid]['STANAME'],
                                                                          fit_clim.pvalues[1],
                                                                          irr_area, fit_clim.params[1])
        if fit_clim.pvalues[1] < 0.05:
            print(desc_str)
            plt.scatter(_def, irr)
            plt.plot(_def, clim_line)
            plt.xlabel('ETr / PPT')
            plt.ylabel('Irrigation')
            plt.suptitle(desc_str)
            fig_name = os.path.join(fig_d, '{}.png'.format(sid))
            plt.savefig(fig_name)
            plt.close('all')
            irr_impact += 1
        else:
            non_impact += 1

    print(irr_impact, non_impact)


def daily_temperature_plot(df, sid, fig_d):
    df.plot()
    plt.xlabel('time')
    plt.ylabel('temp')
    plt.suptitle(sid)
    fig_name = os.path.join(fig_d, '{}.png'.format(sid))
    plt.savefig(fig_name)
    plt.close('all')
    return None


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    # c = '/media/research/IrrigationGIS/gages/hydrographs/group_stations/stations_annual.csv'
    # fig = '/media/research/IrrigationGIS/gages/figures/bfi_vs_irr.png'
    # src = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'

    data = '/media/research/IrrigationGIS/gages/merged_q_ee/JAS_Comp_4AUG2021'
    jsn = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows_gridded_JAS_4AUG2021.json'
    fig_dir = '/media/research/IrrigationGIS/gages/figures/sig_irr_qb_wy_comp_scatter'
    filter_by_significance(jsn, data, fig_dir, out_jsn=None)
    # two_variable_relationship('qb', 'irr', jsn, data, fig_dir)
# ========================= EOF ====================================================================
