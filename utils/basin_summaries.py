import os
import json
from datetime import date
import warnings

from scipy.stats.stats import linregress
import matplotlib.pyplot as plt
import fiona
import numpy as np

from gage_data import hydrograph

warnings.filterwarnings('ignore')

from gage_lists import EXCLUDE_STATIONS


def get_water_year_cc_q_ratio(climate_dir, in_json, out_json):
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    reff_dct = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        sid = os.path.basename(csv).strip('.csv')

        s_meta = metadata[sid]

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_frac = mean_irr * (1. / 1e6) / s_meta['AREA']
        if irr_frac < 0.001:
            continue

        years = np.arange(1991, 2021)

        dates = [(date(y - 1, 10, 1), date(y, 9, 30)) for y in years]
        q = np.array([df['q'][d[0]: d[1]].sum() for d in dates])
        cc = np.array([df['cc'][d[0]: d[1]].sum() for d in dates])
        ratio = cc / q
        lr = linregress(years, ratio)
        b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue

        reff_dct[sid] = {'b': b,
                         'r': r,
                         'p': p,
                         'irr_frac': irr_frac,
                         'q_data': list(q),
                         'cc_data': list(cc),
                         'q_cc_ratio_data': list(ratio)}

    with open(out_json, 'w') as f:
        json.dump(reff_dct, f, indent=4)


def water_balance_ratios(metadata, ee_series, stations=None, metadata_out=None, out_shp=None):
    with open(metadata, 'r') as f:
        metadata = json.load(f)

    dct = {}
    for sid, v in metadata.items():
        if sid in EXCLUDE_STATIONS:
            continue
        _file = os.path.join(ee_series, '{}.csv'.format(sid))
        if not os.path.exists(_file):
            continue
        cdf = hydrograph(_file)
        cdf['cci'] = cdf['cc'] / cdf['irr']
        years = np.arange(1991, 2021)
        cc_dates = [(date(y, 4, 1), date(y, 10, 31)) for y in years]
        clim_dates = [(date(y, 1, 1), date(y, 12, 31)) for y in years]
        q = np.array([cdf['q'][d[0]: d[1]].sum() for d in clim_dates])
        ppt = np.array([cdf['ppt'][d[0]: d[1]].sum() for d in clim_dates])
        etr = np.array([cdf['etr'][d[0]: d[1]].sum() for d in clim_dates])
        cc = np.array([cdf['cc'][d[0]: d[1]].sum() for d in cc_dates])
        irr = np.array([cdf['irr'][d[0]: d[1]].mean() for d in cc_dates])
        cci = np.array([cdf['cci'][d[0]: d[1]].sum() for d in cc_dates])
        # if not np.all(irr > 0.0):
        #     continue
        print('cci: {:.3f}, {}'.format(np.mean(cci), v['STANAME']))

        dct[sid] = v
        irr_area = (np.mean(irr)).item() / 1e6
        irr_frac = irr_area / v['AREA']
        if irr_area < 0.001:
            irr_area = 0.0

        cc_q_ratio = (cc.sum() / q.sum()).item()
        if cc_q_ratio < 0.0:
            cc_q_ratio = 0.0

        dct[sid].update({'IAREA': irr_area})
        dct[sid].update({'irr_frac': irr_frac})
        dct[sid].update({'cc_q': cc_q_ratio})
        dct[sid].update({'cci': (np.mean(cci)).item()})
        dct[sid].update({'q_ppt': (q.sum() / ppt.sum()).item()})
        dct[sid].update({'ai': (etr.sum() / ppt.sum()).item()})

    if metadata_out:
        with open(metadata_out, 'w') as fp:
            json.dump(dct, fp, indent=4, sort_keys=False)
    if stations:
        with fiona.open(stations, 'r') as src:
            features = [f for f in src]
            meta = src.meta

        del meta['schema']['properties']['LAT']
        del meta['schema']['properties']['LON']

        [meta['schema']['properties'].update({k: 'float:19.11'}) for k, v in dct[sid].items()
         if k not in meta['schema']['properties'].keys() and isinstance(v, float)]

        areas = {f['properties']['STAID']: f['properties']['AREA'] for f in features}
        area_arr = np.array([areas[_id] for _id in dct.keys()])
        areas = {k: (a - min(area_arr)) / (max(area_arr) - min(area_arr)) for k, a in areas.items()}

        with fiona.open(out_shp, 'w', **meta) as dst:
            for f in features:
                sid = f['properties']['STAID']
                if sid in dct.keys():
                    d = {k: v for k, v in dct[sid].items() if isinstance(v, str)}
                    d.update({k: v for k, v in dct[sid].items() if isinstance(v, float)})
                    d['STAID'] = sid
                    d['AREA'] = areas[sid]
                    f['properties'] = d
                    dst.write(f)


def get_water_year_runoff_efficiency(climate_dir, in_json, out_json):
    l = sorted([os.path.join(climate_dir, x) for x in os.listdir(climate_dir)])
    reff_dct = {}
    with open(in_json, 'r') as f:
        metadata = json.load(f)
    for csv in l:
        sid = os.path.basename(csv).strip('.csv')

        s_meta = metadata[sid]

        df = hydrograph(csv)
        mean_irr = np.nanmean(df['irr'].values)
        irr_frac = mean_irr * (1. / 1e6) / s_meta['AREA']

        years = np.arange(1991, 2021)

        dates = [(date(y - 1, 10, 1), date(y, 9, 30)) for y in years]
        q = np.array([df['q'][d[0]: d[1]].sum() for d in dates])
        ppt = np.array([df['gm_ppt'][d[0]: d[1]].sum() for d in dates])
        r_eff = q / ppt
        lr = linregress(years, r_eff)
        b, inter, r, p = lr.slope, lr.intercept, lr.rvalue, lr.pvalue

        reff_dct[sid] = {'inter': inter,
                         'b': b,
                         'r': r,
                         'p': p,
                         'irr_frac': irr_frac,
                         'q_data': list(q),
                         'ppt_data': list(ppt),
                         'reff_data': list(r_eff)}

    with open(out_json, 'w') as f:
        json.dump(reff_dct, f, indent=4)


def summarize_climate_q(_dir):
    _files = [os.path.join(_dir, 'climate_q_{}.json'.format(m)) for m in range(1, 13)]
    lags, slopes, r_vals = [], [], []
    for m, f in enumerate(_files, start=1):
        m_lags, m_slopes, areas = [], [], []
        insig, sig = 0, 0
        pos, neg = 0, 0
        with open(f, 'r') as fp:
            d_obj = json.load(fp)
        for k, v in d_obj.items():
            key_ = '{}-{}'.format(m, m)
            if key_ not in v.keys():
                insig += 1
                continue
            lag, b, r = v[key_]['lag'], v[key_]['b'], v[key_]['r']
            if b > 0.0:
                pos += 1
            else:
                neg += 1
            sig += 1
            lags.append(lag)
            m_lags.append(lag)
            slopes.append(b)
            m_slopes.append(b)
            if m == 7:
                r_vals.append(r)
                areas.append(v['AREA'])

        if m == 7:
            plt.scatter(np.log10(areas), np.sqrt(r_vals))
            plt.show()
            plt.close()
        print('month {}, mean lag {:.3f}, mean slope {:.3f}, {} insig'.format(m, np.array(m_lags).mean(),
                                                                              np.array(m_slopes).mean(), insig))
    print('mean lag {:.3f}, mean slope {:.3f}'.format(np.array(lags).mean(),
                                                      np.array(slopes).mean()))


def summarize_cc_qres(_dir, out_json, glob=None):
    dct = {}
    _files = [os.path.join(_dir, '{}_{}.json'.format(glob, m)) for m in range(1, 12)]

    for m, f in enumerate(_files, start=1):
        insig, sig = 0, 0
        with open(f, 'r') as fp:
            d_obj = json.load(fp)

        diter = [[(kk, k, r) for k, r in vv.items() if isinstance(r, dict)] for kk, vv in d_obj.items()]
        diter = [i for ll in diter for i in ll]
        for k, cc, d in diter:
            if d['p'] > 0.05:
                insig += 1
                continue
            sig += 1
            if k not in dct.keys():
                dct[k] = {m: {cc: d['b']}}
            elif m not in dct[k].keys():
                dct[k][m] = {cc: d['b']}
            else:
                dct[k][m].update({cc: d['b']})

    with open(out_json, 'w') as fp:
        json.dump(dct, fp, indent=4)


def summarize_trends(_dir):
    _files = [os.path.join(_dir, 'trend_data_{}.json'.format(m)) for m in range(1, 13)]

    for var in ['time_cc', 'time_qres', 'time_ai', 'time_aim', 'time_q', 'time_etr', 'time_ppt', 'time_irr']:
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

    pass


if __name__ == '__main__':
    root = '/media/research/IrrigationGIS/gages'
    if not os.path.exists(root):
        root = '/home/dgketchum/data/IrrigationGIS/gages'

    ee_data = os.path.join(root, 'merged_q_ee/monthly_ssebop_tc_gm_q_Comp_21DEC2021_unfiltered_q')
    i_json = os.path.join(root, 'gridmet_analysis', 'station_metadata.json')
    station_pts = os.path.join(root, 'gage_loc_usgs', 'selected_gages.shp')
    shp = os.path.join(root, 'gridmet_analysis', 'fig_shapes', 'basin_cc_ratios_annual_q.shp')

    water_balance_ratios(i_json, ee_data, stations=station_pts, out_shp=shp)
# ========================= EOF ====================================================================
