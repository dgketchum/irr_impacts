import json
import os
from datetime import date

import fiona
import numpy as np
import pandas as pd

from utils.gage_lists import INCLUDE_STATIONS


def summarize_climate_flow(_dir, out_csv):
    climate_flow_file = os.path.join(climate_flow_data, 'climate_flow_{}.json')
    first = True
    rsqa = {}
    stations = []
    for m in range(1, 13):
        slopes, rsqm = [], {}
        month_name = date(2000, m, 1).strftime('%B')
        out_data = climate_flow_file.format(m)
        with open(out_data) as fp:
            dct = json.load(fp)
        if first:
            df = pd.DataFrame(index=[k for k in dct.keys()], columns=[month_name, 'Station Name'])
            rdf = pd.DataFrame(index=[k for k in dct.keys()], columns=[m])
            first = False
        for k, v in dct.items():
            stations.append(k)
            record_str = '{}, {}, {:.3f}, {:.3f}, {:.3f}'.format(len(v['q']), v['lag'], v['r'], v['p'], v['b'])
            if k not in rsqm.keys():
                rsqm[k] = v['r']
            if k not in rsqa.keys():
                rsqa[k] = v['r']
            if v['r'] > rsqa[k]:
                rsqa[k] = v['r']

            slopes.append(v['b'])
            df.loc[k, month_name] = record_str
            df.loc[k, 'Station Name'] = v['STANAME']
            rdf.loc[k, m] = v['r']

        r = np.array([v for k, v in rsqm.items()])
        p75, p25 = np.nanpercentile(r, [75, 25])
        median = np.median(r)
        print('{} median {:.2f} iqr: {:.2f}'.format(month_name, median, p75 - p25))

    df.sort_index(inplace=True)
    df.to_csv(out_csv)


def basin_summaries(in_data, summary, shp, out_csv):
    summary = pd.read_csv(summary, index_col=0)
    analyzed = [str(i).rjust(8, '0') for i in list(summary.index)]
    with fiona.open(shp, 'r') as src:
        feats = [f for f in src]

    geo_ = {f['properties']['STAID']: f['geometry']['coordinates'] for f in feats}

    with open(in_data) as fp:
        dct = json.load(fp)
    df = pd.DataFrame(columns=['USGS Station ID', 'Station Name', 'Area', 'Irr Area',
                               'Irr Fraction', 'Lat', 'Lon'])
    for k, v in dct.items():
        if k not in analyzed:
            continue
        g = geo_[k]
        df.loc[k] = {'USGS Station ID': k, 'Station Name': v['STANAME'], 'Area': v['AREA'],
                     'Irr Area': v['irr_area'],
                     'Irr Fraction': v['irr_frac'], 'Lat': g[1], 'Lon': g[0]}
    df.to_csv(out_csv)


if __name__ == '__main__':
    root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
    if not os.path.exists(root):
        root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')
    analysis_directory = os.path.join(root, 'analysis')
    climate_flow_data = os.path.join(analysis_directory, 'climate_flow')
    out_ = os.path.join(root, 'gages', 'climate_flow_summary.csv')
    summarize_climate_flow(climate_flow_data, out_)

    summary = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
    shape = os.path.join('/media/research/IrrigationGIS/impacts/gages/selected_gages.shp')
    out_summary = os.path.join(root, 'gages', 'irrigated_gage_metadata.csv')
    basin_summaries(summary, out_, shape, out_summary)
# ========================= EOF ====================================================================
