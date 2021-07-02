import os
import json
from copy import copy
from collections import OrderedDict
import numpy as np
from pandas import to_datetime, read_csv, DataFrame
from datetime import datetime
from dateutil.rrule import rrule, DAILY
import fiona

os.environ['R_HOME'] = '/home/dgketchum/miniconda3/envs/renv/lib/R'
import rpy2.robjects as ro
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter
from rpy2.rinterface_lib.embedded import RRuntimeError

EXCLUDE_STATIONS = ['05015500', '06154400', '06311000', '06329590', '06329610', '06329620',
                    '09125800', '09131495', '09147022', '09213700', '09362800', '09398300',
                    '09469000', '09509501', '09509502', '12371550', '12415500', '12452000',
                    '13039000', '13106500', '13115000', '13119000', '13126000', '13142000',
                    '13148200', '13171500', '13174000', '13201500', '13238500', '13340950',
                    '14149000', '14150900', '14153000', '14155000', '14162100', '14168000',
                    '14180500', '14186100', '14186600', '14207740', '14207770', '14234800']


def hydrograph(c):
    df = read_csv(c)
    try:
        df['dt'] = to_datetime(df['dt'])
    except KeyError:
        df['dt'] = to_datetime(df['datetimeUTC'])
    df = df.set_index('dt')
    try:
        df.drop(columns='datetimeUTC', inplace=True)
    except KeyError:
        pass
    return df


def peak_q_doy(daily_q_dir):
    l = [os.path.join(daily_q_dir, x) for x in os.listdir(daily_q_dir)]
    for c in l:
        df = hydrograph(c)
        s, e = df['datetimeUTC'].iloc[0], df['datetimeUTC'].iloc[-1]
        df['doy'] = [int(datetime.strftime(x, '%j')) for x in rrule(DAILY, dtstart=s, until=e)]
        peak_doy = []
        for y in range(s.year, e.year + 1):
            ydf = copy(df.loc['{}-01-01'.format(y): '{}-12-31'.format(y)])
            if ydf.shape[0] < 364:
                continue
            peak_doy.append(ydf['doy'].loc[ydf['q'].idxmax()])

        print(peak_doy)


def irrigation_fraction(dir_, metadata):
    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]
    irr_idx = []
    with open(metadata, 'r') as f:
        meta = json.load(f)
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        m = meta[sid]
        df = hydrograph(c)
        if np.all(df['irr'].values < 10e4):
            continue
            # print('no irrigation', m['STANAME'])
        df['if'] = df['cc'] / df['q']
        frac = df['cc'].sum() / df['q'].sum()
        if df['cc'].sum() < 0:
            print(sid, frac, m['STANAME'])
        irr_idx.append(frac)
        # print(sid, frac, meta[sid]['STANAME'])

    # [print(x) for x in irr_idx]


def get_response(_dir, in_shp, out_shp):
    pandas2ri.activate()
    r['source']('aic.R')
    aic_glm = robjects.globalenv['aic_glm']
    l = [os.path.join(_dir, x) for x in os.listdir(_dir)]
    significance, coefficients = {}, {}
    for c in l:
        sid = os.path.basename(c).split('.')[0]
        significance[sid] = {}
        coefficients[sid] = {}
        df = hydrograph(c)

        try:
            response = aic_glm(c)
        except RRuntimeError as e:
            print(sid, e)

        for v in ['cc', 'irr', 'pr', 'etr']:
            try:
                significance[sid][v] = response.loc[v]['Pr(>|t|)']
                coefficients[sid][v] = response.loc[v]['Estimate']
            except KeyError:
                significance[sid][v] = 9999
                coefficients[sid][v] = 9999

    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        features = [f for f in src if f['properties']['STAID'] in significance.keys()]

    shape_meta['schema']['properties']['cc_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['irr_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['pr_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_coef'] = 'float:19.11'

    shape_meta['schema']['properties']['cc_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['irr_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['pr_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_sig'] = 'float:19.11'
    ct = 0
    with fiona.open(out_shp, 'w', **shape_meta) as dst:
        for f in features:
            ct += 1
            feature = {'geometry': f['geometry'],
                       'id': ct,
                       'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                  ('STANAME', f['properties']['STANAME']),
                                                  ('SQMI', f['properties']['SQMI']),

                                                  ('cc_coef', coefficients[f['properties']['STAID']]['cc']),
                                                  ('irr_coef', coefficients[f['properties']['STAID']]['irr']),
                                                  ('pr_coef', coefficients[f['properties']['STAID']]['pr']),
                                                  ('etr_coef', coefficients[f['properties']['STAID']]['etr']),

                                                  ('cc_sig', significance[f['properties']['STAID']]['cc']),
                                                  ('irr_sig', significance[f['properties']['STAID']]['irr']),
                                                  ('pr_sig', significance[f['properties']['STAID']]['pr']),
                                                  ('etr_sig', significance[f['properties']['STAID']]['etr']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'])
            except TypeError:
                pass


def write_flow_parameters(dir_, metadata_in, out_dir, metadata_out):
    pandas2ri.activate()
    r['source']('BaseflowSeparationFunctions.R')
    rec_const_r = robjects.globalenv['baseflow_RecessionConstant']
    bfi_max_r = robjects.globalenv['baseflow_BFImax']
    bf_eckhardt_r = robjects.globalenv['baseflow_Eckhardt']

    l = [os.path.join(dir_, x) for x in os.listdir(dir_)]

    with open(metadata_in, 'r') as f:
        meta = json.load(f)

    for c in l:

        sid = os.path.basename(c).split('.')[0]
        if sid in EXCLUDE_STATIONS:
            print('exclude {}'.format(sid))
            continue

        if sid not in meta.keys():
            print(sid, 'not found')
            continue
        print(sid, meta[sid]['STANAME'])

        df = hydrograph(c)
        df.rename(columns={list(df.columns)[0]: 'q'}, inplace=True)

        keys = ('bfi_early', 'k_early'), ('bfi_late', 'k_late'), ('bfi_pr', 'k_pr')
        slices = [('1991-01-01', '2005-12-31'), ('2006-01-01', '2020-12-31'), ('1991-01-01', '2020-12-31')]

        for bk, s in zip(keys, slices):

            with localconverter(ro.default_converter + pandas2ri.converter):
                dfs = df['q'].loc[s[0]: s[1]]
                dfr = ro.conversion.py2rpy(dfs)
                dfs = DataFrame(dfs)

            try:
                k = rec_const_r(dfr)[0]
                bfi_max = bfi_max_r(dfr, k)[0]
                dfs['qb'] = bf_eckhardt_r(dfr, bfi_max, k)
                meta[sid].update({bk[0]: bfi_max, bk[1]: k})

            except RRuntimeError:
                print('error ', sid, meta[sid]['STANAME'])

            if 'pr' in bk[0]:
                dfs.to_csv(os.path.join(out_dir, '{}.csv'.format(sid)))

        try:
            meta[sid]['bfi_dlt'] = meta[sid]['bfi_late'] - meta[sid]['bfi_early']
        except KeyError:
            pass

    with open(metadata_out, 'w') as f:
        json.dump(meta, f)


def write_bfi_to_shapefile(in_shp, out_shp, meta):
    with open(meta, 'r') as f:
        meta = json.load(f)

    features = []
    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        for f in src:
            sid = f['properties']['STAID']
            if len(meta[sid].keys()) > 10:
                features.append(f)

    shape_meta['schema']['properties']['bfi_early'] = 'float:19.11'
    shape_meta['schema']['properties']['bfi_late'] = 'float:19.11'
    shape_meta['schema']['properties']['bfi_dlt'] = 'float:19.11'

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

                                                  ('bfi_early', meta[sid]['bfi_early']),
                                                  ('bfi_late', meta[sid]['bfi_late']),
                                                  ('bfi_dlt', meta[sid]['bfi_dlt']),

                                                  ('start', f['properties']['start']),
                                                  ('end', f['properties']['end'])]),
                       'type': 'Feature'}
            ct += 1
            try:
                dst.write(feature)
                print(f['properties']['STAID'])
            except TypeError:
                pass


if __name__ == '__main__':
    data = '/media/research/IrrigationGIS/gages/merged_q_ee'

    daily = '/media/research/IrrigationGIS/gages/hydrographs/daily_q'
    daily_bf = '/media/research/IrrigationGIS/gages/hydrographs/daily_q_bf'
    src = '/media/research/IrrigationGIS/gages/merged_q_ee/July_Oct'
    dst = '/media/research/IrrigationGIS/gages/responses'

    jsn = '/media/research/IrrigationGIS/gages/station_metadata/metadata.json'
    jsn_out = '/media/research/IrrigationGIS/gages/station_metadata/metadata_flows.json'

    watersheds = '/media/research/IrrigationGIS/gages/watersheds/selected_watersheds.shp'
    # watersheds_sig = '/media/research/IrrigationGIS/gages/watersheds/model_JulyOct.shp'
    watersheds_bfi = '/media/research/IrrigationGIS/gages/watersheds/watersheds_bfi.shp'
    # peak_q_dates(s, e, daily, peak_dir)
    # irrigation_fraction(data, jsn)
    write_flow_parameters(daily, jsn, daily_bf, jsn_out)
    # write_bfi_to_shapefile(watersheds, watersheds_bfi, jsn_out)
# ========================= EOF ====================================================================
