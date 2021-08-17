import os
import json
from copy import copy
from collections import OrderedDict
from itertools import permutations, product

from scipy.stats.stats import pearsonr
import numpy as np
from pandas import DataFrame, read_csv
from datetime import datetime, date
from dateutil.rrule import rrule, DAILY
from dateutil.relativedelta import relativedelta as rdlt
import fiona
import statsmodels.tools.sm_exceptions as sm_exceptions
import statsmodels.api as sm

from hydrograph import hydrograph

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

CLMB_STATIONS = ['12302055',
                 '12323600',
                 '12323770',
                 '12324200',
                 '12324590',
                 '12324680',
                 '12325500',
                 '12329500',
                 '12330000',
                 '12331500',
                 '12332000',
                 '12334510',
                 '12334550',
                 '12335500',
                 '12340000',
                 '12340500',
                 '12342500',
                 '12344000',
                 '12350250',
                 '12352500',
                 '12353000',
                 '12354000',
                 '12354500',
                 '12358500',
                 '12359800',
                 '12362500',
                 '12363000',
                 '12366000',
                 '12370000',
                 '12371550',
                 '12372000',
                 '12374250',
                 '12375900',
                 '12377150',
                 '12388700',
                 '12389000',
                 '12389500',
                 '12390700',
                 '12392155',
                 '12392300',
                 '12395000',
                 '12395500',
                 '12396500',
                 '12398600',
                 '12409000',
                 '12411000',
                 '12413000',
                 '12413210',
                 '12413470',
                 '12413500',
                 '12414500',
                 '12414900',
                 '12415500',
                 '12419000',
                 '12422000',
                 '12422500',
                 '12424000',
                 '12426000',
                 '12433000',
                 '12433200',
                 '12445900',
                 '12447390',
                 '12448500',
                 '12448998',
                 '12449500',
                 '12449950',
                 '12451000',
                 '12452000',
                 '12452500',
                 '12452800',
                 '12456500',
                 '12457000',
                 '12458000',
                 '12459000',
                 '12462500',
                 '12464800',
                 '12465000',
                 '12465400',
                 '12467000',
                 '12472600',
                 '12484500',
                 '12488500',
                 '12500450',
                 '12502500',
                 '12506000',
                 '12508990',
                 '12510500',
                 '12513000',
                 '13010065',
                 '13011000',
                 '13011500',
                 '13011900',
                 '13014500',
                 '13015000',
                 '13018300',
                 '13018350',
                 '13018750',
                 '13022500',
                 '13023000',
                 '13025500',
                 '13027500',
                 '13032500',
                 '13037500',
                 '13038500',
                 '13039000',
                 '13039500',
                 '13042500',
                 '13046000',
                 '13047500',
                 '13049500',
                 '13050500',
                 '13052200',
                 '13055000',
                 '13055340',
                 '13056500',
                 '13057000',
                 '13057500',
                 '13057940',
                 '13058000',
                 '13062500',
                 '13063000',
                 '13066000',
                 '13069500',
                 '13073000',
                 '13075000',
                 '13075500',
                 '13075910',
                 '13077000',
                 '13078000',
                 '13081500',
                 '13082500',
                 '13083000',
                 '13090000',
                 '13090500',
                 '13094000',
                 '13105000',
                 '13106500',
                 '13108150',
                 '13112000',
                 '13115000',
                 '13116500',
                 '13119000',
                 '13120000',
                 '13120500',
                 '13126000',
                 '13127000',
                 '13132500',
                 '13135000',
                 '13135500',
                 '13137000',
                 '13137500',
                 '13141500',
                 '13142000',
                 '13142500',
                 '13147900',
                 '13148200',
                 '13148500',
                 '13152500',
                 '13153500',
                 '13159800',
                 '13161500',
                 '13168500',
                 '13171500',
                 '13171620',
                 '13172500',
                 '13174000',
                 '13174500',
                 '13181000',
                 '13183000',
                 '13185000',
                 '13186000',
                 '13190500',
                 '13200000',
                 '13201500',
                 '13206000',
                 '13213000',
                 '13213100',
                 '13233300',
                 '13235000',
                 '13236500',
                 '13238500',
                 '13239000',
                 '13240000',
                 '13245000',
                 '13246000',
                 '13247500',
                 '13249500',
                 '13250000',
                 '13251000',
                 '13258500',
                 '13265500',
                 '13266000',
                 '13269000',
                 '13295000',
                 '13296000',
                 '13296500',
                 '13297330',
                 '13297350',
                 '13297355',
                 '13302500',
                 '13305000',
                 '13307000',
                 '13309220',
                 '13310700',
                 '13311000',
                 '13313000',
                 '13316500',
                 '13317000',
                 '13331500',
                 '13333000',
                 '13336500',
                 '13337000',
                 '13337500',
                 '13338500',
                 '13339500',
                 '13340000',
                 '13340600',
                 '13340950',
                 '13341050',
                 '13342450',
                 '13342500',
                 '13344500',
                 '13345000',
                 '13346800',
                 '13348000',
                 '13351000',
                 '14013000',
                 '14015000',
                 '14018500',
                 '14020000',
                 '14020300',
                 '14033500',
                 '14034470',
                 '14034480',
                 '14034500',
                 '14038530',
                 '14044000',
                 '14046000',
                 '14046500',
                 '14048000',
                 '14076500',
                 '14087400',
                 '14091500',
                 '14092500',
                 '14092750',
                 '14093000',
                 '14096850',
                 '14097100',
                 '14101500',
                 '14103000',
                 '14107000',
                 '14113000',
                 '14113200',
                 '14120000',
                 '14123500',
                 '14137000',
                 '14138800',
                 '14138850',
                 '14138870',
                 '14138900',
                 '14139800',
                 '14140000',
                 '14141500',
                 '14142500',
                 '14144800',
                 '14144900',
                 '14145500',
                 '14147500',
                 '14148000',
                 '14149000',
                 '14150000',
                 '14150800',
                 '14150900',
                 '14151000',
                 '14152000',
                 '14152500',
                 '14153000',
                 '14153500',
                 '14154500',
                 '14155000',
                 '14155500',
                 '14157500',
                 '14158500',
                 '14158790',
                 '14158850',
                 '14159110',
                 '14159200',
                 '14159500',
                 '14161100',
                 '14161500',
                 '14162100',
                 '14162200',
                 '14162500',
                 '14163000',
                 '14163150',
                 '14163900',
                 '14165000',
                 '14165500',
                 '14166000',
                 '14166500',
                 '14168000',
                 '14169000',
                 '14170000',
                 '14171000',
                 '14174000',
                 '14178000',
                 '14179000',
                 '14180500',
                 '14181500',
                 '14182500',
                 '14183000',
                 '14184100',
                 '14185000',
                 '14185800',
                 '14185900',
                 '14186100',
                 '14186600',
                 '14187000',
                 '14187200',
                 '14187500',
                 '14188800',
                 '14189000',
                 '14190500',
                 '14191000',
                 '14200000',
                 '14200300',
                 '14201500',
                 '14202000',
                 '14202980',
                 '14203500',
                 '14206900',
                 '14207500',
                 '14207740',
                 '14207770',
                 '14208700',
                 '14209000',
                 '14209500',
                 '14210000',
                 '14211500',
                 '14211550',
                 '14211720',
                 '14216000',
                 '14216500',
                 '14219000',
                 '14219800',
                 '14220500',
                 '14222500',
                 '14226500',
                 '14231000',
                 '14233500',
                 '14234800',
                 '14236200',
                 '14238000',
                 '14242580',
                 '14243000']


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
    significance, coefficients, irr_pct = {}, {}, {}
    for c in l:

        df = hydrograph(c).loc['1997-01-01': '2017-12-31']
        irr = df['irr'].mean()
        if irr < 0.01:
            continue

        sid = os.path.basename(c).split('.')[0]

        # if sid not in CLMB_STATIONS:
        #     continue

        significance[sid] = {}
        coefficients[sid] = {}
        irr_pct[sid] = irr

        try:
            response = aic_glm(c)
        except RRuntimeError as e:
            print(sid, e)
            continue

        for v in ['irr', 'ppt', 'etr', 'ppt_lt', 'etr_lt']:
            try:
                significance[sid][v] = response.loc[v]['Pr(>|t|)']
                coefficients[sid][v] = response.loc[v]['Estimate']
            except KeyError:
                significance[sid][v] = 9999
                coefficients[sid][v] = 9999

    with fiona.open(in_shp, 'r') as src:
        shape_meta = src.meta
        features = [f for f in src if f['properties']['STAID'] in significance.keys()]

    shape_meta['schema']['properties']['irr_pct'] = 'float:19.11'

    # shape_meta['schema']['properties']['cc_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['irr_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['ppt_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['ppt_lt_coef'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_lt_coef'] = 'float:19.11'

    # shape_meta['schema']['properties']['cc_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['irr_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['ppt_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['ppt_lt_sig'] = 'float:19.11'
    shape_meta['schema']['properties']['etr_lt_sig'] = 'float:19.11'

    ct = 0
    with fiona.open(out_shp, 'w', **shape_meta) as dst:
        for f in features:
            try:
                # cc_sig = significance[f['properties']['STAID']]['cc']
                i_sig = significance[f['properties']['STAID']]['irr']
                if i_sig > 0.05:
                    continue
                feature = {'geometry': f['geometry'],
                           'id': ct,
                           'properties': OrderedDict([('STAID', f['properties']['STAID']),
                                                      ('STANAME', f['properties']['STANAME']),
                                                      ('SQMI', f['properties']['SQMI']),

                                                      # ('cc_coef', coefficients[f['properties']['STAID']]['cc']),
                                                      ('irr_coef', coefficients[f['properties']['STAID']]['irr']),
                                                      ('ppt_coef', coefficients[f['properties']['STAID']]['ppt']),
                                                      ('etr_coef', coefficients[f['properties']['STAID']]['etr']),
                                                      ('ppt_lt_coef', coefficients[f['properties']['STAID']]['ppt_lt']),
                                                      ('etr_lt_coef', coefficients[f['properties']['STAID']]['etr_lt']),

                                                      # ('cc_sig', significance[f['properties']['STAID']]['cc']),
                                                      ('irr_sig', significance[f['properties']['STAID']]['irr']),
                                                      ('ppt_sig', significance[f['properties']['STAID']]['ppt']),
                                                      ('etr_sig', significance[f['properties']['STAID']]['etr']),
                                                      ('ppt_lt_sig', significance[f['properties']['STAID']]['ppt_lt']),
                                                      ('etr_lt_sig', significance[f['properties']['STAID']]['etr_lt']),

                                                      ('irr_pct', irr_pct[f['properties']['STAID']]),

                                                      ('start', f['properties']['start']),
                                                      ('end', f['properties']['end'])]),
                           'type': 'Feature'}
            except KeyError:
                print(f['properties']['STAID'], ' no irr')
                continue
            try:
                dst.write(feature)
                print(f['properties']['STAID'])
                ct += 1
            except TypeError:
                pass
    print('wrote {} watersheds'.format(ct))


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


def cross_corr(x, y, lag):
    return x.corr(y.shift(lag))


def baseflow_correlation_search(climate_dir, station_json):
    l = [os.path.join(climate_dir, x) for x in os.listdir(climate_dir)]
    offsets = [x for x in range(1, 60)]
    windows = {}
    for csv in l:
        station = os.path.basename(csv).strip('.csv')
        print('\n{}'.format(station))
        df = hydrograph(csv)
        df['ai'] = df['ppt'] - df['etr']
        try:
            mean_m_qb = [df[df.index.month == m].dropna()['qb'].mean() for m in range(1, 13)]
        except KeyError:
            continue
        lf = mean_m_qb.index(min(mean_m_qb))
        qb, ai = df[df.index.month.isin([lf])]['qb'].dropna(), df['ai']
        years = [x.year for x in qb.index]
        corr = (0, 0, 0.0)
        for span in offsets:
            for lag in offsets:
                if span > lag:
                    continue
                dates = [(date(y, lf, 1) + rdlt(months=-lag), date(y, lf, 1) + rdlt(months=-(lag - span))) for y in
                         years]
                ind = [ai[d[0]: d[1]].sum() for d in dates]
                rpearson = pearsonr(ind, qb)
                c, p = abs(rpearson[0]), rpearson[1]
                if c > corr[2]:
                    corr = (lag, span, c)
                    d = {'lag': lag, 'span': span, 'pearsonr': c, 'pval': p}
                    print('high', lag, span, c, p)
        windows[station] = d
        print(windows)

    with open(station_json, 'w') as f:
        json.dump(windows, f)


if __name__ == '__main__':
    clim_dir = '/media/research/IrrigationGIS/gages/merged_q_ee/q_terraclim'
    _json = '/media/research/IrrigationGIS/gages/station_metadata/climate_sensitivity_metadata.json'
    baseflow_correlation_search(clim_dir, _json)
# ========================= EOF ====================================================================
