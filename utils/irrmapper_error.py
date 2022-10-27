import os
import json
from datetime import datetime

import ee
import numpy as np
import pandas as pd

BASIN_MAP = {'missouri': 'users/dgketchum/boundaries/umrb_ylstn_clip',
             'columbia': 'users/dgketchum/boundaries/CMB_RB_CLIP',
             'colorado': 'users/dgketchum/boundaries/CO_RB'}

BASIN_YEARS = {'missouri': [2008, 2009, 2010, 2011, 2012, 2013],
               'colorado': [1998, 2003, 2006, 2009, 2013, 2016],
               'columbia': [1988, 1996, 1997, 1998, 2001, 2006, 2008, 2009, 2010,
                            2011, 2012, 2013]}

BOUNDARIES = 'users/dgketchum/boundaries'
CO_STATES = ['WY', 'CO', 'UT', 'NV', 'NM', 'AZ']


def confusion(irr_labels, unirr_labels, irr_image, unirr_image, system, clip=False, state=None):
    fc = ee.FeatureCollection(BASIN_MAP[system])
    domain = fc.toList(fc.size()).get(0)
    domain = ee.Feature(domain)

    true_positive = irr_image.eq(irr_labels)
    false_positive = irr_image.eq(unirr_labels)
    true_negative = unirr_image.eq(unirr_labels)
    false_negative = unirr_image.eq(irr_labels)

    if clip:
        true_positive = true_positive.clip(domain)
        false_positive = false_positive.clip(domain)
        true_negative = true_negative.clip(domain)
        false_negative = false_negative.clip(domain)

        if state:
            fc_ = ee.FeatureCollection(os.path.join(BOUNDARIES, state))
            state = fc_.toList(fc_.size()).get(0)
            state = ee.Feature(state)
            true_positive = true_positive.clip(state)
            false_positive = false_positive.clip(state)
            true_negative = true_negative.clip(state)
            false_negative = false_negative.clip(state)

    TP = true_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FP = false_positive.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    FN = false_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )
    TN = true_negative.reduceRegion(
        geometry=domain.geometry(),
        reducer=ee.Reducer.count(),
        maxPixels=1e13,
        crs='EPSG:5070',
        scale=30
    )

    out = {'TP': TP.getInfo(), 'FP': FP.getInfo(), 'FN': FN.getInfo(), 'TN': TN.getInfo()}
    return out


def create_rf_labels(year):
    rf = ee.ImageCollection('projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp').filterDate('{}-01-01'.format(year),
                                                                                               '{}-12-31'.format(year))
    rf = rf.select('classification').mosaic()
    irrMask = rf.lt(1)
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not())
    irrImage = ee.Image(1).byte().updateMask(irrMask)
    return irrImage, unirrImage


def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/dryland')
        fallow = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/fallow')
        irrigated = ee.FeatureCollection('projects/ee-dgketchum/assets/training_polygons/irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow)
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))
    else:
        root = 'users/dgketchum/validation/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated')
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'dryland'))
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands'))

        fallow = ee.FeatureCollection(root + 'fallow')
        irrigated = ee.FeatureCollection(root + 'irrigated')
        fallow = fallow.filter(ee.Filter.eq('YEAR', year))
        non_irrigated = non_irrigated.merge(fallow).map(lambda x: x.buffer(-50))
        irrigated = irrigated.filter(ee.Filter.eq('YEAR', year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0)
    irr_labels = irr_labels.updateMask(irr_labels.Not())
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not())

    return irr_labels, unirr_labels


def metrics(arr):
    precision = arr[0, 0] / (arr[0, 1] + arr[0, 0])
    recall = arr[0, 0] / (arr[1, 0] + arr[0, 0])

    return precision, recall


def irrmapper_basin_acc(out_dir):
    globe_conf = np.zeros((2, 2))
    stations = BASIN_YEARS
    stations = {k: [0.0, v] for k, v in stations.items()}
    csv_out = os.path.join(out_dir, 'irrmapper_accuracy_colorado.csv')

    cols = ['YEAR', 'STATE', 'P', 'R', 'TP', 'FP', 'FN', 'TN']
    ind_len = 0
    for k, v in stations.items():
        ind_len += len(v[1])

    df = pd.DataFrame(columns=cols)

    ind = 0
    _all = len(stations.keys())
    for num, (sid, (area, years)) in enumerate(stations.items()):
        out_json = os.path.join(out_dir, 'basin_json', '{}.json'.format(sid))
        if not years:
            continue
        basin_conf = np.zeros((2, 2))
        print(sid)
        # add for-loop over basin states if computation is timing out
        # to clip to smaller area and break up the counting
        for year in years:
            row = {c: None for c in cols}
            try:
                row['YEAR'] = year
                row['STAID'] = sid

                irr_labels, unirr_labels = create_irrigated_labels(False, int(year))

                if not irr_labels:
                    continue

                irr_image, unirr_image = create_rf_labels(year)
                cmt = confusion(irr_labels, unirr_labels, irr_image, unirr_image, sid, clip=True, state=None)
                if not cmt:
                    continue
                if cmt['TP']['constant'] + cmt['FN']['constant'] == 0:
                    print('no positive reference data')
                    continue

                print('{} {} {}'.format(sid, year, cmt))
                for pos, ct in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ['TP', 'FP', 'FN', 'TN']):
                    globe_conf[pos] += cmt[ct]['constant']
                    basin_conf[pos] += cmt[ct]['constant']
                    row[ct] = cmt[ct]['constant']

                p, r = metrics(basin_conf)
                row['P'], row['R'] = np.round(p, decimals=3), np.round(r, decimals=3)

                if np.isnan(p) or np.isnan(r):
                    continue

                df = df.append(row, ignore_index=True)
                ind += 1

            except Exception as e:
                print(e, row['STAID'], row['YEAR'])
                pass

        p, r = metrics(basin_conf)
        print('prec {:.3f}, rec {:.3f}'.format(p, r))
        print(print('{} {} of {}\n{}\n\n\n'.format(sid, num, _all, basin_conf)))
        jsn = {sid: {k: v for k, v in zip(['TP', 'FP', 'FN', 'TN'], basin_conf.flatten())}}
        with open(out_json, 'w') as f:
            json.dump(jsn, f, indent=4)

        df.to_csv(csv_out.replace('.csv', '_.csv'))

    print(globe_conf)
    p, r = metrics(globe_conf)
    df.to_csv(csv_out)
    print('prec {}, rec {}'.format(p, r))
    print(datetime.now())


def initialize():
    try:
        ee.Initialize()
        print('Authorized')
    except Exception as e:
        print('You are not authorized: {}'.format(e))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
