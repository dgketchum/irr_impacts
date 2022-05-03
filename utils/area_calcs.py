import os
import json

import fiona
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from pandas import read_csv
import matplotlib.pyplot as plt
import geopandas as gpd


def get_domain_area_info(_basins, _counties, _nass):
    l = [os.path.join(_basins, x) for x in os.listdir(_basins) if x.endswith('.shp')]
    basin_geos = []
    for s in l:
        with fiona.open(s, 'r') as src:
            for f in src:
                basin_geos.append(shape(f['geometry']))

    domain = unary_union(basin_geos)

    counties = []
    with fiona.open(_counties, 'r') as src:
        for f in src:
            if shape(f['geometry']).intersects(domain):
                counties.append(f['properties']['GEOID'])

    co_areas = {}
    df = read_csv(_nass, index_col='GEOID')

    irr_cols, cult_cols = [c for c in df.columns if 'IRR' in c], [c for c in df.columns if 'CROP' in c]
    for i, r in df.iterrows():
        co = str(i).rjust(5, '0')
        if co in counties:
            irr = r[irr_cols].mean(skipna=True) / 247.105
            cult = r[cult_cols].mean(skipna=True) / 247.105
            land = r['land'] / 247.105
            water = r['water'] / 247.105
            co_areas[co] = {'irr': irr, 'cult': cult, 'land': land, 'water': water}

    tot_irr = np.nansum(np.array([v['irr'] for k, v in co_areas.items()]))
    tot_cult = np.nansum(np.array([v['cult'] for k, v in co_areas.items()]))
    tot_land = np.nansum(np.array([v['land'] for k, v in co_areas.items()]))
    tot_water = np.nansum(np.array([v['water'] for k, v in co_areas.items()]))
    tot_area = tot_land + tot_water
    print('irr: {:.1f}\n cult: {:.1f}\n total area: {:.1f}'.format(tot_irr,
                                                                   tot_cult,
                                                                   tot_area))
    print('cult frac: {:.3f}'.format(tot_cult / tot_area))
    print('irr/cult: {:.3f}'.format(tot_irr / tot_cult))


def get_impacted_vs_nonimpacted_areas(gage_basins, significant_imapacts):
    with open(significant_imapacts, 'r') as qcc:
        dct = json.load(qcc)

    sig_sid = [k for k, v in dct.items() if v != 'None']
    insig_sid = [k for k, v in dct.items() if v == 'None']

    basin_geos = []
    insig_basin_geos = []
    with fiona.open(gage_basins, 'r') as src:
        for f in src:
            sid = f['properties']['STAID']
            geo = shape(f['geometry'])
            if sid in sig_sid:
                basin_geos.append((geo.area / 1e9, sid, geo))
            elif sid in insig_sid:
                insig_basin_geos.append((geo.area / 1e9, sid, geo))
            else:
                pass

    basin_geos = sorted(basin_geos, key=lambda x: x[0], reverse=True)
    first = True
    parent_basins = []
    for a, s, g in basin_geos:
        if first:
            parent_basins = [(a, s, g)]
            first = False
        else:
            inter = [g.intersection(geo_[2]).area / g.area < 0.1 for geo_ in parent_basins]
            if not np.any(inter):
                parent_basins.append((a, s, g))

    insig_geo = []
    for a, s, g in insig_basin_geos:
        inter = [g.intersection(geo_[2]).area / g.area < 0.9 for geo_ in parent_basins]
        if not np.any(inter):
            if len(insig_geo) > 0:
                inter = [g.intersection(geo_[2]).area / g.area < 0.1 for geo_ in insig_geo]
                if not np.any(inter):
                    insig_geo.append((a, s, g))
            insig_geo.append((a, s, g))
    pass


if __name__ == '__main__':
    basin = '../paper_data/basins'
    # county = '../paper_data/counties'
    # nass_info = '../paper_data/USDA_NASS_2002-2017.csv'
    # get_domain_area_info(basin, county, nass_info)
    rt = '/media/research/IrrigationGIS/gages/station_metadata'
    gage_basins_ = '../paper_data/basins/gage_basins.shp'
    sig_basins = os.path.join(rt, 'cci_impacted_bayes_ccerr_0.196_qreserr_0.17_forShape.json')
    get_impacted_vs_nonimpacted_areas(gage_basins_, sig_basins)

# ========================= EOF ====================================================================
