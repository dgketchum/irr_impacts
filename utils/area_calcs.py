import os

import fiona
import numpy as np
from shapely.geometry import shape
from shapely.ops import unary_union
from pandas import read_csv


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
    print('cult frac: {:.3f}'.format(tot_cult/tot_area))
    print('irr/cult: {:.3f}'.format(tot_irr/tot_cult))

if __name__ == '__main__':
    basin = '../paper_data/basins'
    county = '../paper_data/counties'
    nass_info = '../paper_data/USDA_NASS_2002-2017.csv'
    get_domain_area_info(basin, county, nass_info)
# ========================= EOF ====================================================================
