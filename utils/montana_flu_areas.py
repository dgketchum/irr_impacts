import os

import numpy as np
import fiona
from shapely.geometry import shape


def sum_flu_area(shapes):
    a_pivot, a_other = 0, 0
    with fiona.open(shapes, 'r') as src:
        for feat in src:
            geo = shape(feat['geometry'])
            props = feat['properties']
            if props['IType'] in ['S', 'F']:
                a_other += geo.area / 1e6
            if props['IType'] in ['P']:
                a_pivot += geo.area / 1e6

    return a_pivot, a_other


if __name__ == '__main__':
    r = '/media/research/IrrigationGIS/gages/flu_data'
    early_shp = os.path.join(r, 'mt_itype_2009_aea.shp')
    late_shp = os.path.join(r, 'mt_itype_2019_aea.shp')
    e_pivot, e_other = sum_flu_area(early_shp)
    l_pivot, l_other = sum_flu_area(late_shp)
    pass
# ========================= EOF ====================================================================
