import os

from utils.irrmapper_error import irrmapper_basin_acc, initialize
from utils.gridmet_error import get_rmse, gridmet_ppt_error, gridmet_etr_error

root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
if not os.path.exists(root):
    root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')

error_data = os.path.join(root, 'tables', 'error_estimates')


# ET accuracy is taken directly from water balance ET estimates in Senay et al,. RSE, 2022


def irrmapper_acc():
    initialize()
    d_ = os.path.join(error_data, 'irrmapper_accuracy')
    irrmapper_basin_acc(d_)


stations_ = os.path.join(root, 'stations', 'study_basisns_ghcn_stations.csv')
# the GHCN catalog is > 130 GB, future recommendation is use the API
# https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C00861
station_data_ = os.path.join(root, 'ghcn', 'ghcn_daily_summaries_4FEB2022')


def gridmet_ppt_error():
    for basin_ in ['missouri', 'columbia', 'colorado']:
        out_csv_ = os.path.join(error_data, 'gridmet_ppt', 'ghcn_gridmet_comp_{}.csv'.format(basin_))
        gridmet_ppt_error(stations_, station_data_, out_csv_, basin=basin_)
        get_rmse(out_csv_, vars_=['st_ppt', 'gm_ppt'], basins=True)


station_path_ = os.path.join(error_data, 'gridmet_etr', 'gridwxcomp_basins_all.csv')
# see https://github.com/WSWUP/gridwxcomp
# this doesn't use the code, but uses their station data obtained through personal comm.
etr_station_data_ = os.path.join(error_data, 'gridmet_etr', 'station_data')
etr_comp_csv = os.path.join(error_data, 'gridmet_etr', 'etr_comp.csv')


def gridmet_etr_error():
    gridmet_etr_error(station_path_, etr_station_data_, etr_comp_csv)
    get_rmse(etr_comp_csv, vars_=['st_etr', 'gm_etr'], basins=False)


if __name__ == '__main__':
    irrmapper_acc()
    gridmet_ppt_error()
    gridmet_etr_error()
# ========================= EOF ====================================================================
