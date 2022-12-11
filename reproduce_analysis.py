import os
import json
from pprint import pprint

import numpy as np

from gage_data import get_station_daily_data, get_station_daterange_data
from gridded_data import export_gridded_data
from input_tables import merge_gridded_flow_data
from climate_flow import climate_flow_correlation
from uv_trends_analysis import run_bayes_univariate_trends, summarize_univariate_trends
from ols_trends_analysis import initial_trends_test
from mv_trends_analysis import run_bayes_multivariate_trends, summarize_multivariate_trends
from crop_consumption_flow import initial_impacts_test, run_bayes_regression_cc_qres
from crop_consumption_flow import bayes_write_significant_cc_qres

root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
if not os.path.exists(root):
    root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')

figures = os.path.join(root, 'figures')

gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')

daily_q = os.path.join(root, 'tables', 'hydrographs', 'daily_q')
daily_q_fig = os.path.join(figures, 'hydrographs', 'daily_hydrograph_plots')

monthly_q = os.path.join(root, 'tables', 'hydrographs', 'monthly_q')
monthly_q_fig = os.path.join(figures, 'hydrographs', 'monthly_hydrograph_plots')
start_yr, end_yr = 1987, 2021
months = list(range(1, 13))
select = '13269000'


def get_gage_data():
    # gather daily streamflow data from basins with irrigation, saving only complete months' records
    get_station_daily_data('{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr), gages_metadata,
                           daily_q, plot_dir=daily_q_fig, overwrite=False)
    # sum streamflow over each month, convert to cubic meters per month
    get_station_daterange_data(daily_q, monthly_q, convert_to_mcube=True, resample_freq='M', plot_dir=monthly_q_fig)


basins = 'users/dgketchum/gages/gage_basins'
bucket = 'wudr'
desc = 'ee_gridded_21OCT2022'
with open(gages_metadata, 'r') as fp:
    stations = json.load(fp)

analysis_directory = os.path.join(root, 'analysis')
static_irr = False
print('\nassuming static irrigation mask: {}'.format(static_irr))

if static_irr:
    desc = 'cc_static_4NOV2022'
    extracts = os.path.join(root, 'tables', 'gridded_tables', 'IrrMapperComp_static_4NOV2022')
    data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_static_4NOV2022')
    climate_flow_data = os.path.join(analysis_directory, 'climate_flow_static_irr')
    climate_flow_file = os.path.join(climate_flow_data, 'climate_flow_{}.json')
    uv_trends_initial = os.path.join(analysis_directory, 'trends_static_irr', 'trends_initial_{}.json')
    uv_trends_traces = os.path.join(root, 'traces', 'trends_static_irr')
    uv_trends_bayes = os.path.join(analysis_directory, 'trends_static_irr', 'trends_bayes_{}.json')

else:
    extracts = os.path.join(root, 'tables', 'gridded_tables', 'IrrMapperComp_21OCT2022')
    data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
    climate_flow_data = os.path.join(analysis_directory, 'climate_flow')
    climate_flow_file = os.path.join(climate_flow_data, 'climate_flow_{}.json')

    ols_trends_data = os.path.join(analysis_directory, 'ols_trends', 'trends_initial_{}.json')

    uv_trends_traces = os.path.join(root, 'uv_traces', 'uv_trends')
    uv_trends_bayes = os.path.join(analysis_directory, 'uv_trends', 'trends_bayes_{}.json')

    mv_trends_traces = os.path.join(root, 'mv_traces', 'mv_trends')
    mv_trends_bayes = os.path.join(analysis_directory, 'mv_trends', 'trends_bayes_{}.json')

# crop consumption and climate-normalized flow data
cc_q_file = os.path.join(analysis_directory, 'cc_q', 'cc_q_initial_{}.json')
cc_q_bayes_file = os.path.join(analysis_directory, 'cc_q', 'cc_q_bayes_{}.json')
cc_q_traces = os.path.join(root, 'mv_traces', 'cc_q')


def get_gridded_data():
    # extract SSEBop ET, irrigation status, and TerraClimate at monthly time-step from Earth Engine
    # precede gage data by five years so climate-flow correlation can look back 60 months before first discharge month
    extract_years = np.arange(start_yr - 5, end_yr + 1)
    basin_ids = [station_id for station_id, _ in stations.items()]
    export_gridded_data(basins, bucket, extract_years, features=basin_ids, min_years=5, description=desc, debug=False)
    # transfer gridded data from GCS bucket to local system


def build_tables():
    # merge aggregated gridded data with monthly discharge at each gage-basin
    # start 5 years before the study period to account for basin lag times
    # merge_gridded_flow_data(extracts, monthly_q, data_tables, start_yr - 5, end_yr, glob=desc)
    merge_gridded_flow_data(extracts, monthly_q, data_tables, start_yr, end_yr, glob=desc)


def climate_flow_correlations():
    # find each gage's monthly characteristic response period to basin climate (precip and reference ET)
    for m in months:
        if m != 8:
            continue
        out_data = climate_flow_file.format(m)
        climate_flow_correlation(data_tables, m, gages_metadata, out_data)


def calculate_ols_trends():
    all_ct, ct, first = None, None, True
    for m in months:
        print('\n\n\ntrends {}'.format(m))
        in_data = climate_flow_file.format(m)
        out_data = ols_trends_data.format(m)
        ct_, all_ = initial_trends_test(in_data, out_data, plot_dir=None)
        if first:
            ct = ct_
            all_ct = all_
            first = False
        else:
            ct = {k: [v[0] + ct_[k][0], v[1] + ct_[k][1]] for k, v in ct.items()}
            all_ct = {k: v + all_[k] for k, v in all_ct.items()}

    pprint(ct)
    pprint(all_ct)


processes = 25
overwrite_bayes = True


def univariate_trends():
    for m in months:
        print('\n\n\nunivariate trends {}'.format(m))
        in_data = ols_trends_data.format(m)
        out_data = uv_trends_bayes.format(m)
        run_bayes_univariate_trends(uv_trends_traces, in_data, processes, overwrite=overwrite_bayes,
                                    selectors=['time_q'])
        # summarize_univariate_trends(in_data, uv_trends_traces, out_data, m, update_selectors=['time_q'])


def multivariate_trends():
    for m in months:
        print('\n\n\nmultivariate trends {}'.format(m))
        in_data = climate_flow_file.format(m)
        out_data = mv_trends_bayes.format(m)
        run_bayes_multivariate_trends(mv_trends_traces, in_data, processes, overwrite=overwrite_bayes,
                                      selector='time_q')
        # summarize_multivariate_trends(in_data, mv_trends_traces, out_data, m, update_selectors=['time_q'])


def irrigation_impacts():
    count = 0
    for m in months:
        in_data = climate_flow_file.format(m)
        print('\n\n\n cc_qres {}'.format(m))

        out_data = cc_q_file.format(m)
        # initial_impacts_test(in_data, data_tables, out_data, m, cc_res=False)
        in_data = out_data
        out_data = cc_q_bayes_file.format(m)
        run_bayes_regression_cc_qres(cc_q_traces, in_data, processes, overwrite_bayes)
        # bayes_write_significant_cc_qres(in_data, cc_q_traces, out_data, m)

    print(count)


if __name__ == '__main__':
    # get_gage_data()
    # get_gridded_data()
    # build_tables()
    # climate_flow_correlations()
    # calculate_ols_trends()
    univariate_trends()
    multivariate_trends()
    irrigation_impacts()
# ========================= EOF ====================================================================
