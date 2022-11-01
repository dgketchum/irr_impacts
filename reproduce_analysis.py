import os
import json

import numpy as np

from gage_data import get_station_daily_data, get_station_daterange_data
from gridded_data import export_gridded_data
from input_tables import merge_gridded_flow_data
from climate_flow import climate_flow_correlation
from trends_analysis import initial_trends_test, run_bayes_regression_trends
from trends_analysis import bayes_write_significant_trends
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


def get_gage_data():
    # gather daily streamflow data from basins with irrigation, saving only complete months' records
    get_station_daily_data('{}-01-01'.format(start_yr), '{}-12-31'.format(end_yr), gages_metadata,
                           daily_q, plot_dir=daily_q_fig, overwrite=False)
    # sum streamflow over each month, convert to cubic meters per month
    get_station_daterange_data(daily_q, monthly_q, convert_to_mcube=True, resample_freq='M', plot_dir=monthly_q_fig)


basins = 'users/dgketchum/gages/gage_basins'
bucket = 'wudr'
desc = 'export_cc_28OCT2022'
with open(gages_metadata, 'r') as fp:
    stations = json.load(fp)


def get_gridded_data():
    # extract SSEBop ET, irrigation status, and TerraClimate at monthly time-step from Earth Engine
    # precede gage data by five years so climate-flow correlation can look back 60 months before first discharge month
    extract_years = np.arange(start_yr, end_yr + 1)
    basin_ids = [station_id for station_id, _ in stations.items()]
    export_gridded_data(basins, bucket, extract_years, features=basin_ids, min_years=30, description=desc, debug=False)
    # transfer gridded data from GCS bucket to local system


# extracts = os.path.join(root, 'tables', 'gridded_tables', 'IrrMapperComp_21OCT2022')
# data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')

# static irrigation mask
extracts = os.path.join(root, 'tables', 'gridded_tables', 'IrrMapperComp_static_24OCT2022')
data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_static_24OCT2022')


def build_tables():
    # merge aggregated gridded data with monthly discharge at each gage-basin
    # start 5 years before the study period to account for basin lag times
    # merge_gridded_flow_data(extracts, monthly_q, data_tables, start_yr - 5, end_yr, glob=desc)
    merge_gridded_flow_data(extracts, monthly_q, data_tables, start_yr, end_yr, glob=desc, cc_only=True)


analysis_directory = os.path.join(root, 'analysis')
climate_flow_data = os.path.join(analysis_directory, 'climate_flow')
climate_flow_file = os.path.join(climate_flow_data, 'climate_flow_{}.json')

# climate_flow_data = os.path.join(analysis_directory, 'climate_flow_static_irr')
# climate_flow_file = os.path.join(climate_flow_data, 'climate_flow_{}.json')


def climate_flow_correlations():
    # find each gage's monthly characteristic response period to basin climate (precip and reference ET)
    for m in months:
        out_data = climate_flow_file.format(m)
        climate_flow_correlation(data_tables, m, gages_metadata, out_data)


trends_initial = os.path.join(analysis_directory, 'trends', 'trends_initial_{}.json')
trends_bayes = os.path.join(analysis_directory, 'trends', 'trends_bayes_{}.json')
trends_traces = os.path.join(root, 'traces', 'trends')

# trends_initial = os.path.join(analysis_directory, 'trends_static_irr', 'trends_initial_{}.json')
# trends_initial_figs = os.path.join(figures, 'trends_static_irr_initial')
# trends_bayes = os.path.join(analysis_directory, 'trends_static_irr', 'trends_bayes_{}.json')
# trends_traces = os.path.join(root, 'traces', 'trends_static_irr')
processes = 30
overwrite_bayes = True


def trends():
    # test for trends by first checking OLS, then if p < 0.05, run the trend test with error in Bayes regression
    for m in months[3:10]:
        print('\n\n\ntrends {}'.format(m))
        in_data = climate_flow_file.format(m)
        out_data = trends_initial.format(m)
        initial_trends_test(in_data, out_data, plot_dir=None, selectors=None)

        in_data = out_data
        out_data = trends_bayes.format(m)
        # run_bayes_regression_trends(trends_traces, in_data, processes, overwrite_bayes, selectors=['time_cc'])
        bayes_write_significant_trends(in_data, trends_traces, out_data, m)


# crop consumption and climate-normalized flow data
cc_qres_file = os.path.join(analysis_directory, 'cc_qres', 'cc_qres_initial_{}.json')
cc_qres_results_file = os.path.join(analysis_directory, 'cc_qres', 'cc_qres_bayes_{}.json')
cc_qres_traces = os.path.join(root, 'traces', 'cc_qres')

# climate-normalized crop consumption and climate-normalized flow data
ccres_qres_file = os.path.join(analysis_directory, 'ccres_qres', 'ccres_qres_initial_{}.json')
ccres_qres_results_file = os.path.join(analysis_directory, 'ccres_qres', 'ccres_qres_bayes_{}.json')
ccres_qres_traces = os.path.join(root, 'traces', 'ccres_qres')


def irrigation_impacts():
    for m in months:
        in_data = climate_flow_file.format(m)
        print('\n\n\n cc_qres {}'.format(m))

        out_data = cc_qres_file.format(m)
        # initial_impacts_test(in_data, data_tables, out_data, m, cc_res=False)
        in_data = out_data
        out_data = cc_qres_results_file.format(m)
        # run_bayes_regression_cc_qres(cc_qres_traces, in_data, processes, overwrite_bayes)
        bayes_write_significant_cc_qres(in_data, cc_qres_traces, out_data, m)

        # use same-month climate to normalize cc
        in_data = climate_flow_file.format(m)
        print('\n\n\n ccres_qres {}'.format(m))
        out_data = ccres_qres_file.format(m)
        # initial_impacts_test(in_data, data_tables, out_data, m, cc_res=True)
        in_data = out_data
        out_data = ccres_qres_results_file.format(m)
        # run_bayes_regression_cc_qres(ccres_qres_traces, in_data, processes, overwrite_bayes)
        bayes_write_significant_cc_qres(in_data, ccres_qres_traces, out_data, m)


if __name__ == '__main__':
    # get_gage_data()
    # get_gridded_data()
    # build_tables()
    # climate_flow_correlations()
    trends()
    # irrigation_impacts()
# ========================= EOF ====================================================================
