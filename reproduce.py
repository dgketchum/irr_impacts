import os

from gage_data import get_station_daily_data, get_station_daterange_data
from gridded_data import export_gridded_data
from input_tables import merge_gridded_flow_data
from climate_flow import climate_flow_correlation
from trends_analysis import initial_trends_test, run_bayes_regression_trends
from trends_analysis import bayes_write_significant_trends
from crop_consumption_flow import initial_impacts_test, run_bayes_regression_cc_qres
from crop_consumption_flow import bayes_write_significant_cc_qres

root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
old_root = os.path.join('/media', 'research', 'IrrigationGIS', 'gages')

gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')

daily_q = os.path.join(root, 'hydrographs', 'daily_q')
daily_q_fig = os.path.join(root, 'hydrographs', 'daily_hydrograph_plots')

monthly_q = os.path.join(root, 'hydrographs', 'monthly_q')
monthly_q_fig = os.path.join(root, 'hydrographs', 'monthly_hydrograph_plots')
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
desc = 'export_ee_gridded_21OCT2022'


def get_gridded_data():
    # extract SSEBop ET, irrigation status, and TerraClimate at monthly time-step from Earth Engine
    # precede gage data by five years so climate-flow correlation can look back 60 months before first discharge month
    extract_years = list(range(1982, 1985))
    export_gridded_data(basins, bucket, extract_years, min_years=5, description=desc, debug=False)
    # transfer gridded data from GCS bucket to local system


extracts = os.path.join(root, 'ee_exports', 'IrrMapperComp_21DEC2021')
# data_tables = os.path.join(root, 'compiled_tables', 'IrrMapperComp_21DEC2021')
data_tables = os.path.join(old_root, 'merged_q_ee', 'monthly_ssebop_tc_gm_q_Comp_21DEC2021_unfiltered_q')


def build_tables():
    # merge aggregated gridded data with monthly discharge at each gage-basin
    merge_gridded_flow_data(extracts, monthly_q, data_tables, start_yr, end_yr, glob=desc)


analysis_directory = os.path.join(root, 'analysis')
climate_flow_data = os.path.join(analysis_directory, 'climate_flow_{}')


def climate_flow_correlations():
    # find each gage's monthly characteristic response period to basin climate (precip and reference ET)
    for m in months:
        out_data = climate_flow_data.format(m)
        climate_flow_correlation(data_tables, m, gages_metadata, out_data)


trends_initial = os.path.join(analysis_directory, 'trends_initial_{}')
trends_bayes = os.path.join(analysis_directory, 'trends_bayes_{}')
trends_traces = os.path.join(root, 'traces', 'trends')
processes = 1
overwrite_bayes = False


def trends():
    # test for trends by first checking OLS, then if p < 0.05, run the trend test with error in Bayes regression
    for m in months:
        in_data = climate_flow_data.format(m)
        out_data = trends_initial.format(m)
        initial_trends_test(in_data, out_data)

        in_data = out_data
        out_data = trends_bayes.format(m)
        run_bayes_regression_trends(trends_traces, in_data, processes, overwrite_bayes)
        bayes_write_significant_trends(in_data, trends_traces, out_data)


# crop consumption and climate-normalized flow data
cc_qres_data = os.path.join(analysis_directory, 'cc_qres_data_{}')
cc_qres_traces = os.path.join(root, 'traces', 'cc_qres')

# climate-normalized crop consumption and climate-normalized flow data
ccres_qres_data = os.path.join(analysis_directory, 'ccres_qres_data_{}')
ccres_qres_traces = os.path.join(root, 'traces', 'ccres_qres')


def irrigation_impacts():
    for m in months:
        in_data = climate_flow_data.format(m)

        out_data = cc_qres_data.format(m)
        initial_impacts_test(in_data, data_tables, out_data, cc_res=False)
        run_bayes_regression_cc_qres(cc_qres_traces, out_data, processes, overwrite_bayes)

        out_data = ccres_qres_data.format(m)
        initial_impacts_test(in_data, data_tables, out_data, cc_res=True)
        run_bayes_regression_cc_qres(cc_qres_traces, out_data, processes, overwrite_bayes)


if __name__ == '__main__':
    # get_gage_data()
    get_gridded_data()
    # build_tables()
    climate_flow_correlations()
# ========================= EOF ====================================================================
