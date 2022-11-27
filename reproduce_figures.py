import os

from figs.trace_figs import plot_saved_traces
from figs.figures import plot_climate_flow
from figs.figures import hydrograph_vs_crop_consumption

import warnings

warnings.filterwarnings('ignore')

root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
if not os.path.exists(root):
    root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')

figures = os.path.join(root, 'figures')
start_yr, end_yr = 1987, 2021
months = list(range(1, 13))

analysis_directory = os.path.join(root, 'analysis')

select = '13269000'

gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
climate_flow_data = os.path.join(analysis_directory, 'climate_flow', 'climate_flow_{}.json')
climate_flow_figs = os.path.join(figures, 'climate_flow')


def climate_flow():
    for m in months[7:8]:
        fig_dir = os.path.join(climate_flow_figs, str(m))
        data = climate_flow_data.format(m)
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        plot_climate_flow(data, fig_dir, selected=[select])


ccres_qres_results_file = os.path.join(analysis_directory, 'ccres_qres', 'ccres_qres_bayes_{}.json')
ccres_qres_traces = os.path.join(root, 'traces', 'ccres_qres')
ccres_qres_figs = os.path.join(figures, 'traces', 'ccres_qres')


def cc_qres_traces():
    for m in months:
        fig_dir = os.path.join(ccres_qres_figs, str(m))
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        plot_saved_traces(gages_metadata, ccres_qres_traces, fig_dir, selected=None, overwrite=False)


trends_bayes = os.path.join(analysis_directory, 'trends', 'trends_bayes_{}.json')
trends_traces = os.path.join(root, 'traces', 'trends')
trends_figs = os.path.join(figures, 'traces', 'trends')


def trend_traces():
    for m in months[7:8]:
        fig_dir = os.path.join(trends_figs, str(m))
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)
        traces = os.path.join(trends_traces, 'time_cc')
        plot_saved_traces(gages_metadata, traces, fig_dir, selected=[select], overwrite=True)
        traces = os.path.join(trends_traces, 'time_qres')
        plot_saved_traces(gages_metadata, traces, fig_dir, selected=[select], overwrite=True)


data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
hydrograph_comparison = os.path.join(figures, 'cc_vs_q_barplot')


def q_cc_time_series():
    hydrograph_vs_crop_consumption(data_tables, select, hydrograph_comparison)


if __name__ == '__main__':
    # climate_flow()
    cc_qres_traces()
    # trend_traces()
    # q_cc_time_series()
# ========================= EOF ====================================================================
