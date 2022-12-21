import os

from figs.trace_figs import plot_saved_traces, trace_only
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

select = '13172500'
# select = '09371010'

gages_metadata = os.path.join(root, 'gages', 'irrigated_gage_metadata.json')
climate_flow_data = os.path.join(analysis_directory, 'climate_flow', 'climate_flow_{}.json')
climate_flow_figs = os.path.join(figures, 'climate_flow')

gage_figs = os.path.join(figures, 'regressions_stack')
gage_figs = os.path.join(gage_figs, select)

if not os.path.exists(gage_figs):
    os.mkdir(gage_figs)

cc_q_file = os.path.join(analysis_directory, 'cc_q', 'cc_q_initial_{}.json')
cc_q_bayes_file = os.path.join(analysis_directory, 'cc_q', 'cc_q_bayes_{}.json')
cc_q_traces = os.path.join(root, 'mv_traces', 'cc_q')


def climate_flow():
    for m in months:
        data = climate_flow_data.format(m)
        plot_climate_flow(data, gage_figs, selected=[select], label=False)


def cc_qres_traces():
    for m in months:
        plot_saved_traces(gages_metadata, cc_q_traces, m, overwrite=True, station=[select],
                          selectors=['cc_q'])


trends_bayes = os.path.join(analysis_directory, 'uv_trends', 'trends_bayes_{}.json')
trends_traces = os.path.join(root, 'uv_traces', 'uv_trends')
trends_figs = os.path.join(figures, 'uv_traces', 'uv_trends')


def trends():
    for m in months:
        if m != 8:
            continue
        meta = trends_bayes.format(m)
        plot_saved_traces(meta, trends_traces, m, overwrite=True, station=[select], selectors=['time_cc', 'time_q'])


data_tables = os.path.join(root, 'tables', 'input_flow_climate_tables', 'IrrMapperComp_21OCT2022')
hydrograph_comparison = os.path.join(figures, 'cc_vs_q_barplot')


def q_cc_time_series():
    hydrograph_vs_crop_consumption(data_tables, select, hydrograph_comparison)


if __name__ == '__main__':
    # climate_flow()
    # cc_qres_traces()
    trends()
    # q_cc_time_series()
# ========================= EOF ====================================================================
