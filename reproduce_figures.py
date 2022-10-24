import os
import json

from figs.response_histogram import plot_response

root = os.path.join('/media', 'research', 'IrrigationGIS', 'impacts')
if not os.path.exists(root):
    root = os.path.join('/home', 'dgketchum', 'data', 'IrrigationGIS', 'impacts')

figures = os.path.join(root, 'figures')
start_yr, end_yr = 1987, 2021
months = list(range(1, 13))

analysis_directory = os.path.join(root, 'analysis')
climate_flow_data = os.path.join(analysis_directory, 'climate_flow')
response_hist = os.path.join(figures, 'response_histogram.png')


def response_histogram():
    plot_response(climate_flow_data, response_hist, glob='climate_flow')





if __name__ == '__main__':
    response_histogram()
# ========================= EOF ====================================================================
