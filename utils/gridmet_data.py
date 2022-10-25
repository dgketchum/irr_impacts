import warnings
from tempfile import mkdtemp
from datetime import datetime
from urllib.parse import urlunparse
from pandas import date_range, DataFrame
from xarray import open_dataset

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class GridMet:
    """ U of I Gridmet

    Return as numpy array per met variable in daily stack unless modified.

    """

    def __init__(self, variable=None, date=None, start=None, end=None, bbox=None,
                 target_profile=None, clip_feature=None, lat=None, lon=None):

        self.date = date
        self.start = start
        self.end = end

        if isinstance(start, str):
            self.start = datetime.strptime(start, '%Y-%m-%d')
        if isinstance(end, str):
            self.end = datetime.strptime(end, '%Y-%m-%d')
        if isinstance(date, str):
            self.date = datetime.strptime(date, '%Y-%m-%d')

        if self.start and self.end is None:
            raise AttributeError('Must set both start and end date')

        self.bbox = bbox
        self.target_profile = target_profile
        self.clip_feature = clip_feature
        self.lat = lat
        self.lon = lon

        self.service = 'thredds.northwestknowledge.net:8080'
        self.scheme = 'http'

        self.temp_dir = mkdtemp()

        self.variable = variable
        self.available = ['elev', 'pr', 'rmax', 'rmin', 'sph', 'srad',
                          'th', 'tmmn', 'tmmx', 'pet', 'vs', 'erc', 'bi',
                          'fm100', 'pdsi']

        if self.variable not in self.available:
            Warning('Variable {} is not available'.
                    format(self.variable))

        self.kwords = {'bi': 'daily_mean_burning_index_g',
                       'elev': '',
                       'erc': 'energy_release_component-g',
                       'fm100': 'dead_fuel_moisture_100hr',
                       'fm1000': 'dead_fuel_moisture_1000hr',
                       'pdsi': 'daily_mean_palmer_drought_severity_index',
                       'etr': 'daily_mean_reference_evapotranspiration_alfalfa',
                       'pet': 'daily_mean_reference_evapotranspiration_grass',
                       'pr': 'precipitation_amount',
                       'rmax': 'daily_maximum_relative_humidity',
                       'rmin': 'daily_minimum_relative_humidity',
                       'sph': 'daily_mean_specific_humidity',
                       'srad': 'daily_mean_shortwave_radiation_at_surface',
                       'th': 'daily_mean_wind_direction',
                       'tmmn': 'daily_minimum_temperature',
                       'tmmx': 'daily_maximum_temperature',
                       'vs': 'daily_mean_wind_speed', }

        if self.date:
            self.start = self.date
            self.end = self.date

        if self.start.year < self.end.year:
            self.single_year = False

        if self.start > self.end:
            raise ValueError('start date is after end date')

        if not self.bbox and not self.lat:
            raise AttributeError('No bbox or coordinates given')

    def get_point_timeseries(self):

        url = self._build_url()
        url = url + '#fillmismatch'
        xray = open_dataset(url)
        subset = xray.sel(lon=self.lon, lat=self.lat, method='nearest')
        subset = subset.loc[dict(day=slice(self.start, self.end))]
        subset = subset.rename({'day': 'time'})
        date_ind = self._date_index()
        subset['time'] = date_ind
        time = subset['time'].values
        series = subset[self.kwords[self.variable]].values
        df = DataFrame(data=series, index=time)
        df.columns = [self.variable]
        return df

    def _build_url(self):

        # ParseResult('scheme', 'netloc', 'path', 'params', 'query', 'fragment')
        if self.variable == 'elev':
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/MET/{0}/metdata_elevationdata.nc'.format(self.variable),
                              '', '', ''])
        else:
            url = urlunparse([self.scheme, self.service,
                              '/thredds/dodsC/agg_met_{}_1979_CurrentYear_CONUS.nc'.format(self.variable),
                              '', '', ''])

        return url

    def _date_index(self):
        date_ind = date_range(self.start, self.end, freq='d')

        return date_ind
# ========================= EOF ====================================================================
