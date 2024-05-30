import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Optional
import pandas as pd
from random import sample
import itertools
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from bootstrap_sampling import BootstrapSampling
from markov_chain import FirstOrderMarkovChain
from lag_one import LagOne
from annual_forecaster import autoArimaFourierFeatures, Utils #,naiveARIMA, autoArima, autoArimaDeepSearch, autoArimaBoxCoxEndogTransformer, Utils
from utilities import multisite_disaggregation, adjust_annual_precipitation
from constants_ import PRECIPITATION, DATE, LONGITUDE, LATITUDE, SAMPLE_DATE, T_MIN, T_MAX
from g2s import g2s

class IBMWeatherGen:

    DEFAULT_WET_EXTREME_THRESHOLD = 0.999

    """
    Semi-parametric stochastic weather generator capable of generate syntetic timeseries of weather observations at the time 
    resolution provided by having the precipitation as the "key variable". 

    Args
    ----------
    file_in_path : str
        Full path with the .csv file of the historic data.

    years : list
        List of years to be simulated.

    file_out_path : str
        Path where the .zip file will be stored.

    nsimulations : int
        Number of simulation requested for each year.

    Properties
    ----------
    file_out_path : pd.DataFrame
        Interval and mean of the total annual precipitation forecasted.

    number_of_simulations : pd.DataFrame
        Total annual precipitation for each year of the observed data.

    file_in_path : pd.DataFrame
        Daily precipitation of the observed data.

    simulation_year_list: dict
        The values to be used in the monthly quantile computation and in the labeling proccess.
    
    raw_data : pd.DataFrame
        Data in original spatial and temporal format, without aggregations ("multisite", "subhourly"). Receives 'Date', 'Latitude',
    'Longitude', 'precipitation' [, 't_min', 't_max', 'wind_10m', wind_100m']

    daily_data : pd.DataFrame
        Data in daily format and "single-site".

    annual_data : pd.DataFrame
        Data in annual format.

    weather_variables : list[str]
        Original names of each weather variables to be used in the new timeseries.

    weather_variables_mean : list[str]
        Weathe variable names after any needed caalculation (e.g mean).
        
    use_g2s : bool
        Flag to determine whether to use G2S for spatial variability enhancement.
    """

    def __init__(self, file_in_path, years,use_g2s,  wet_extreme_quantile_threshold: Optional[float] = DEFAULT_WET_EXTREME_THRESHOLD, nsimulations=1):
        self.number_of_simulations = nsimulations
        self.file_in_path = file_in_path
        self.simulation_year_list = years
        self.raw_data = None 
        self.daily_data = None 
        self.annual_data = None
        self.frequency = None
        self.weather_variables = list()
        self.wet_extreme_quantile_threshold = wet_extreme_quantile_threshold
        self.randomly_clip = False
        self.weather_variables_mean = list()
        self.use_g2s = use_g2s

    def closest(self, lst, K):
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

    def select_bbox(self, df):
        lat0 = sample(sorted(list(df[LATITUDE].unique()))[1:len(list(df[LATITUDE].unique()))-10], 1)[0]
        lat1 = self.closest(lst=sorted(list(df[LATITUDE].unique())), K=lat0+1)
        lon0 = sample(sorted(list(df[LONGITUDE].unique()))[1:len(list(df[LONGITUDE].unique()))-10], 1)[0]
        lon1 = self.closest(lst=sorted(list(df[LONGITUDE].unique())), K=lon0+1)
        return [lon0, lon1, lat0, lat1]

    def generate_daily(self, frequency, df):
        if frequency != 0:
            self.daily_data = df.groupby(by=[df[DATE].dt.date, LONGITUDE, LATITUDE]).sum() * (frequency / 60)
            self.daily_data.reset_index(inplace=True)
            self.daily_data[DATE] = pd.to_datetime(self.daily_data[DATE])
            self.raw_data['date_'] = self.raw_data[DATE].dt.date
        else:
            self.daily_data = df.copy()
        return self.daily_data

    def compute_daily_variables(self)->pd.DataFrame:

        self.raw_data = pd.read_csv(self.file_in_path, parse_dates=[DATE]).dropna()#.reset_index()

        if (T_MIN and T_MAX in self.raw_data.columns):
            self.raw_data = self.raw_data.assign(temperature = (self.raw_data[T_MIN] + self.raw_data[T_MAX])/2)
            self.weather_variables_mean = [element for element in list(self.raw_data.columns) if element not in [DATE, LONGITUDE, LATITUDE, T_MIN, T_MAX]]
        
        self.weather_variables = [weather_var for weather_var in self.raw_data.columns if weather_var not in [DATE, LONGITUDE, LATITUDE]]
        
        self.frequency = self.raw_data[DATE].diff().min().seconds//60

        #TODO: FOR N_SIMULATIONS == 1 --> AT THE CENTER?
        if ((max(self.raw_data.Latitude) - min(self.raw_data.Latitude)) > 1 or (max(self.raw_data.Longitude) - min(self.raw_data.Longitude)) > 1):
            
            selected_bbox = self.select_bbox(self.raw_data)
            self.sub_raw_data = self.raw_data[(self.raw_data.Longitude <= selected_bbox[1]) & (self.raw_data.Longitude >= selected_bbox[0]) & (self.raw_data.Latitude >= selected_bbox[2]) & (self.raw_data.Latitude <= selected_bbox[3])]
            
            self.daily_data = self.generate_daily(self.frequency,self.sub_raw_data)
            
        else:
            self.daily_data = self.generate_daily(self.frequency,self.raw_data)

        return self.daily_data.groupby(self.daily_data[DATE])[self.weather_variables].mean().reset_index()
    
    
    def compute_annual_prcp(self) -> pd.DataFrame:
        self.daily_data = self.compute_daily_variables()
        self.annual_data = self.daily_data.groupby(self.daily_data[DATE].dt.year)[PRECIPITATION].sum()
        self.annual_data.index = pd.period_range(str(self.annual_data.index[0]), str(self.annual_data.index[-1]), freq='Y')
        return self.annual_data
    
    # def generate_forecasted_values(self):
        
    #     self.annual_data = self.compute_annual_prcp()
        
    #     #l_m = [2]
    #     l_m = range(2, len(self.annual_data.index), 3)
    #     list_autoArimaFourierFeatures = []

    #     for m in l_m:
    #         for k in range(1,(int(m/2)+1)):
    #             list_autoArimaFourierFeatures.append(autoArimaFourierFeatures(k=k,m=m))

    #     list_models = [#naiveARIMA(p=1, d=0, q=1),
    #                 #autoArima(),
    #                 #autoArimaDeepSearch(),
    #                 #autoArimaBoxCoxEndogTransformer()  
    #                 ]

    #     #list_models = list_models + list_autoArimaFourierFeatures
    #     list_models = list_autoArimaFourierFeatures

    #     return Utils.model_selection(list_models, self.annual_data)  
    
    def generate_forecasted_values(self):
        
        self.annual_data = self.compute_annual_prcp()
        list_autoArimaFourierFeatures = []
        comb_list = []

        l_m = list(range(2, len(self.annual_data.index), 3))
        l_m = sample(list(l_m), k=4)
        comb_list = [list(itertools.product( [m], list( range(1,(int(m/2)+1)))) ) for m in list(l_m)]
        comb_list = list(itertools.chain(*comb_list))

        for m,k in comb_list:
            list_autoArimaFourierFeatures.append(autoArimaFourierFeatures(k=k,m=m))
        
        list_models = list_autoArimaFourierFeatures

        return Utils.model_selection(list_models, self.annual_data) 
    
    def adjust_prediction(self, prediction) -> pd.DataFrame:

        year = prediction['mean_ci_lower'].index.values[0]
        
        if prediction['mean_ci_lower'].values[0] < 0:
            stp = self.annual_data.values.std()*0.8
            prediction['mean_ci_lower'] = self.annual_data[str(year)]
            prediction['mean'] = prediction['mean_ci_lower'] + stp
            prediction['mean_ci_upper'] = prediction['mean_ci_lower'] + 2*stp

        else:
            vle = ((prediction['mean'] - self.annual_data[str(year)])/self.annual_data[str(year)]).values[0]
            stp = self.annual_data.values.std()*0.8
            
            if vle < -0.05: #annual > predicted_mean
                diff = prediction['mean'] - prediction['mean_ci_lower'] 
                prediction['mean'] = prediction['mean'] - (0.95*(prediction['mean'] - self.annual_data[str(year)])).values[0]
                prediction['mean_ci_lower'] = (prediction['mean_ci_lower'] + diff)
                prediction['mean_ci_upper'] = (prediction['mean_ci_upper'] + diff)
                #print(f'MENOR: {prediction}')
            if vle > 0.05:
                diff = prediction['mean'] - prediction['mean_ci_lower'] 
                prediction['mean'] = prediction['mean'] - (0.95*abs(prediction['mean'] - self.annual_data[str(year)])).values[0]
                prediction['mean_ci_lower'] = (prediction['mean_ci_lower'] - diff)
                prediction['mean_ci_upper'] = (prediction['mean_ci_upper'] - diff)
        
        return prediction

    def generate_weather_series(self):

        simulations = list()
        for simulation_year in self.simulation_year_list:

            for num_simulation in range(self.number_of_simulations):
                
                print(f'\nYear: [[{simulation_year}]] | Simulation: [[{num_simulation+1}/{self.number_of_simulations}]]')
                self.annual_data = self.compute_annual_prcp()

                if str(simulation_year) in self.annual_data.index:
                    predicted = {}
                    stp = self.annual_data.values.std()*0.8
                    print('Predicting the range for the year.')
                    predicted['mean'] = self.annual_data[str(simulation_year)]
                    predicted['mean_ci_lower'] = predicted['mean'] - stp
                    predicted['mean_ci_upper'] = predicted['mean'] + stp
                    predicted = pd.DataFrame(index=[str(simulation_year)], data=predicted)
                    #print(predicted)

                else:
                    print('ARIMA forecast being done...This might take a while.')
                    self.best_annual_forecaster = self.generate_forecasted_values()
                    predicted = self.best_annual_forecaster.predict_year(str(simulation_year))
                    # print(type(predicted))
                    # print(predicted.index)
                    # print(predicted)
                
                bootstrap = BootstrapSampling(predicted, self.annual_data.to_frame(), self.daily_data, self.wet_extreme_quantile_threshold)
                training_data, thresh = bootstrap.get_labels_states()

                prcp_occurence = FirstOrderMarkovChain(training_data, simulation_year, self.weather_variables)
                df_simulation, thresh_markov_chain = prcp_occurence.simulate_state_sequence()
                
                single_timeseries = LagOne(training_data, df_simulation, self.weather_variables, self.weather_variables_mean)
                df_simulation = single_timeseries.get_series()

                df_simulation = multisite_disaggregation(df_simulation, self.raw_data, self.frequency)

                df_simulation = adjust_annual_precipitation(df_simulation, predicted)
                
                ##print(f"Use G2S: {self.use_g2s}")
                if self.use_g2s:
                    df_simulation = self.improve_spatial_variability(df_simulation)
                
                df_simulation = df_simulation.assign( n_simu = num_simulation+1 )
                simulations.append(df_simulation.drop([SAMPLE_DATE], axis=1).set_index(DATE)) #for tests, consider the 'sample_date'

        dfnl = pd.concat(simulations)


        return dfnl
    
    def improve_spatial_variability(self, df_simulation):
        for variable in self.weather_variables:
            # Prepare the training image and destination image for QS
            ti = df_simulation[variable].values
            di = np.full_like(ti, np.nan)
            di[np.isnan(ti)] = np.nan
            sim, *_ = g2s('-a', 'qs',
                                  '-ti', ti,
                                  '-di', di,
                                  '-dt', [0],  # Zero for continuous variables
                                  '-k', 1.2,
                                  '-n', 50,
                                  '-j', 0.5)
            df_simulation[variable] = sim
        return df_simulation
    
    def downscale(self, df_simulation, reference_images):
        downscaled_images = []
        for variable, ref_img in zip(self.weather_variables, reference_images):
            di = df_simulation[variable].values
            sim, *_ = g2s('-a', 'qs',
                                  '-ti', ref_img,
                                  '-di', di,
                                  '-dt', [0],  # Assuming continuous variable
                                  '-k', 1.2,
                                  '-n', 50,
                                  '-j', 0.5)
            downscaled_images.append(sim)
        return downscaled_images
    
    def generate_extreme_events(self, df_simulation, return_period):
        extreme_images = []
        for variable in self.weather_variables:
            di = df_simulation[variable].values
            sim, *_ = g2s('-a', 'qs',
                                  '-ti', di,
                                  '-di', di,
                                  '-dt', [0],  # Assuming continuous variable
                                  '-k', 1.2,
                                  '-n', 50,
                                  '-j', 0.5)
            extreme_images.append(sim)
        return extreme_images
    
    
    
    
