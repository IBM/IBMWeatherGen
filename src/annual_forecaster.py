from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from pmdarima import model_selection
from pmdarima import pipeline
from pmdarima import preprocessing as ppc
from pmdarima import arima
from pmdarima.preprocessing import BoxCoxEndogTransformer

class Forecaster():
    """
    The Forecaster defines the interface of interest to clients.
    """

    def __init__(self, model: Model) -> None:
        self.__model = model

    @property
    def model(self) -> Model:
        return self.__model

    @model.setter
    def model(self, model: Model) -> None:
        self.__model = model

    def fit(self, data: pd.Series) -> None:
        #print("Forecaster: fitting model")
        self.__model.fit(data)

    def summary(self):
        self.__model.summary()

    def plot(self) -> None:
        self.__model.plot()

    def predict_year(self, year):

        fitted_model = self.model.fitted_model
        data = self.model.data

        in_sample_preds, in_sample_confint = fitted_model.predict_in_sample(X=None, return_conf_int=True)
        out_sample_preds, out_sample_confint = fitted_model.predict(n_periods=100, return_conf_int=True)

        index = data.index.to_series().astype(str)

        df_in = pd.DataFrame(index=index, data={'mean': in_sample_preds,
                                                'mean_ci_lower': in_sample_confint[:, 0],
                                                'mean_ci_upper': in_sample_confint[:, 1]})

        index = list(range(int(index[-1]), int(index[-1])+101))
        index = index[1:]
        index = [str(e) for e in index]

        df_out = pd.DataFrame(index=index, data={'mean': out_sample_preds,
                                                 'mean_ci_lower': out_sample_confint[:, 0],
                                                 'mean_ci_upper': out_sample_confint[:, 1]})

        df_preds = pd.concat([df_in, df_out])

        return df_preds[df_preds.index == year]

class Model(ABC):
    """
    The Model interface declares operations common to all supported versions
    of some algorithm.

    The Forecaster uses this interface to call the algorithm defined by Concrete
    Strategies.
    """

    @abstractmethod
    def fit(self, data: pd.Series):
        pass

    @abstractmethod
    def predict_year(self, year) -> pd.DataFrame:
        """
        it must merge in sample predictions and out sample prediction (forecasting)
        """
        pass

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def summary(self):
        pass

class naiveARIMA(Model):

    def __init__(self, p: int = 1, d: int = 1, q: int = 1):
        self.__fitted_model = []
        self.__data = []
        self.__p = p
        self.__d = d
        self.__q = q
        self.__name = 'naive ARIMA'

    @property
    def name(self):
        return self.__name

    @property
    def fitted_model(self):
        return self.__fitted_model

    @property
    def data(self):
        return self.__data

    def fit(self, data: pd.Series):
        self.__data = data
        model = pm.ARIMA( order=(self.__p, self.__d, self.__q ),seasonal_order=(0, 1, 1, 12),suppress_warnings=True)
        self.__fitted_model = model.fit(data)
        return self.__fitted_model

    def plot(self):
        Utils.plot_annual_forecaster(self)

    def predict_year(self, year):

        # in_sample_preds, in_sample_confint = self.__fitted_model.predict_in_sample(X=None, return_conf_int=True)
        # out_sample_preds, out_sample_confint = self.__fitted_model.predict(n_periods=100, return_conf_int=True)
        #
        # df_in = pd.DataFrame(index=self.__data.index(), data={'mean': in_sample_preds,
        #                                                       'mean_ci_lower': in_sample_confint[:, 0],
        #                                                       'mean_ci_upper': in_sample_confint[:, 1]})
        #
        # index = pd.date_range(start=self.__data.index[-1].to_timestamp(), periods=101, freq='Y')
        # index = index[1:]
        #
        # df_out = pd.DataFrame(index=index, data={'mean': out_sample_preds,
        #                                          'mean_ci_lower': out_sample_confint[:, 0],
        #                                          'mean_ci_upper': out_sample_confint[:, 1]})
        #
        # df_preds = pd.concat([df_in, df_out])
        #
        # return df[df_preds.index == year]
        pass

    def summary(self):
        #print(self.__fitted_model.summary())
        pass

class autoArimaFourierFeatures(Model):

    def __init__(self, m: int = 12, k: int = 4):
        self.__fitted_model = []
        self.__m = m
        self.__k = k
        self.__data = []
        self.__name = 'auto ARIMA Fourier features k '+str(k) + 'm' +str(m)

    @property
    def name(self):
        return self.__name

    @property
    def fitted_model(self):
        return self.__fitted_model

    @property
    def data(self):
        return self.__data

    def fit(self, data: pd.Series):
        self.__data = data

        # -----------
        pipe = pipeline.Pipeline([
            ("fourier", ppc.FourierFeaturizer(self.__m, self.__k)),
            ("arima", arima.AutoARIMA(stepwise=True, #trace=1, 
                                      error_action="ignore",
                                      seasonal=False,  # because we use Fourier
                                      suppress_warnings=True))
        ])

        self.__fitted_model = pipe.fit(data)

        # -----------
        # self.__fitted_model = pm.auto_arima(data, error_action='ignore', trace=True,suppress_warnings=True, maxiter=5, seasonal=True, m=12)

    def predict_year(self, year) -> pd.DataFrame:
        # """
        # it must merge in sample predictions and out sample prediction (forecasting)
        # """
        # preds, conf_int = self.__fitted_model.predict(n_periods=10, return_conf_int=True)
        # print('autoarima predictions \n')
        # print(preds, conf_int)
        pass

    def plot(self):
        Utils.plot_annual_forecaster(self)

    def summary(self):
        #print(self.__fitted_model.summary())
        pass



class autoArimaBoxCoxEndogTransformer(Model):

    def __init__(self, m: int = 12, k: int = 4):
        self.__fitted_model = []
        self.__m = m
        self.__k = k
        self.__data = []
        self.__name = 'auto ARIMA BoxCoxEndogTransformer'

    @property
    def name(self):
        return self.__name

    @property
    def fitted_model(self):
        return self.__fitted_model

    @property
    def data(self):
        return self.__data

    def fit(self, data: pd.Series):
        self.__data = data

        # -----------
        pipe = pipeline.Pipeline([
            ('boxcox', BoxCoxEndogTransformer(lmbda2=1e-6)),  # lmbda2 avoids negative values
            ('arima', pm.AutoARIMA(seasonal=True,
                                   suppress_warnings=True,
                                   #trace=True,
                                   error_action='ignore'
                                   ))
        ])

        self.__fitted_model = pipe.fit(data)

        # -----------
        # self.__fitted_model = pm.auto_arima(data, error_action='ignore', trace=True,suppress_warnings=True, maxiter=5, seasonal=True, m=12)

    def predict_year(self, year) -> pd.DataFrame:
        # """
        # it must merge in sample predictions and out sample prediction (forecasting)
        # """
        # preds, conf_int = self.__fitted_model.predict(n_periods=10, return_conf_int=True)
        # print('autoarima predictions \n')
        # print(preds, conf_int)
        pass

    def plot(self):
        Utils.plot_annual_forecaster(self)

    def summary(self):
        #print(self.__fitted_model.summary())
        pass

class autoArima(Model):

    def __init__(self):
        self.__fitted_model = []
        self.__data = []
        self.__name = 'auto ARIMA'

    @property
    def name(self):
        return self.__name

    @property
    def fitted_model(self):
        return self.__fitted_model

    @property
    def data(self):
        return self.__data

    def fit(self, data: pd.Series):
        self.__data = data

        # -----------
        self.__fitted_model = pm.auto_arima(data,
                                            start_p = 2, # as in R code
                                            start_q=2, # as in R code
                                            start_P=0,
                                            start_Q=0,
                                            max_p=2, # as in R code
                                            max_q=2, # as in R code
                                            max_P=0, # as in R code
                                            max_Q=0, # as in R code
                                            seasonal=True, # as in R code
                                            #stepwise=True,
                                            #suppress_warnings=True,
                                            #D=10,
                                            #max_D=10,
                                            error_action='ignore')


    def predict_year(self, year) -> pd.DataFrame:
        # """
        # it must merge in sample predictions and out sample prediction (forecasting)
        # """
        # preds, conf_int = self.__fitted_model.predict(n_periods=10, return_conf_int=True)
        # print('autoarima predictions \n')
        # print(preds, conf_int)
        pass

    def plot(self):
        Utils.plot_annual_forecaster(self)

    def summary(self):
        #print(self.__fitted_model.summary())
        pass

class autoArimaDeepSearch(Model):

    def __init__(self):
        self.__fitted_model = []
        self.__data = []
        self.__name = 'auto ARIMA Deep Search'

    @property
    def name(self):
        return self.__name

    @property
    def fitted_model(self):
        return self.__fitted_model

    @property
    def data(self):
        return self.__data

    def fit(self, data: pd.Series):
        self.__data = data

        # -----------
        self.__fitted_model = pm.auto_arima(data,start_p=1, start_q=1, d=0, start_P=1, start_Q=1,
                     max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True, out_of_sample_size=1,
                     stepwise=True, suppress_warnings=True, D=10, max_D=10,
                     error_action='ignore', m=1, #trace=True
                                            )


    def predict_year(self, year) -> pd.DataFrame:
        # """
        # it must merge in sample predictions and out sample prediction (forecasting)
        # """
        # preds, conf_int = self.__fitted_model.predict(n_periods=10, return_conf_int=True)
        # print('autoarima predictions \n')
        # print(preds, conf_int)
        pass

    def plot(self):
        Utils.plot_annual_forecaster(self)

    def summary(self):
        #print(self.__fitted_model.summary())
        pass

class Utils():
    def __init__(self):
        pass

    @staticmethod
    def generate_fake_data(n_days: int = 3650):
        dates = pd.date_range(start="2000-01-01", periods=n_days)
        lats = [1, 2, 3, 4]
        longs = [-1, -2, -3, -4]
        data = list(itertools.product(dates, lats, longs))

        weather_data_df = pd.DataFrame(data=data, columns=['Date', 'Latitude', 'Longitude'])
        weather_data_df['Precipitation'] = np.random.gamma(shape=2, scale=2, size=n_days * len(lats) * len(longs))
        weather_data_df['Temperature'] = np.random.normal(loc=25, scale=0.25, size=n_days * len(lats) * len(longs))

        return weather_data_df

    @staticmethod
    def plot_annual_forecaster(model):
        fig, ax = plt.subplots(figsize=(15, 5))

        in_sample_preds, in_sample_confint = model.fitted_model.predict_in_sample(X=None, return_conf_int=True)

        model.data.iloc[0:].plot(ax=ax, color='k', style='x')
        plt.plot(model.data.index, in_sample_preds, color='b')
        plt.scatter(model.data.index, in_sample_preds, color='b')
        ax.fill_between(model.data.index, in_sample_confint[:, 0], in_sample_confint[:, 1], alpha=0.1, color='b')

        # preds data frame
        preds, conf_int = model.fitted_model.predict(n_periods=50, return_conf_int=True)

        index = pd.date_range(start=model.data.index[-1].to_timestamp(), periods=50, freq='Y')
        fcast = pd.DataFrame(index=index, data={'mean': preds,
                                                'mean_ci_lower': conf_int[:, 0],
                                                'mean_ci_upper': conf_int[:, 1]})

        fcast.drop(index=fcast.index[0], inplace=True)  # first index is the last year in observations

        # preds plot
        fcast['mean'].plot(ax=ax, style='k--')
        ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1);
        plt.scatter(fcast.index, fcast['mean'])

        ax.set_ylim(model.data.min() - 1000, model.data.max() + 1000)
        plt.title(model.name)
        plt.show()

    @staticmethod
    def model_selection(list_models, area_avg_annual_precipitation):

        annualForecaster = Forecaster(naiveARIMA())
        list_fitted_models = []
        list_errors = []

        cv = model_selection.SlidingWindowForecastCV(step=1, h=1, window_size=20)
        # cv = model_selection.RollingForecastCV(step=1, h=1)
        for arima_model in list_models:
            print(arima_model.name)
            annualForecaster.model = arima_model
            annualForecaster.fit(data=area_avg_annual_precipitation)
            list_fitted_models.append(annualForecaster.model)
            # cv score
            try:
                scores = model_selection.cross_val_score(annualForecaster.model.fitted_model,
                                                         area_avg_annual_precipitation,
                                                         scoring='mean_absolute_error', cv=cv, verbose=2)
            except ValueError as e:
                #print('Value Error')
                scores = np.array([100000, 100000, 100000]) # big scores


            list_errors.append(np.average(scores))

        #print(list_errors)
        #print("Forecasters : {}".format([e.name for e in list_fitted_models]))
        better_index = np.nanargmin(list_errors)
        #print("Lowest average SMAPE: {} (model {})".format(list_errors[better_index], list_fitted_models[better_index].name))

        annualForecaster.model = list_fitted_models[better_index]
        return annualForecaster