__all__ = [
    'adjust_annual_precipitation', 
    'multisite_disaggregation',
    'waterday_range'
    ]

import pandas as pd
from datetime import timedelta
import numpy as np
from scipy.stats import truncnorm
from math import log
from typing import List, Dict
from constants_ import DATE, PRECIPITATION, SAMPLE_DATE

def waterday_range(day: pd.Timedelta, window: int)->List[int]: 
    
    """ Compute the days within a given window.
        
        Parameters
        ----------
 
        day : 
            Day to be the center of a given window size. 

        window : int
            Size of the window centered on a given day.

        Returns
        ----------
            A list with 'window'size of the days of the years.
    """
    
    if (window % 2 == 1): 
        l = (window - 1) / 2 
        u = (window - 1) / 2 
    
    else:
        l = window / 2
        u = window / 2 - 1
        
    rng = [(day - timedelta(days=i)).dayofyear for i in range(-int(l), int(u)+1, 1)]
    
    return rng 

def variables_monthly_stats(df: pd.DataFrame, weather_variables: list)->Dict: 
    
    """ Compute the monthly mean and the standard deviation for each weather variable.
        
        Parameters
        ----------

        df : pd.DataFrame
            Selected daily precipitation labeled from the observed data. 

        weather_variables : list
            Weather variables names.

        Returns
        ----------
            A dict with the monthly 'mean' and 'standard deviation' of each variable.
    """

    stats_list=[]
    for month in range(1,13,1):
        stats = {}
        df_month = df[df[DATE].dt.month == month]
        for i in weather_variables:
            stats.update({'month': month, f'{i}_sd': df_month[i].std(), f'{i}_mean': df_month[i].mean()})
        stats_list.append(stats)
    
    return stats_list

def multisite_disaggregation(simulation_dates, weather_data_df, frequency)->pd.DataFrame:
    days_multisite = list()

    if frequency != 0:
        column_name = 'date_'
    else:
        column_name = DATE
    
    for i in range(len(simulation_dates)):
        tmp = weather_data_df[weather_data_df[column_name] == simulation_dates[SAMPLE_DATE][i]].rename(
            columns={DATE:SAMPLE_DATE}).assign( Date = simulation_dates[DATE][i] )

        if frequency:
            tmp[DATE] = tmp[DATE].astype('str') +' '+ tmp[SAMPLE_DATE].dt.time.astype('str')
        
        tmp[DATE] = tmp[DATE].astype('datetime64[ns]') 
        days_multisite.append(tmp)

    df = pd.concat(days_multisite).reset_index(drop=True)

    return df

#TODO: ADJUST RANGE OF PREDICTED
def adjust_annual_precipitation(df, predicted)->pd.DataFrame:

    df_annual = df.groupby(df[DATE].dt.date)[[PRECIPITATION]].mean().reset_index()
    
    df_annual[DATE] = pd.to_datetime(df_annual[DATE])

    df_annual = df_annual.groupby(df_annual[DATE].dt.year)[PRECIPITATION].sum().values[0]

    if (df_annual < predicted['mean_ci_lower'].values[0]) or (df_annual > predicted['mean_ci_upper'].values[0]):

        myclip_a = predicted['mean_ci_lower'].values[0]
        myclip_b = predicted['mean_ci_upper'].values[0]
        my_mean = predicted['mean'].values[0]
        my_std = (myclip_b - myclip_a)/(2*np.sqrt(2*log(2)))

        a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        final_prcp = truncnorm.rvs(a, b, loc = my_mean, scale = my_std, size=1)[0]
        
        RATIO = final_prcp/df_annual
        #df = df.assign( precipitation_calibrated = df[PRECIPITATION]*RATIO)
        df[PRECIPITATION] = df[PRECIPITATION]*RATIO
    
    return df

