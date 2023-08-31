from typing import List
from random import choices
import pandas as pd
import numpy as np
from constants_ import PRECIPITATION, DATE, STATE, T_MAX, WDAY, T_MIN

DEFAULT_K_VALUE = 100
DEFAULT_DRY_WET_THRESHOLD = 0.0001

class BootstrapSampling:

    """
        Using KNN and assigning weights for each k smallest distances, selects from the observed data a sample the 100 years (with repetition) 
        closest to the predicted total annual precipitation value. After, each day of this sample is labeled with 'd' (dry), 'w' (wet) or 'e' 
        (extreme), based on the quatiles values computed for each month within the historic sampled data.

        Args 
        ----------
        predicted: Dataframe with the interval and the mean values of the total annual precipitation forecasted by the 
            best model for the year inputed by the user.

        annual_prcp: Dataframe with the year and the respective total annual precipitation value, computed from the
            observed data.

        daily_prcp: Dataframe with the day and the respective total daily precipitation value, computed from the
            observed data.

        Properties
        ----------
        predicted : pd.DataFrame
            Interval and mean of the total annual precipitation forecasted.

        annual_prcp : pd.DataFrame
            Total annual precipitation for each year of the observed data.

        daily_prcp : pd.DataFrame
            Daily precipitation of the observed data.

        weather_states_thresholds: dict
            The values to be used in the monthly quantile computation and in the labeling proccess.

        """

    def __init__(self, 
                predicted: pd.DataFrame=None, 
                annual_prcp: pd.DataFrame=None,
                daily_prcp: pd.DataFrame=None,
                wet_extreme_quantile_threshold: float=None,
                ) -> None:
        
        self.predicted = predicted
        self.annual_prcp = annual_prcp
        self.daily_prcp = daily_prcp
        self.weather_states_thresholds = {'dry': 0, 'wet': DEFAULT_DRY_WET_THRESHOLD, 'extreme': wet_extreme_quantile_threshold} 
        
        self.train_data = None
        self.thresh = list()

    def get_resampling_years(self) -> List[int]: 
    
        """ Get k years from historic data by performing distance with the modeled year.

            Returns
            ----------
                A list with k nearest years to the total annual precipitation value forecasted.
        """

        k = np.round( max(np.sqrt(len(self.annual_prcp)), 0.5*len(self.annual_prcp) ), 0) 
        
        self.annual_prcp = self.annual_prcp.assign( distance =  np.sqrt((self.predicted['mean'].values[0] - self.annual_prcp[PRECIPITATION])**2) )
        self.annual_prcp.sort_values('distance', inplace=True)
        self.annual_prcp.index = self.annual_prcp.index.set_names(['year'])
        self.annual_prcp.reset_index(inplace=True)
        self.annual_prcp = self.annual_prcp.head(int(k))
        
        self.annual_prcp = self.annual_prcp.assign( prob = (1/(self.annual_prcp.index+1))/(sum(1/(self.annual_prcp.index+1))) )
        
        selected_years = choices(population=self.annual_prcp['year'], weights=self.annual_prcp['prob'], k=DEFAULT_K_VALUE)

        ryears = list(map(lambda x: x.year, selected_years))

        return ryears

    def get_training_data(self)->pd.DataFrame:
        
        """ Get daily data from the historic based on the years resampled.

            Returns
            ----------
                A DataFrame with the days and repective precipitation for each year resampled (with repetition).
        """

        resampled_years = self.get_resampling_years()

        dfs = list()
        
        for year in resampled_years:
            ttemp=self.daily_prcp[self.daily_prcp[DATE].dt.year.isin([year])]
            dfs.append(pd.DataFrame(ttemp.sort_values(by=DATE)))

        train_data = pd.concat(dfs).reset_index(drop=True)

        return train_data

    def get_labels_states(self)->pd.DataFrame:
        
        """ Assign labels for each day based on a threshold. 

            Returns
            ----------
                A DataFrame with the days labeled as 'd' (dry), 'w' (wet) or 'e' (extreme). The assignment of each label is based on monthly quatiles
            using the default thresholds.

        """
        
        self.train_data = self.get_training_data()
        self.train_data = self.train_data.assign(state = list(self.weather_states_thresholds.keys())[1][0])                                                          
        for month in range(1, 13, 1):
            
            td = self.train_data[self.train_data[DATE].dt.month == month]

            self.thresh.append({'month': month, 
                                'thresholds': {'dry_wet': self.weather_states_thresholds['wet'], 
                                               'wet_extreme': np.quantile(td[PRECIPITATION], self.weather_states_thresholds['extreme'])} })
            
            self.train_data.loc[(self.train_data[DATE].dt.month == month) 
                            & (self.train_data[PRECIPITATION] <= self.thresh[month-1]['thresholds']['dry_wet']), STATE] = list(self.weather_states_thresholds.keys())[0][0]
            self.train_data.loc[(self.train_data[DATE].dt.month == month) 
                            & (self.train_data[PRECIPITATION] >= self.thresh[month-1]['thresholds']['wet_extreme']), STATE] = list(self.weather_states_thresholds.keys())[2][0] 
            
        self.train_data.reset_index(drop=True, inplace=True)
        self.train_data[WDAY] = self.train_data[DATE].dt.dayofyear
        
        columns_to_prev = list(filter(lambda item: item not in [DATE, WDAY, T_MIN, T_MAX] , self.train_data.columns.to_list()))
        for name in columns_to_prev:
            self.train_data[f'{name}_prev'] = self.train_data[f'{name}'].shift(1)

        return self.train_data, self.thresh