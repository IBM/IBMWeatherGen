import pandas as pd
import numpy as np
from pomegranate import MarkovChain, DiscreteDistribution, ConditionalProbabilityTable
from typing import Dict, Any, Optional
from constants_ import DATE, PRECIPITATION, SAMPLE_DATE, STATE, STATE_PREV, WDAY

MC_PER_YEAR = 12


def find_thresholds(data: pd.Series, dry_wet_ratio=0.3, wet_extreme_ratio=0.1):
    """
    Automatically set thresholds for dry-wet and wet-extreme states.

    Parameters:
    data: pd.Series - Historical precipitation data
    dry_wet_ratio: float - Percentage for dry state
    wet_extreme_ratio: float - Percentage for extreme wet state

    Returns:
    dry_threshold: float - Threshold for dry state
    wet_threshold: float - Threshold for extreme wet state
    """
    dry_threshold = np.percentile(data, dry_wet_ratio * 100)
    wet_threshold = np.percentile(data, (1 - wet_extreme_ratio) * 100)
    return dry_threshold, wet_threshold

def label_states(data: pd.DataFrame, dry_threshold: float, wet_threshold: float):
    """
    Label data based on thresholds.

    Parameters:
    data: pd.DataFrame - Historical weather data
    dry_threshold: float - Threshold for dry state
    wet_threshold: float - Threshold for extreme wet state

    Returns:
    data: pd.DataFrame - Data with state labels
    """
    conditions = [
        (data[PRECIPITATION] < dry_threshold),
        (data[PRECIPITATION] >= dry_threshold) & (data[PRECIPITATION] < wet_threshold),
        (data[PRECIPITATION] >= wet_threshold)
    ]
    choices = ['d', 'w', 'e']
    data[STATE] = np.select(conditions, choices)
    return data

def compute_extreme_stats(data: pd.DataFrame, state: str):
    """
    Compute statistics for the length and frequency of extreme states.

    Parameters:
    data: pd.DataFrame - Historical weather data
    state: str - State to compute stats for ('d', 'w', or 'e')

    Returns:
    mean_length: float - Mean length of the state
    frequency: float - Frequency of the state
    """
    state_lengths = (data[STATE] == state).astype(int).groupby(data[STATE].ne(data[STATE].shift()).cumsum()).sum()
    mean_length = state_lengths[state_lengths > 0].mean()
    frequency = state_lengths[state_lengths > 0].count() / len(data)
    return mean_length, frequency


class FirstOrderMarkovChain:
    """
    Computes the transition matrix and the state probabilities for each month, building the first order Markov Chain parameters.
    Besides, will also generate a state sequence for each month, creating the structure - on daily basis - of the timeseries being simulated for
    the year inputed by the user.

    Args
    ----------
    training_data: Dataframe sampled from the historic data, with the with the days labeled as 'd' (dry), 'w' (wet) or 'e' (extreme).

    simulation_year: Int year choosed by the user to be be simulated.

    weather_variables: Optional list with the names of the weather variables being simulated (should include 'precipitation')

    Properties
    ----------
    transition_matrix : list[list]
        Transition matrix for the three states being considereted.

    transition_prob : list[dict]
        Probabilites of each of the three states.

    training_data : pd.DataFrame
        Sample data labeled from the historic data.

    simulation_year: int
        The year selected to be simulated.

    columns_names: list[str]
        The columns names which the structure of the new timeseries will have.
        
    dry_wet_ratio: float
        Percentage for dry state
    
    wet_extreme_ratio: float
        Percentage for extreme wet state

    """
        
    def __init__(self,
                 training_data: pd.DataFrame = None,
                 simulation_year: int = 2000,
                 weather_variables: Optional[list] = [PRECIPITATION],
                 dry_wet_ratio=0.3, wet_extreme_ratio=0.1,
                 length_dry: Optional[float] = None, length_wet: Optional[float] = None, length_extreme: Optional[float] = None,
                 freq_dry: Optional[float] = None, freq_wet: Optional[float] = None, freq_extreme: Optional[float] = None) -> None:
        self.transition_matrix = list()
        self.transition_prob = list()
        self.training_data = training_data
        self.simulation_year = simulation_year
        self.columns_names = [SAMPLE_DATE, STATE, STATE_PREV]
        self.columns_names.extend(weather_variables)
        self.dry_wet_ratio = dry_wet_ratio
        self.wet_extreme_ratio = wet_extreme_ratio
        self.length_dry = length_dry
        self.length_wet = length_wet
        self.length_extreme = length_extreme
        self.freq_dry = freq_dry
        self.freq_wet = freq_wet
        self.freq_extreme = freq_extreme

    def create_dataframe_structure(self) -> pd.DataFrame:
        """ Build a DataFrame for the new timeseries being simulated.

            Returns
            ----------
                A DataFrame with the days, days of the year already filled and the other columns as np.nan

        """

        dates = list(pd.date_range(start=pd.Timestamp(self.simulation_year, 1, 1),
                                   end=pd.Timestamp(self.simulation_year, 12, 31), freq='D'))

        wday = [date.dayofyear for date in dates]

        df = pd.concat([pd.DataFrame({DATE: dates, WDAY: wday}),
                        pd.DataFrame(np.nan, index=np.arange(0, len(dates)), columns=self.columns_names)], axis=1)

        return df

    def estimate_markov_chains(self, df_month: pd.DataFrame) -> Dict[str, Dict]:
        """ Computer the first order Markov Chain transition matrix and the probabilities of each state within a month.

            Parameters
            ----------
            df_month : pd.DataFrame
                Selected month labeled from the observed data.

            Returns
            ----------
                A dict with the first order Markov Chain parameters.
        """

        # marginal probs
        states = list(df_month['state'])
        mc_prob = MarkovChain.from_samples(states)
        self.transition_prob = mc_prob.distributions[0].parameters

        # transition matrix
        states.append(states[0:])
        mc_tr = MarkovChain.from_samples(states)
        self.transition_matrix = [[item[0], item[1], float(item[2])] for item in mc_tr.distributions[1].parameters[0]]

        return {'weather_probs': self.transition_prob, 'transition_matrix': self.transition_matrix}
    
    
    def adjust_markov_chain(self):
        """
        Adjust Markov Chain parameters based on user-defined lengths and frequencies of states.
        """
        if self.length_dry or self.freq_dry:
            mean_length, frequency = compute_extreme_stats(self.training_data, 'd')
            if self.length_dry:
                adjustment_ratio = self.length_dry / mean_length
                self.transition_matrix = [
                    [i, j, p * adjustment_ratio if i == 'd' and j == 'd' else p / adjustment_ratio]
                    for i, j, p in self.transition_matrix
                ]
            if self.freq_dry:
                adjustment_ratio = self.freq_dry / frequency
                self.transition_prob = [
                    {k: v * adjustment_ratio if k == 'd' else v / adjustment_ratio for k, v in dist.items()}
                    for dist in self.transition_prob
                ]

        if self.length_wet or self.freq_wet:
            mean_length, frequency = compute_extreme_stats(self.training_data, 'w')
            if self.length_wet:
                adjustment_ratio = self.length_wet / mean_length
                self.transition_matrix = [
                    [i, j, p * adjustment_ratio if i == 'w' and j == 'w' else p / adjustment_ratio]
                    for i, j, p in self.transition_matrix
                ]
            if self.freq_wet:
                adjustment_ratio = self.freq_wet / frequency
                self.transition_prob = [
                    {k: v * adjustment_ratio if k == 'w' else v / adjustment_ratio for k, v in dist.items()}
                    for dist in self.transition_prob
                ]

        if self.length_extreme or self.freq_extreme:
            mean_length, frequency = compute_extreme_stats(self.training_data, 'e')
            if self.length_extreme:
                adjustment_ratio = self.length_extreme / mean_length
                self.transition_matrix = [
                    [i, j, p * adjustment_ratio if i == 'e' and j == 'e' else p / adjustment_ratio]
                    for i, j, p in self.transition_matrix
                ]
            if self.freq_extreme:
                adjustment_ratio = self.freq_extreme / frequency
                self.transition_prob = [
                    {k: v * adjustment_ratio if k == 'e' else v / adjustment_ratio for k, v in dist.items()}
                    for dist in self.transition_prob
                ]
                
    def simulate_state_sequence(self) -> Any:
        """ Generate the state sequences for each period (monthly)

            Returns
            ----------
                A DataFrame with the structure needed to build the new timeseries for the year being simulated and a list with
                the Markov Chain parameters for the year being simulated.
        """

        dfsimu = self.create_dataframe_structure()

        seq_monthly = list()
        mchain = list()
        for month in range(1, MC_PER_YEAR + 1, 1):
            df_month = self.training_data[self.training_data[DATE].dt.month == month]

            markov_models_parameters = self.estimate_markov_chains(df_month)
            self.adjust_markov_chain()
            mchain.append(markov_models_parameters)

            d1 = DiscreteDistribution(markov_models_parameters['weather_probs'][0])
            d2 = ConditionalProbabilityTable(markov_models_parameters['transition_matrix'], [d1])

            mc = MarkovChain([d1, d2])
            seq = mc.sample(len(dfsimu[dfsimu[DATE].dt.month == month]))

            seq_monthly.extend(seq)

        dfsimu[STATE] = seq_monthly
        dfsimu[STATE_PREV] = dfsimu[STATE].shift(1)

        return dfsimu, mchain 