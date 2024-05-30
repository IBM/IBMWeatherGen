import numpy as np
import datetime
from IBMWeatherGen import IBMWeatherGen 
import json 
import pandas as pd

DEFAULT_WET_EXTREME_THRESHOLD = 0.999

if __name__ == "__main__":

    with open('ibmwg-input.json', 'r') as jsn:
        config = json.load(jsn)
    
    path_in = str(config["PATH_FILE_IN"])
    path_out = str(config["PATH_FILE_OUT"])
    wet_extreme_quantile_threshold = float(config["WET_EXTREME"])
    start_year = int(config["START_YEAR"])
    end_year = start_year + int(config["NUM_YEARS"])
    number_of_simulations = int(config["NUM_SIMULATIONS"])
    use_g2s = bool(config.get("USE_G2S", True))  # Default to True if not specified

    if wet_extreme_quantile_threshold != 0:
        wet_extreme_quantile_threshold = float(config["WET_EXTREME"])
    else:
        wet_extreme_quantile_threshold = DEFAULT_WET_EXTREME_THRESHOLD
    
    _st = datetime.datetime.now()
    
    wg_weather = IBMWeatherGen(file_in_path=path_in,
                               years=list(np.arange(start_year, end_year)), 
                               nsimulations=number_of_simulations,
                               wet_extreme_quantile_threshold=wet_extreme_quantile_threshold,
                               use_g2s=use_g2s)

    df = wg_weather.generate_weather_series()

    #############################
    # METRICS: — modifying Jorge
    ############################
    df_historic = pd.read_csv(path_in, parse_dates=['Date'])
    print(df.columns)
    print(df.head())

    marker = datetime.datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    file_name = "{}ibmwg-simulations_{}.csv.zip".format(path_out, marker)
    print('\nSaving .zip file.')
    df.to_csv(file_name, index=True, compression="zip")
    print(f'\nOUTPUT PATH: {file_name}\n')
    
    print("(Total simulation time: {})\n\n".format(datetime.datetime.now() - _st))
    jsn.close()