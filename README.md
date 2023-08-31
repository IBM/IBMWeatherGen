# IBMWeatherGen: Stochastic Weather Generator Tool

Welcome to IBMWeatherGen, a powerful gridded, multisite, multivariate, and daily stochastic weather generator based on resampling methodology.

## Setup

To get started, follow these steps:

1. **Clone the Repository**: Begin by cloning this repository to your local machine.
   
2. **Create and Activate Virtual Environment**: Set up a virtual environment using your preferred tool (conda, pyenv, virtualenv, etc.). Here's an example using conda:
   
    ```
    conda create -n wg-env python=3.8.14
    conda activate wg-env
    ```

3. **Install Dependencies**: Install the required dependencies by executing the following command:
   
    ```
    pip install -r requirements.txt
    ```

## Running Tests

Follow these steps to run tests:

1. **Unzip the Dataset**: Unzip the provided dataset named `dset_wg_d.csv.zip` and place it in the `/data` directory.

2. **Navigate to Source Directory**: Move to the `src` directory in the terminal.

3. **Run Simulation**: Execute the simulation by running the following command:
   
    ```
    python execute_IBMWeatherGen_json.py
    ```

This will generate simulations stored in the `simulation` folder.

## Customization

To customize the weather generator according to your needs:

1. **Weather Dataset**: Prepare a weather dataset in the same format as the provided dataset (`dset_wg_d.csv`).

2. **Edit Input Parameter File**: Modify the [input parameter file](./src/ibmwg-input.json) with your desired settings:

   ```json
   {
       "PATH_FILE_IN": "../data/dset_wg_d.csv",
       "PATH_FILE_OUT": "../simulations/",
       "START_YEAR": 2010,
       "NUM_YEARS": 10,
       "NUM_SIMULATIONS": 5,
       "WET_EXTREME": 0.999
   }

Where:

- `PATH_FILE_IN`: Path to the weather data used for training the generator.
- `PATH_FILE_OUT`: Path to store generated simulations.
- `START_YEAR`: Starting year for simulations.
- `NUM_YEARS`: Number of years to simulate ahead.
- `NUM_SIMULATIONS`: Number of simulation sets.
- `WET_EXTREME`: Threshold for extreme events.

## Authors

- Maria Julia de Castro Villafranca Garcia

- Jorge Luis Guevara Diaz
  - GitHub: [jorjasso](https://github.com/jorjasso)
  - Email: jorgegd@br.ibm.com

- Leonardo Tizzei
  - GitHub: [ltizzei](https://github.com/ltizzei)
  - Email: ltizzei@br.ibm.com

## References

Kindly contemplate referencing this works in case you make use of the weather generator:

- Jorge Luis Guevara Diaz, Maria Garcia, Priscilla Avegliano, et al. *Direct Sampling for Extreme Events Generation and Spatial Variability Enhancement of Weather Generators*. Authorea. December 02, 2022.

- Steinschneider, Scott et al. "A semiparametric multivariate, multisite weather generator with low-frequency variability for use in climate risk assessments." DOI: [10.1002/wrcr.20528](https://doi.org/10.1002/wrcr.20528) (2013).

Enhance your understanding with these references:

- Apipattanavis, Somkiat et al. "A semiparametric multivariate and multisite weather generator." DOI: [10.1029/2006WR005714](https://doi.org/10.1029/2006WR005714) (2007).

- Kwon, Hyun-Han et al. "Stochastic simulation model for nonstationary time series using an autoregressive wavelet decomposition: Applications to rainfall and temperature." DOI: [10.1029/2006WR005258](https://doi.org/10.1029/2006WR005258) (2006).

- Rajagopalan, Balaji et al. "A k-nearest-neighbor simulator for daily precipitation and other weather variables." DOI: [10.1029/1999WR900028](https://doi.org/10.1029/1999WR900028) (1999).

- WeatherGen: [walkerjeffd.github.io/weathergen/](https://walkerjeffd.github.io/weathergen/)
