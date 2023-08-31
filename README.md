# IBMWeatherGen
IBMWeatherGen tool is a multisite semi-parametric precipitation generator, which generates simulation of weather variables (e.g., precipitation) for a specified region of interest. 

## Setup 
Pre-requisite:
* Install python3 (3.8.14 is the recommended version)


Steps:
1. clone the repository (`git clone git@github.ibm.com:2944/wg.git`)
1. create virtual-env and activate it. You can use conda, pyenv, virtualenv, etc. E.g.:
    - `conda create -n "wg-env" python=3.8.14`
    - `conda activate wg-env`
1. go to `wg` directory and install dependencies: `pip install -r requirements.txt`
1. edit [input parameter file](./src/cli/input_params_example.json) (a JSON file) or create a new one
1. go to `wg/src` directory
1. run `python cli/simulation_cli.py --configpath=<path-to-json-file>`
1. simulations are generated and stored into `<PERSIST_DATA_DIR>/simulated_data.zip`  
