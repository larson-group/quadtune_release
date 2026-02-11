# User guide for tuning

This folder contains scripts that, given regional files as input,
optimize parameter values and create diagnostic plots. 
The scripts have been tested using Python 3.12.
The regional files must be created beforehand by running the scripts
in folder `create_regional_files`.

## Getting started


1) Make sure you installed all used python libraries from `requirements.txt` as described in the top-level `README.md`.

2) QuadTune has no namelist file, but 
you may copy `configs_tests/config_v1_paper_test.py` into `tuning_files` and specify the configuration of your tuning run, 
including the names of parameters to tune, the regional metrics to use, 
and the file names containing that information.
For more information on the configurable quantities, see the code comments in `configs_tests/config_v1_paper_test.py`.


3) Then run QuadTune with something like:  
 `python3 quadtune_driver.py --config_filename config_default.py`  
or  
 `python3 quadtune_driver.py -c config_example.py`  
and view the plots at http://127.0.0.1:8050/ in your web browser.

## Some setup variables in config_default.py

`folder_name`:  This designates the folder where the regional files are stored.  There are 2P regional .nc files plus a default file.
`obsPathAndFilename`: This designates the folder and name of the .nc file where observational data is recorded.

## Under the hood

QuadTune takes information from regional files in netcdf format.

Under the hood, `quadtune_driver.py` ingests information from `config.py`
with the help of functions in `set_up_inputs.py`.  Then it
optimizes parameter values, and calls `create_nonbootstrap_figs.py`.

`create_nonbootstrap_figs.py` generates figures and displays
figures on a plotly dash dashboard.

The other files, namely `create_bootstrap_figs.py` and `do_bootstrap_calcs.py`,
generate bootstrap samples from the regional metrics and create plots.
These scripts are experimental for now.
