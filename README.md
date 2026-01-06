# QuadTune
A tuner for global atmospheric models with a quadratic emulator.

This repo contains python source code in three folders: 
1) `create_regional_files`,
2) `tuning_files` and
3) `tests`.

To use any of the scripts you should install the necessary python modules by  
typing  `pip install -r requirements.txt`  in your command line.

The scripts in `create_regional_files` take output from 
global atmospheric simulations and create regional files.
See `create_regional_files/README_regional.md` for a
user guide for creating regional files.

The scripts in `tuning_files` take the regional files,
optimize parameter values, and display diagnostic plots
on a dashboard created with plotly dash.
See `tuning_files/README_tuning.md` for a user guide for tuning.

The scripts in `tests` are pytest tests which test the functionality of quadtune (https://docs.pytest.org).  
(You may need to initialize git lfs for your user account:  `git lfs install`.)  
Then to run all tests, go to the `quadtune` directory and execute the command `pytest`.  
To run specific tests, also go to the `quadtune` directory and execute the command `pytest -k ' {pattern found within one or more testfile names e.g. "v1" } `
