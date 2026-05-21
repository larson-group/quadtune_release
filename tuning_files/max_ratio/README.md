# Effectiveness of observable constraints on the worst-case uncertainty of a global atmospheric model

This folder contains all code and data necessary to reproduce the base results of the manuscript.

To run the analysis and create plots run the following steps:

0) Create an empty virtual environment via `conda` or `venv`
1) Navigate to the main Quadtune directory (one directory up) and run `pip install -r requirements.txt`
2) Return to the directory, where this README resides, and run `python create_paper_plots.py -d ./data --ppe_data ./PPE_data --ppe_data_sst4k ./PPE_data_sst4k -o ./output_dir  --constr_opt`

The code is split into 4 main files:
1) `create_paper_plots.py`: Contains the configuration and main function to execute the optimizations and plotting routines.
2) `optimization.py`: Core functions used for the Basin-hopping optimization and ratio evaluation.
3) `data_io.py`: Handles reading and processing of E3SM NetCDF files and the calculation of the field specific L2^2 perturbation from the default.
4) `plotting.py`: Contains the functions to create all presented plots.

To reproduce the presented results, no changes to any files should be necessary.