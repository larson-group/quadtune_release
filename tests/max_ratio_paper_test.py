import numpy as np
import numpy.testing as npt

from pathlib import Path
import os
import sys
"""
The following line adds the parent directory (quadtune/) to the python path, so that imports from directory tuning_files work.
The alternative would be to make quadtune a package using __init__.py files, but that would require changing
all import statements in other files.
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tuning_files/max_ratio')))
from tuning_files.max_ratio import create_paper_plots

def test_simulation_results():

    result_path = Path(__file__).parent / "files_for_max_ratio_paper/reference_optimization_results.npz"

    data_path = str(Path(__file__).parent.parent / "tuning_files/max_ratio/")

    expected_flat = np.load(result_path)
    
    flattened_result =create_paper_plots.main(["-d",data_path+"/data","--ppe_data",data_path+"/PPE_data","--ppe_data_sst4k",data_path+"/PPE_data_sst4k","-o", data_path+"/output_dir", "--constr_opt", "--testing"]) 

    
    expected_keys = set(expected_flat.files)
    actual_keys = set(flattened_result.keys())
    assert expected_keys == actual_keys, "The structure of the output dictionary has changed"
    

    for key in expected_keys:
        npt.assert_allclose(
            flattened_result[key], 
            expected_flat[key], 
            atol=1e-3,
            rtol=1e-3,
            err_msg=f"Different result for {key}"
        )