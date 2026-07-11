"""
This test is made to be run using pytest.

It computes Sensitivity and Curvature matrices from PPE data and checks if they match the sensitivity and curvature matrices
calculated from the regional files by fitting a quadratic polynomial.

"""

import os
import sys
"""
The following line adds the parent directory (quadtune/) to the python path, so that imports from directory tuning_files work.
The alternative would be to make quadtune a package using __init__.py files, but that would require changing
all import statements in other files.
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tuning_files import quadtune_driver
import numpy as np

def test_fit_quad_emulator_to_ppe_test() -> None:

    polyfit_paramsNames_order =  ['clubb_c8','clubb_c_invrs_tau_n2','clubb_c_invrs_tau_sfc','clubb_c_invrs_tau_wpxp_n2_thresh','clubb_c_invrs_tau_n2_wp2']

    thisfile_abspath = os.path.dirname(os.path.abspath(__file__))

    quadtune_results = quadtune_driver.main(["-c", "configs_tests.config_ppe_test","--ppe"])

    # These are the sensitivity and curvature matrices calculated by solving the ppe matrix equation
    ppe_sens_matrix = quadtune_results[2]
    ppe_curv_matrix = quadtune_results[3]

    # These are the sensitivity and curvature matrices calculated from the v1 paper regional files using quadratic polynomial fitting
    normlzdSensMatrixPoly = np.load(thisfile_abspath + "/files_for_ppe_test/normlzdSensMatrixPoly.npy")
    normlzdCurvMatrix = np.load(thisfile_abspath + "/files_for_ppe_test/normlzdCurvMatrix.npy")

    paramsNames = quadtune_results[1]
    polyfit_indices = [list(paramsNames).index(name) for name in polyfit_paramsNames_order]

    ppe_sens_matrix = ppe_sens_matrix[:,polyfit_indices]
    ppe_curv_matrix = ppe_curv_matrix[:,polyfit_indices]



    assert np.allclose(ppe_sens_matrix, normlzdSensMatrixPoly)
    assert np.allclose(ppe_curv_matrix, normlzdCurvMatrix)
    
    