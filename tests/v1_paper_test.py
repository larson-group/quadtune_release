"""
This test is made to be run using pytest to verify, that the results of the v1 paper are reproduceable
using the current version of quadtune.
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

def test_v1_paper_tuning() -> None:
    
    quadtune_v1_test_parameter_values, _ = quadtune_driver.main(["-c", "configs_tests.config_v1_paper_test"])
    true_parameter_values = [0.7970267, 0.4655446, 0.0880349, 0.0004552707, 0.1357132]
    assert np.allclose(quadtune_v1_test_parameter_values.flatten(), true_parameter_values, rtol=1e-5)

if __name__ == "__main__":
    test_v1_paper_tuning()