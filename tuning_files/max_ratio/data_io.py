
from concurrent.futures import ProcessPoolExecutor

from scipy.linalg import block_diag

import numpy as np

import xarray as xr


from pathlib import Path

from optimization import normalize_metrics_data


import sys
engine = "h5netcdf" if sys.platform == "win32" else "netcdf4"

def get_metrics_names(varPrefixes, boxSize):
    """
    Generates a list of regional metric names based on a specified grid resolution.

    Parameters
    ----------
    varPrefixes : list of str
        List of variable base names (e.g., ['SWCF', 'PRECT']).
    boxSize : int
        Grid resolution in degrees (assumes a global 360x180 grid).

    Returns
    -------
    numpy.ndarray
        1D array of formatted string names corresponding to each grid box 
        (e.g., ['SWCF_1_1', ...]).
    """
    metricsNames = []

    for varPrefix in varPrefixes:
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")

    metricsNames = np.array(metricsNames)

    return metricsNames


def get_filenames(path, dataprefix):
    """
    Retrieves all regional NetCDF file paths matching the pattern 
    "<dataprefix>_*_Regional.nc".

    Parameters
    ----------
    path : str or pathlib.Path
        Target directory containing the data files.
    dataprefix : str
        The prefix identifier used to filter the files.

    Returns
    -------
    list of str
        List of file paths matching the pattern.

    Raises
    ------
    ValueError
        If the provided path is not a valid directory.
    """
    folder = Path(path)
    
    if not folder.is_dir():
        raise ValueError(f"The provided path '{path}' is not a valid directory.")
    

    pattern= f"{dataprefix}_*_Regional.nc"

    files = [str(file) for file in folder.glob(pattern) if file.is_file()]

    return files


def _flatten_dataset_variables(dataset, names):
    """
    Extracts and flattens a set of dataset variables into a single 1D NumPy array.

    Parameters
    ----------
    dataset : xarray.Dataset
        The opened NetCDF dataset.
    names : sequence of str
        Names of the variables to extract and flatten.

    Returns
    -------
    numpy.ndarray
        1D array of concatenated variable values.
    """
    return np.concatenate([np.ravel(dataset[name].values) for name in names])


def get_params_from_files(files, params_names):
    """
    Extracts specified parameter values across multiple NetCDF datasets.

    Parameters
    ----------
    files : list of str
        List of file paths to process.
    params_names : list of str
        List of parameter names to extract from each dataset.

    Returns
    -------
    list of numpy.ndarray
        List containing 1D arrays of parameter values for each file.
    """
    params = []

    for file in files:
        with xr.open_dataset(file, engine=engine) as dataset:
            params.append(_flatten_dataset_variables(dataset, params_names))

    return params

def _read_params_and_metrics_from_file(file, params_names, metric_names):
    """
    Reads both parameter and regional metric values from a single NetCDF file.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the NetCDF file.
    params_names : sequence of str
        Names of the model parameters to extract.
    metric_names : sequence of str
        Names of the regional metrics to extract.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        A tuple containing the 1D array of parameter values and the 1D array 
        of metric values.
    """
    with xr.open_dataset(file, engine=engine) as dataset:
        param_values = _flatten_dataset_variables(dataset, params_names)
        metric_values = _flatten_dataset_variables(dataset, metric_names)

    return param_values, metric_values


def get_params_and_metrics_from_files(files, params_names, varNames, boxSize):
    """
    Extracts both parameter values and regional metrics from multiple NetCDF datasets.

    Parameters
    ----------
    files : list of str
        List of file paths to process.
    params_names : list of str
        List of parameter names to extract from each dataset.
    varNames : list of str
        List of variable base names for metrics extraction.
    boxSize : int
        Grid resolution in degrees (assumes a global 360x180 grid).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        A tuple containing:
        - 2D array of parameter values of shape (num_files, num_params).
        - 2D array of regional metrics of shape (num_files, num_metrics).
    """
    metric_names = get_metrics_names(varNames, boxSize)

    if not files:
        return [], []

    if len(files) == 1:
        results = [_read_params_and_metrics_from_file(files[0], params_names, metric_names)]
    else:
        max_workers = max(1, min(8, len(files)))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                _read_params_and_metrics_from_file,
                files,
                [params_names] * len(files),
                [metric_names] * len(files)
            ))

    params = np.array([result[0] for result in results])
    metrics = np.array([result[1] for result in results])

    return params, metrics

def calc_metric_sum_delta_from_data(data, default_data, global_averages):
    """
    Calculates the normalized regional deviations between a dataset and a 
    default simulation, reshaped by field.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of flattened spatial data across all evaluated fields.
    default_data : numpy.ndarray
        1D array of flattened default simulation data.
    global_averages : numpy.ndarray
        1D array of reference global averages used as normalization factors.

    Returns
    -------
    numpy.ndarray
        2D array of shape (num_fields, num_boxes) containing the normalized 
        deviations from the default simulation for each metric.
    """

    normalized_data = normalize_metrics_data(data, default_data, global_averages)
    
    num_avgs = len(global_averages)
    return normalized_data.reshape(num_avgs, -1)

def calc_metric_sum_delta_from_file(file, default_data, global_averages, metric_names):
    """
    Reads a NetCDF dataset from disk and computes its normalized regional 
    deviations from a default simulation.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the NetCDF file to evaluate.
    default_data : numpy.ndarray
        1D array of flattened default simulation data.
    global_averages : numpy.ndarray
        1D array of reference global averages used as normalization factors.
    metric_names : sequence of str
        Specific regional variable names to extract and evaluate from the file.

    Returns
    -------
    numpy.ndarray
        2D array of shape (num_fields, num_boxes) containing the normalized 
        regional deviations.
    """
    with xr.open_dataset(file, engine=engine) as dataset:
        metrics_data = _flatten_dataset_variables(dataset, metric_names)

    results = calc_metric_sum_delta_from_data(metrics_data, default_data, global_averages)
    print(f"Processed file: {file}")
    return results

def flatten_results(nested_dict):
    """
    Flattens the nested optimization result dictionary for automated testing.

    Parameters
    ----------
    nested_dict : dict
        Nested dictionary mapping variable names and scenario identifiers to 
        tuples of (parameter_vector, objective_scalar).

    Returns
    -------
    dict
        Flattened dictionary with keys formatted as "<var>/<scenario>/vector" 
        and "<var>/<scenario>/scalar".
    """
    flat_dict = {}
    for var_name, scenarios in nested_dict.items():
        for scenario_name, (vector, scalar) in scenarios.items():
            base_key = f"{var_name}/{scenario_name}"
            
            flat_dict[f"{base_key}/vector"] = vector
            flat_dict[f"{base_key}/scalar"] = np.array(scalar) 
            
    return flat_dict


def get_correct_model_runs_delta(denormalized_params, all_verified_params, files, sst4k_files, default_file, sst4k_default_file, varNames, global_averages, box_size) :
    """
    Identifies specific parameter configurations within a set of E3SM verification 
    runs and extracts their normalized regional deviations from the default run.

    Parameters
    ----------
    denormalized_params : sequence of numpy.ndarray
        2D array or list of target parameter sets in physical units to locate.
    all_verified_params : numpy.ndarray
        2D array of all available parameter sets in the verification dataset.
    files : numpy.ndarray of str
        Array of present-day simulation file paths.
    sst4k_files : numpy.ndarray of str
        Array of future (+4K SST) simulation file paths.
    default_file : str
        Path to the default present-day simulation file.
    sst4k_default_file : str
        Path to the default +4K simulation file.
    varNames : sequence of str
        List of variable names to evaluate.
    global_averages : numpy.ndarray
        Reference global averages used for normalization.
    box_size : int
        Grid resolution in degrees (assumes a 360x180 global grid).

    Returns
    -------
    numpy.ndarray or None
        A 4D array of shape (2, num_found_files, num_fields, num_boxes) containing 
        the normalized regional deltas for present-day and +4K simulations. 
        Returns None if any target parameter set is missing from the dataset.
    """
    idxs = [np.where(np.all(np.isclose(all_verified_params, paramset, atol = 1e-3, rtol = 1e-3),axis =1))[0] for paramset in denormalized_params]
    if np.any(np.array([len(idx) for idx in idxs]) < 1):
        return None
    

    metric_names = get_metrics_names(varNames, box_size)

    with xr.open_dataset(default_file, engine=engine) as default_ds:
        default_dataset = _flatten_dataset_variables(default_ds, metric_names)
    with xr.open_dataset(sst4k_default_file, engine=engine) as default_ds_sst4k:
        default_dataset_sst4k = _flatten_dataset_variables(default_ds_sst4k, metric_names)

    E3SM_metric_deltas_sst4k = np.array([
        calc_metric_sum_delta_from_file(file[0], default_dataset_sst4k, global_averages, metric_names)
        for file in sst4k_files[idxs]
    ])
    E3SM_metric_deltas = np.array([
        calc_metric_sum_delta_from_file(file[0], default_dataset, global_averages, metric_names)
        for file in files[idxs]
    ])


    e3sm_res_deltas = np.array([E3SM_metric_deltas, E3SM_metric_deltas_sst4k])

    return e3sm_res_deltas


def extract_field_matrices_from_whole(matrix, available_fields, used_fields, params_idxs, boxSize):
    """
    Extracts sub-matrices for specific climate fields from a vertically stacked 
    global sensitivity or curvature matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        2D array of shape (num_available_fields * num_boxes, num_all_parameters).
    available_fields : sequence of str
        Sequence of strings representing the exact vertical ordering of the 
        matrix blocks.
    used_fields : list of str
        Subset of fields to extract.
    params_idxs : list of int
        Column indices corresponding to the parameters to extract.
    boxSize : int
        Grid resolution in degrees (assumes a global 360x180 grid).

    Returns
    -------
    dict of {str: numpy.ndarray}
        Dictionary mapping field names to their extracted 2D sub-matrices of 
        shape (num_boxes, num_used_parameters).
    """
    
    num_boxes = (360//boxSize) * (180//boxSize)

    all_matrices = {}

    for field in used_fields:
        idx = list(available_fields).index(field)

        field_matrix = matrix[num_boxes*idx:num_boxes*(idx+1), params_idxs]

        all_matrices[field] = field_matrix 

    return all_matrices


def get_inv_gamma_for_fields(gamma_joint, cov_variables, field_names, box_size, variance_threshold=0.99):
    """
    Computes the trace-normalized, Tikhonov-regularized inverse noise covariance 
    matrix for one or multiple observable fields.

    For multiple fields, individual N x N spatial covariance blocks are inverted 
    and trace-normalized separately before being combined into a block-diagonal 
    matrix, neglecting cross-field covariances.

    Parameters
    ----------
    gamma_joint : numpy.ndarray
        2D joint covariance matrix of shape (num_cov_vars * num_boxes, num_cov_vars * num_boxes).
    cov_variables : list of str
        Ordered list of variable names represented in `gamma_joint`.
    field_names : list of str
        Names of the fields for which to construct the inverse covariance matrix.
    box_size : int
        Grid resolution in degrees (assumes a global 360x180 grid).
    variance_threshold : float, optional
        Cumulative variance threshold used to dynamically select the singular 
        value for Tikhonov regularization (default is 0.99).

    Returns
    -------
    numpy.ndarray or None
        2D regularized and trace-normalized inverse covariance matrix of shape 
        (len(field_names) * num_boxes, len(field_names) * num_boxes), or None 
        if `field_names` is empty.

    Raises
    ------
    ValueError
        If a requested field name is not present in `cov_variables`.
    """
    if not field_names:
        return None


    if len(field_names) > 1:
        blocks = [
            get_inv_gamma_for_fields(gamma_joint, cov_variables, [f], box_size, variance_threshold)
            for f in field_names
        ]
        return block_diag(*blocks)

    f = field_names[0]
    if f not in cov_variables:
        raise ValueError(f"field {f} not found in covariance data.")

    num_boxes = (360 // box_size) * (180 // box_size)
    idx = list(cov_variables).index(f)
    indices = list(range(idx * num_boxes, (idx + 1) * num_boxes))

    gamma_sliced = gamma_joint[np.ix_(indices, indices)]

    U, s, Vt = np.linalg.svd(gamma_sliced)
    total_var = np.sum(s)
    cum_var = np.cumsum(s) / total_var

    n_components = int(np.searchsorted(cum_var, variance_threshold) + 1)
    n_components = min(n_components, len(s))

    raw_floor = s[n_components - 1] if n_components <= len(s) else s[-1]
    min_safe_floor = 1e-2 * (total_var / len(s))
    lambda_sq = max(raw_floor, min_safe_floor)

    s_reg = s + lambda_sq

    gamma_inv = (U / s_reg) @ Vt


    gamma_inv = gamma_inv * (num_boxes / np.trace(gamma_inv))

    return gamma_inv
