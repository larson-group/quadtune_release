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
        1D array of formatted string names corresponding to each grid box(e.g., ['SWCF_1_1, ...]).
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
    Retrieves all regional NetCDF file paths matching the pattern "dataprefix_*_Regional.nc".

    Parameters
    ----------
    path : str or pathlib.Path
        Target directory containing the data files.
    dataprefix : str
        The prefix identifier used to filter the files.

    Returns
    -------
    list of str
        List of absolute file paths matching the pattern.
        
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
        dataset = xr.open_dataset(file,engine="engine")
        param_values = dataset[params_names].to_array().values
        params.append(param_values.flatten())

    return params


def calc_metric_sum_delta_from_data(data, default_data, global_averages):
    """
    Calculates the sum of squared normalized deviations between a dataset and a default.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of flattened spatial data.
    default_data : numpy.ndarray
        1D array of flattened default data.
    global_averages : numpy.ndarray
        1D array of reference averages used as weighting factors.

    Returns
    -------
    numpy.ndarray
        1D array of the summed squared deviations for each metric. The number of metrics gets derived from the length of 'global_averages'.
    """

    normalized_data = normalize_metrics_data(data, default_data, global_averages)
    
    num_avgs = len(global_averages)
    return np.sum(normalized_data.reshape(num_avgs, -1)**2, axis=1)

def calc_metric_sum_delta_from_file(file, default_data, global_averages, metric_names):
    """
    Reads a dataset from disk and computes its squared normalized deviations from a default.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to the NetCDF file to evaluate.
    default_data : numpy.ndarray
        1D array of flattened default data.
    global_averages : numpy.ndarray
        1D array of reference averages used as weighting factors.
    metric_names : list of str
        Specific variables to extract and evaluate from the file.

    Returns
    -------
    numpy.ndarray
        1D array of the summed squared deviations. The number of metrics gets derived from the length of 'global_averages'.
    """
    dataset = xr.open_dataset(file, engine='engine')

    metrics_data = dataset[metric_names].to_array().values.flatten()

    results = calc_metric_sum_delta_from_data(metrics_data, default_data,global_averages)
    dataset.close()

    return results


def get_correct_model_runs_delta(denormalized_params, all_verified_params, files, sst4k_files, default_file, sst4k_default_file, varNames, global_averages, box_size) :
    """
    Identifies specific parameter configurations within a larger dataset and extracts 
    their corresponding squared normalized deviations from a default .

    Parameters
    ----------
    denormalized_params : numpy.ndarray
        2D array of target parameter sets to locate.
    all_verified_params : numpy.ndarray
        2D array of all available parameter sets in the dataset.
    files : list of str
        List of standard simulation file paths.
    sst4k_files : list of str
        List of future (e.g., +4K) simulation file paths.
    default_file : str
        Path to the default standard simulation file.
    sst4k_default_file : str
        Path to the default +4K simulation file.
    varNames : list of str
        List of variable names to evaluate.
    global_averages : list of float
        Reference averages for weighting.
    box_size : int
        Grid resolution in degrees (assumes a 360x180 global grid).

    Returns
    -------
    numpy.ndarray or None
        A 3D array of shape (2, num_found_files, num_metrics) containing the metric 
        deltas for the standard and shifted simulations. Returns None if any target 
        parameter set is missing from the verified array.
    """
    idxs = [np.where(np.all(np.isclose(all_verified_params, paramset, atol = 1e-3, rtol = 1e-3),axis =1))[0] for paramset in denormalized_params]
    if np.any(np.array([len(idx) for idx in idxs]) < 1):
        return None
    

    metric_names = get_metrics_names(varNames, box_size)

    default_dataset = xr.open_dataset(default_file, engine="engine")[metric_names].to_array().values.flatten()
    default_dataset_sst4k = xr.open_dataset(sst4k_default_file, engine="engine")[metric_names].to_array().values.flatten()

    
    E3SM_metric_deltas_sst4k = np.array([calc_metric_sum_delta_from_file(file[0],default_dataset_sst4k,global_averages, metric_names) for file in sst4k_files[idxs]])
    E3SM_metric_deltas = np.array([calc_metric_sum_delta_from_file(file[0],default_dataset,global_averages, metric_names) for file in files[idxs]])


    e3sm_res_deltas = np.array([E3SM_metric_deltas, E3SM_metric_deltas_sst4k])

    return e3sm_res_deltas


def extract_field_matrices_from_whole(matrix, available_fields, used_fields, params_idxs, boxSize):
    """
    Extracts sub-matrices for specific climate fields from a stacked global sensitivity/curvature matrix.

    Assumes the input matrix is stacked vertically in the exact order of `available_fields`.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        2D array of shape (num_fields * num_boxes, num_parameters).
    available_fields : list or numpy.ndarray
        Sequence of strings representing the exact vertical ordering of the matrix blocks.
    used_fields : list of str
        Subset of fields to extract.
    params_idxs : list of int
        Column indices corresponding to the parameters to extract.
    box_size : int
        Grid resolution in degrees. Assumes a global 360x180 grid.

    Returns
    -------
    dict
        Dictionary mapping field names to their extracted 2D numpy.ndarray sub-matrices.
    """
    
    num_boxes = (360//boxSize) * (180//boxSize)

    all_matrices = {}

    for field in used_fields:
        idx = list(available_fields).index(field)

        field_matrix = matrix[num_boxes*idx:num_boxes*(idx+1), params_idxs]

        all_matrices[field] = field_matrix 

    return all_matrices