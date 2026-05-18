import numpy as np

import xarray as xr


from pathlib import Path

from optimization import normalize_metrics_data


def get_metrics_names(varPrefixes:list[str], boxSize:int):
    metricsNames = []

    for varPrefix in varPrefixes:
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")

    metricsNames = np.array(metricsNames)

    return metricsNames

def get_data_from_file(path, varPrefix, boxSize=15):

    metrics_names = get_metrics_names(varPrefix,boxSize)

    dataset = xr.open_dataset(path, engine='netcdf4')

    return dataset[metrics_names].to_array().values.flatten()


def get_filenames(path, dataprefix):
    folder = Path(path)
    
    if not folder.is_dir():
        raise ValueError(f"The provided path '{path}' is not a valid directory.")
    

    pattern= f"{dataprefix}_*_Regional.nc"

    files = [str(file) for file in folder.glob(pattern) if file.is_file()]

    return files


def get_params_from_files(files, params_names):
    params = []

    for file in files:
        dataset = xr.open_dataset(file,engine="netcdf4")
        param_values = dataset[params_names].to_array().values
        params.append(param_values.flatten())

    return params


def calc_metric_sum_delta_from_data(data, default_data, global_averages):

    normalized_data = normalize_metrics_data(data, default_data, global_averages)
    
    num_avgs = len(global_averages)
    return np.sum(normalized_data.reshape(num_avgs, -1)**2, axis=1)

def calc_metric_sum_delta_from_file(file, default_data, global_averages, metric_names):
    dataset = xr.open_dataset(file, engine='netcdf4')

    metrics_data = dataset[metric_names].to_array().values.flatten()

    results = calc_metric_sum_delta_from_data(metrics_data, default_data,global_averages)
    dataset.close()

    return results


def get_correct_model_runs_delta(denormalized_params, all_E3SM_params, files, sst4k_files, default_file, sst4k_default_file, varNames, global_averages, box_size):

    idxs = [np.where(np.all(np.isclose(all_E3SM_params, paramset, atol = 1e-3, rtol = 1e-3),axis =1))[0] for paramset in denormalized_params]
    if np.any(np.array([len(idx) for idx in idxs]) < 1):
        return None
    

    metric_names = get_metrics_names(varNames, box_size)

    default_dataset = xr.open_dataset(default_file, engine="netcdf4")[metric_names].to_array().values.flatten()
    default_dataset_sst4k = xr.open_dataset(sst4k_default_file, engine="netcdf4")[metric_names].to_array().values.flatten()

    
    E3SM_metric_deltas_sst4k = np.array([calc_metric_sum_delta_from_file(file[0],default_dataset_sst4k,global_averages, metric_names) for file in sst4k_files[idxs]])
    E3SM_metric_deltas = np.array([calc_metric_sum_delta_from_file(file[0],default_dataset,global_averages, metric_names) for file in files[idxs]])


    e3sm_res_deltas = np.array([E3SM_metric_deltas, E3SM_metric_deltas_sst4k])

    return e3sm_res_deltas


def extract_field_matrices_from_whole(matrix,available_fields, used_fields, params_idxs, boxSize):
    
    num_boxes = (360//boxSize) * (180//boxSize)

    all_matrices = {}

    for field in used_fields:
        idx = list(available_fields).index(field)

        field_matrix = matrix[num_boxes*idx:num_boxes*(idx+1), params_idxs]

        all_matrices[field] = field_matrix 

    return all_matrices