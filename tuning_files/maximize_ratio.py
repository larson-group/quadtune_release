import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
import pandas as pd

import os 
import sys





parent_dir = os.path.abspath('../..')
if parent_dir not in sys.path:
    sys.path.append(parent_dir)



import seaborn as sns
sns.set_theme()

from scipy.optimize import basinhopping


def get_metrics_names(varPrefixes:list[str], boxSize:int):
    metricsNames = []

    for varPrefix in varPrefixes:
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")

    metricsNames = np.array(metricsNames)

    return metricsNames

def get_H_at_dp(SensMatrix, CurvMatrix, dp):
    J = SensMatrix + CurvMatrix * dp
    return J.T@J



def get_data_from_file(path, varPrefix, boxSize=15):

    metrics_names = get_metrics_names(varPrefix,boxSize)

    dataset = xr.open_dataset(path, engine='netcdf4')

    return dataset[metrics_names].to_array().values.flatten()



def maximize_ratio(numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, bounds, eps=1e-4, starting_pos = None,seed = 26052026, use_grad=True, reg_gamma = 0):



    if starting_pos is None:
        starting_pos = np.zeros(numerator_SensMatrix.shape[1])
    
    def ratio_fun(dp):
        return -evaluate_ratio(dp, numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, eps = eps, reg_gamma = reg_gamma)
    
    def jac_fun(dp):
        return -evaluate_grad(dp, numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, eps=eps)

    
    if use_grad:
        result = basinhopping(ratio_fun, starting_pos, niter=1000, seed=seed, minimizer_kwargs={"method": "SLSQP", "bounds": bounds, "jac": jac_fun})
    else:
        result = basinhopping(ratio_fun, starting_pos, niter=1000, seed=seed, minimizer_kwargs={"method": "SLSQP", "bounds": bounds})

    return result.x, -result.fun




def evaluate_grad(dp, numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, eps=0.):
    v_num = evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
    v_den = evaluate_model(dp, denominator_SensMatrix, denominator_CurvMatrix)
    
    num_val = np.sum(v_num**2)
    den_val = np.sum(v_den**2) + eps
    
    grad_num = 2 * (numerator_SensMatrix + numerator_CurvMatrix * dp).T @ v_num
    grad_den = 2 * (denominator_SensMatrix + denominator_CurvMatrix * dp).T @ v_den
    
    return (grad_num * den_val - num_val * grad_den) / (den_val**2)



def evaluate_ratio(dp, numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, eps =0.,reg_gamma = 0 ):
    numerator = np.sum(evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix) ** 2)
    denominator = np.sum((evaluate_model(dp, denominator_SensMatrix, denominator_CurvMatrix)) ** 2) + reg_gamma * np.sum(dp**2)
    return numerator/(denominator+ eps)

def evaluate_model(dp, SensMatrix, CurvMatrix):
    return SensMatrix @ dp + 0.5 * CurvMatrix @ dp**2

def calculate_ratio_from_normalized_data(numerator_data, denominator_data, eps = 1e-4):
    numerator = np.sum(numerator_data ** 2)
    denominator = np.sum(denominator_data ** 2) + eps
    return numerator/denominator

def get_correct_model_runs_delta(denormalized_params, all_E3SM_params, files, sst4k_files, default_file, sst4k_default_file, varNames, global_averages):
    # get correct E3SM runs

    idxs = [np.where(np.all(np.isclose(all_E3SM_params, paramset, atol = 1e-2, rtol = 1e-2),axis =1))[0] for paramset in denormalized_params]
    if np.any(np.array([len(idx) for idx in idxs]) < 1):
        return None

    E3SM_metric_deltas_sst4k = np.array([calc_metric_sum_delta_from_file(file[0],sst4k_default_file,varNames,global_averages,15) for file in sst4k_files[idxs]])
    E3SM_metric_deltas = np.array([calc_metric_sum_delta_from_file(file[0],default_file, varNames,global_averages,15) for file in files[idxs]])


    e3sm_res_deltas = np.array([E3SM_metric_deltas, E3SM_metric_deltas_sst4k])

    return e3sm_res_deltas



def normalize_metrics_data(metrics_data, default_data, global_averages):

    weights = np.ones_like(metrics_data)

    length = metrics_data.shape[0]//len(global_averages)

    for i, global_average in enumerate(global_averages):
        weights[i*length:(i+1)*length] = np.abs(global_average)

    return (metrics_data - default_data) / weights

def denormalize_metrics_data(normalized_data, default_data, global_averages):

    weights = np.ones_like(normalized_data)

    length = normalized_data.shape[0]//len(global_averages)

    for i, global_average in enumerate(global_averages):
        weights[i*length:(i+1)*length] = np.abs(global_average)

    return (normalized_data * weights) + default_data


def calc_metric_sum_delta_from_data(data, default_data, global_averages):
    normalized_data = normalize_metrics_data(data, default_data, global_averages)

    results = np.zeros_like(global_averages)

    length = data.shape[0]//len(global_averages)
    for i in range(len(global_averages)):
        results[i] = np.sum(normalized_data[i*length:(i+1)*length]**2)

    return results

def calc_metric_sum_delta_from_file(file, default_file, varPrefixes, global_averages, boxSize):

    dataset = xr.open_dataset(file, engine='netcdf4')
    default_dataset = xr.open_dataset(default_file, engine='netcdf4')

    metric_names = get_metrics_names(varPrefixes, boxSize)

    metrics_data = dataset[metric_names].to_array().values.flatten()
    default_data = default_dataset[metric_names].to_array().values.flatten()

    results = calc_metric_sum_delta_from_data(metrics_data, default_data,global_averages)

    return results

def get_params_from_files(files, params_names):
    params = []

    for file in files:
        dataset = xr.open_dataset(file,engine="netcdf4")
        param_values = dataset[params_names].to_array().values
        params.append(param_values.flatten())

    return params



def calc_A_opt_metrics(dp, base_SensMatrix, base_CurvMatrix, constr_SensMatrix, constr_CurvMatrix):
    H_base = get_H_at_dp(base_SensMatrix, base_CurvMatrix, dp)
    H_constr = get_H_at_dp(constr_SensMatrix, constr_CurvMatrix, dp)
    H_base_inv = np.linalg.inv(H_base)
    F_vol = np.linalg.det(H_base_inv@H_constr)**(1/H_base.shape[0])

    A_opt = np.trace(H_base_inv@H_constr)

    A_opt_over_n = A_opt/H_base.shape[0]

    F_ortho_a_opt = A_opt_over_n / (F_vol)

    return A_opt_over_n, F_ortho_a_opt

def calc_D_opt_metrics(dp, base_SensMatrix, base_CurvMatrix, constr_SensMatrix, constr_CurvMatrix):
    """


    Returns  n-th root 
    """
    H_base = get_H_at_dp(base_SensMatrix, base_CurvMatrix, dp)
    H_constr = get_H_at_dp(constr_SensMatrix, constr_CurvMatrix, dp)

    H_base_inv = np.linalg.inv(H_base)

    F_scale = np.linalg.det(H_base_inv@H_constr)**(1/H_base.shape[0])

    D_optimality = np.linalg.det(H_base + H_constr)/ np.linalg.det(H_base)

    D_opt_root = D_optimality ** (1/H_base.shape[0])

    F_ortho = D_opt_root/ (1 + F_scale)


    return D_opt_root, F_ortho







def create_parameter_bar_chart(optimizer_results, params_names, base_var_name, constr_var_name):
    column_names = [fr'$dp_{{max}}^{{F,{base_var_name[0]}}}$',
                             fr'$dp_{{max}}^{{F,{base_var_name[0]}{constr_var_name[0]}}}$',
                             fr'$dp_{{max}}^{{{constr_var_name[0]},{base_var_name[0]}}}$',
                             fr'$dp_{{min}}^{{{constr_var_name[0]},{base_var_name[0]}}}$'
                            ]

    normalized_parameters = [entry[0] for entry in optimizer_results.values()]
    

    df = pd.DataFrame(np.array(normalized_parameters).T,index=params_names, columns = column_names)

    ax = df.plot.barh(figsize=(12,6))
    plt.xlim((-1,1))

    return ax

def make_scatterplot(results, base_var_name, PD_base_SensMatrix, PD_base_CurvMatrix, F_base_SensMatrix,
                      F_base_CurvMatrix,constr_var_name, PD_constr_SensMatrix, PD_constr_CurvMatrix,
                        F_constr_SensMatrix, F_constr_CurvMatrix, combined_future = False, additional_params = None, e3sm_PPE_results=None, e3sm_optimized_results=None, ax =None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))


    if combined_future:
        Future_SensMatrix = np.vstack((F_base_SensMatrix,F_constr_SensMatrix))
        Future_CurvMatrix = np.vstack((F_base_CurvMatrix,F_constr_CurvMatrix))
  
    else:
        Future_SensMatrix = F_base_SensMatrix
        Future_CurvMatrix = F_base_CurvMatrix


    combined_PD_SensMatrix = np.vstack((PD_base_SensMatrix,PD_constr_SensMatrix))
    combined_PD_CurvMatrix =  np.vstack((PD_base_CurvMatrix,PD_constr_CurvMatrix))


    R_F_B_fun = lambda dp: evaluate_ratio(dp,Future_SensMatrix, Future_CurvMatrix,PD_base_SensMatrix, PD_base_CurvMatrix)
    R_F_BC_fun = lambda dp: evaluate_ratio(dp,Future_SensMatrix, Future_CurvMatrix,combined_PD_SensMatrix, combined_PD_CurvMatrix)

   

    R_max_F_B = results[f"res_max_F_{base_var_name[0]}"][1]

    dp_max_F_B = results[f"res_max_F_{base_var_name[0]}"][0]
    dp_max_F_BC = results[f"res_max_F_{base_var_name[0]}{constr_var_name[0]}"][0]
    dp_max_C_B = results[f"res_max_{constr_var_name[0]}_{base_var_name[0]}"][0]
    dp_min_C_B = results[f"res_min_{constr_var_name[0]}_{base_var_name[0]}"][0]


    sns.scatterplot(x=[R_F_B_fun(dp_max_F_B)/R_max_F_B], y=[R_F_BC_fun(dp_max_F_B)/R_max_F_B],label=fr'$dp_{{max}}^{{F,{base_var_name[0]}}}$', color="red",s=100,linewidth=2,ax = ax)
    sns.scatterplot(x=[R_F_B_fun(dp_max_F_BC)/R_max_F_B], y=[R_F_BC_fun(dp_max_F_BC)/R_max_F_B],label=fr'$dp_{{max}}^{{F,{base_var_name[0]}{constr_var_name[0]}}}$', color="red",marker="x",s=100,linewidth=2,ax = ax)
    sns.scatterplot(x=[R_F_B_fun(dp_max_C_B)/R_max_F_B], y=[R_F_BC_fun(dp_max_C_B)/R_max_F_B],label=fr'$dp_{{max}}^{{{constr_var_name[0]},{base_var_name[0]}}}$', color="red",marker="+",s=100,linewidth=2,ax = ax)
    sns.scatterplot(x=[R_F_B_fun(dp_min_C_B)/R_max_F_B], y=[R_F_BC_fun(dp_min_C_B)/R_max_F_B],label=fr'$dp_{{min}}^{{{constr_var_name[0]},{base_var_name[0]}}}$', color="red",marker="*",edgecolor="red",s=100,linewidth=2,ax = ax)



    if additional_params is not None:

        PPE_R_F_B_results = np.apply_along_axis(R_F_B_fun, 1, np.array(additional_params))

        PPE_R_F_BC_results = np.apply_along_axis(R_F_BC_fun, 1, np.array(additional_params))

        sns.scatterplot(x=PPE_R_F_B_results/R_max_F_B, y=PPE_R_F_BC_results/R_max_F_B, label="PPE samples", color="purple",alpha=0.5,ax = ax)

    if e3sm_PPE_results is not None:

        
        PPE_deltas = e3sm_PPE_results[0]
        PPE_deltas_sst4k = e3sm_PPE_results[1]

        if combined_future:
            PPE_F_B_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/np.sum(PPE_deltas,axis=1)
        else:
            PPE_F_B_E3SM_ratios = PPE_deltas_sst4k[:,0]/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = PPE_deltas_sst4k[:,0]/np.sum(PPE_deltas,axis=1)


        sns.scatterplot(x=PPE_F_B_E3SM_ratios/R_max_F_B, y=PPE_F_BC_E3SM_ratios/R_max_F_B, label="PPE samples", color="purple",alpha=0.5,ax = ax)

    if e3sm_optimized_results is not None:
        PPE_deltas = e3sm_optimized_results[0]
        PPE_deltas_sst4k = e3sm_optimized_results[1]

        if combined_future:
            PPE_F_B_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/np.sum(PPE_deltas,axis=1)
        else:
            PPE_F_B_E3SM_ratios = PPE_deltas_sst4k[:,0]/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = PPE_deltas_sst4k[:,0]/np.sum(PPE_deltas,axis=1)

        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[0]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[0]/R_max_F_B],label=r'E3SM $dp_{max}^{F,S}$', color="green",s=100,linewidth=2,ax = ax)
        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[1]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[1]/R_max_F_B],label=r'E3SM $dp_{max}^{F,SL}$', color="green",marker="x",s=100,linewidth=2,ax = ax)
        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[2]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[2]/R_max_F_B],label=r'E3SM $dp_{max}^{L,S}$', color="green",marker="+",s=100,linewidth=2,ax = ax)


    ax.set_xlabel(fr'$\frac{{R_{{F\_{base_var_name[0]}}}}}{{R_{{F\_{base_var_name[0]}}}^{{max}}}}$',fontsize=20)
    ax.set_ylabel(fr'$\frac{{R_{{F\_{base_var_name[0]}{constr_var_name[0]}}}}}{{R_{{F\_{base_var_name[0]}}}^{{max}}}}$',fontsize=20,rotation=0)
    sns.lineplot(x=[0,1],y=[0,1], label="Upper bound", color="black", linestyle="--",alpha=0.5,ax = ax)
    return ax












def optimize_all(base_var_name, PD_base_SensMatrix, PD_base_CurvMatrix, F_base_SensMatrix, F_base_CurvMatrix,constr_var_name, 
                 PD_constr_SensMatrix, PD_constr_CurvMatrix, F_constr_SensMatrix, F_constr_CurvMatrix, normlzd_param_bounds, combined_future = False, eps = 1e-4, use_grad = True, reg_gamma = 0):

    if combined_future:
        Future_SensMatrix = np.vstack((F_base_SensMatrix,F_constr_SensMatrix))
        Future_CurvMatrix = np.vstack((F_base_CurvMatrix,F_constr_CurvMatrix))
        future_var_name = base_var_name + constr_var_name
    else:
        Future_SensMatrix = F_base_SensMatrix
        Future_CurvMatrix = F_base_CurvMatrix
        future_var_name = base_var_name

    combined_PD_SensMatrix = np.vstack((PD_base_SensMatrix,PD_constr_SensMatrix))
    combined_PD_CurvMatrix =  np.vstack((PD_base_CurvMatrix,PD_constr_CurvMatrix))


    results = {}
    # optimize for future over base
    print(f"Maximizing {future_var_name} future over {base_var_name} present-day ")
    results[f"res_max_F_{base_var_name[0]}"] = \
        maximize_ratio(
        numerator_SensMatrix=Future_SensMatrix,
        numerator_CurvMatrix=Future_CurvMatrix,
        denominator_SensMatrix=PD_base_SensMatrix,
        denominator_CurvMatrix=PD_base_CurvMatrix,
        bounds=normlzd_param_bounds,
        eps = eps,
        use_grad=use_grad,
        reg_gamma = reg_gamma
        )

    # optimize for future over restricted base
    print(f"Maximizing {future_var_name} future over {base_var_name} and {constr_var_name} present-day ")
    results[f"res_max_F_{base_var_name[0]}{constr_var_name[0]}"] =\
          maximize_ratio(
        numerator_SensMatrix=Future_SensMatrix,
        numerator_CurvMatrix=Future_CurvMatrix,
        denominator_SensMatrix=combined_PD_SensMatrix,
        denominator_CurvMatrix=combined_PD_CurvMatrix,
        bounds=normlzd_param_bounds,
        eps = eps,
        use_grad=use_grad,
        reg_gamma = reg_gamma
        )

    # optimize for constraint over base
    print(f"Maximizing present-day {constr_var_name} over present-day {base_var_name}")
    results[f"res_max_{constr_var_name[0]}_{base_var_name[0]}"] = maximize_ratio(
    numerator_SensMatrix=PD_constr_SensMatrix,
    numerator_CurvMatrix=PD_constr_CurvMatrix,
    denominator_SensMatrix=PD_base_SensMatrix,
    denominator_CurvMatrix=PD_base_CurvMatrix,
    bounds=normlzd_param_bounds,
    eps = eps,
    use_grad=use_grad,
    reg_gamma = reg_gamma
    )


    # optimize for base over constraint
    print(f"Minimizing present-day {constr_var_name} over present-day {base_var_name}")
    temp_res = maximize_ratio(
    numerator_SensMatrix=PD_base_SensMatrix,
    numerator_CurvMatrix=PD_base_CurvMatrix,
    denominator_SensMatrix=PD_constr_SensMatrix,
    denominator_CurvMatrix=PD_constr_CurvMatrix,
    bounds=normlzd_param_bounds,
    eps = eps,
    use_grad=use_grad,
    reg_gamma = reg_gamma
    )  


    results[f"res_min_{constr_var_name[0]}_{base_var_name[0]}"] = (temp_res[0],1/temp_res[1])

    return results





