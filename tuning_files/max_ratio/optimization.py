
import numpy as np

from scipy.optimize import basinhopping



"""
All functions needed for the optimization of the different ratios
"""

def maximize_ratio(numerator_SensMatrix, numerator_CurvMatrix, denominator_SensMatrix, denominator_CurvMatrix, bounds, eps=1e-4, starting_pos = None, seed = 26052026, use_grad=True, reg_gamma = 0):



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

def optimize_all(base_var_name, PD_base_SensMatrix, PD_base_CurvMatrix, F_base_SensMatrix, F_base_CurvMatrix,constr_var_name, 
                 PD_constr_SensMatrix, PD_constr_CurvMatrix, F_constr_SensMatrix, F_constr_CurvMatrix, normlzd_param_bounds, combined_future = False, eps = 1e-4, use_grad = True, reg_gamma = 0, seed=None):

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

    if constr_var_name == "TMQ":
        short_constr_name = 'Q'
    else:
        short_constr_name = constr_var_name[0]

    if base_var_name == "TMQ":
        short_base_name = 'Q'
    else:
        short_base_name = base_var_name[0]


    results = {}
    # optimize for future over base
    print(f"Maximizing {future_var_name} future over {base_var_name} present-day ")
    results[f"res_max_F_{short_base_name}"] = \
        maximize_ratio(
        numerator_SensMatrix=Future_SensMatrix,
        numerator_CurvMatrix=Future_CurvMatrix,
        denominator_SensMatrix=PD_base_SensMatrix,
        denominator_CurvMatrix=PD_base_CurvMatrix,
        bounds=normlzd_param_bounds,
        eps = eps,
        use_grad=use_grad,
        reg_gamma = reg_gamma,
        seed=seed
        )

    # optimize for future over restricted base
    print(f"Maximizing {future_var_name} future over {base_var_name} and {constr_var_name} present-day ")
    results[f"res_max_F_{short_base_name}{short_constr_name}"] =\
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
    results[f"res_max_{short_constr_name}_{short_base_name}"] = maximize_ratio(
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


    results[f"res_min_{short_constr_name}_{short_base_name}"] = (temp_res[0],1/temp_res[1])

    return results


def normalize_metrics_data(metrics_data, default_data, global_averages):
    length = metrics_data.shape[0] // len(global_averages)
    
    weights = np.repeat(np.abs(global_averages), length)
    
    return (metrics_data - default_data) / weights

def denormalize_metrics_data(normalized_data, default_data, global_averages):

    weights = np.ones_like(normalized_data)

    length = normalized_data.shape[0]//len(global_averages)

    for i, global_average in enumerate(global_averages):
        weights[i*length:(i+1)*length] = np.abs(global_average)

    return (normalized_data * weights) + default_data


def get_H_at_dp(SensMatrix, CurvMatrix, dp):
    J = SensMatrix + CurvMatrix * dp
    return J.T@J


def calc_A_opt_metrics(H_base, H_constr):
    H_base_inv = np.linalg.pinv(H_base)


    R_scale = (np.linalg.det(H_constr)/ np.linalg.det(H_base)) ** (1/H_base.shape[0])

    A_opt = np.trace(H_base_inv@H_constr)

    E_RLS = A_opt/H_base.shape[0]

    A_opt_over_n_slope = 1 /(1+ E_RLS)

    R_ortho = E_RLS / (R_scale)

    return R_scale, E_RLS, R_ortho, A_opt_over_n_slope

def calc_D_opt_metrics(H_base, H_constr):


    H_base_inv = np.linalg.pinv(H_base)

    F_scale = np.linalg.det(H_base_inv@H_constr)**(1/H_base.shape[0])

    D_optimality = np.linalg.det(H_base + H_constr)/ np.linalg.det(H_base)

    D_opt_root = D_optimality ** (1/H_base.shape[0])

    F_ortho = D_opt_root/ (1 + F_scale)


    return D_opt_root, F_ortho


def calc_E_RR(H_base, H_constr):
    trace = np.linalg.trace(  np.linalg.pinv( np.eye(H_base.shape[0]) +  np.linalg.pinv(H_base)  @  H_constr)     )

    return trace / H_base.shape[0]


def normalize_params(params, default_params):
    return (params - default_params) / np.abs(default_params)

def denormalize_params(normlzd_params, default_params):
    return normlzd_params * np.abs(default_params) + default_params