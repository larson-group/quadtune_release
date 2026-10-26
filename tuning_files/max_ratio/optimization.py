
import numpy as np

from scipy.optimize import basinhopping

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

"""
All functions needed for the optimization of the different ratios
"""
def maximize_ratio(
    numerator_model,
    denominator_model,
    bounds,
    eps=1e-4,
    starting_pos=None,
    seed=None,
    inv_gamma_num=None,
    inv_gamma_den=None
):
    """
    Finds the parameter perturbation vector `dp` that globally maximizes the 
    variance ratio between a numerator model and a denominator model using 
    basin-hopping.

    Parameters
    ----------
    numerator_model : callable or torch.nn.Module
        Emulator model evaluating the numerator field perturbation at `dp`.
    denominator_model : callable or torch.nn.Module
        Emulator model evaluating the denominator field perturbation at `dp`.
    bounds : sequence of tuple
        Sequence of (min, max) bounds for each normalized parameter.
    eps : float, optional
        Small regularization constant added to the denominator to prevent 
        division by zero (default is 1e-4).
    starting_pos : numpy.ndarray, optional
        Initial parameter guess for the optimizer. If None, initializes to a 
        zero vector (default is None).
    seed : int, optional
        Random seed for the stochastic basin-hopping algorithm (default is None).
    inv_gamma_num : numpy.ndarray, optional
        Inverse noise covariance matrix for the numerator field. If None, uses 
        the unweighted sum of squares (default is None).
    inv_gamma_den : numpy.ndarray, optional
        Inverse noise covariance matrix for the denominator field. If None, uses 
        the unweighted sum of squares (default is None).

    Returns
    -------
    tuple of (numpy.ndarray, float)
        A tuple containing:
        - 1D array of the optimal normalized parameter perturbation vector `dp`.
        - Maximum achieved variance ratio (multiplied by N).
    """
    if starting_pos is None:
        starting_pos = np.zeros(len(bounds))

    is_pytorch = False
    if TORCH_AVAILABLE:
        is_pytorch = isinstance(numerator_model, torch.nn.Module)

    if is_pytorch:
        numerator_model.eval()
        denominator_model.eval()


    def objective_fun(dp):
        if is_pytorch:
            dp_tensor = torch.tensor(dp, dtype=torch.float64)
            with torch.no_grad():
                num_out = numerator_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
                den_out = denominator_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
            
            if inv_gamma_num is not None:
                inv_sig_num_t = torch.tensor(inv_gamma_num, dtype=torch.float64)
                num_val = num_out @ inv_sig_num_t @ num_out
            else:
                num_val = torch.sum(num_out**2)
                
            if inv_gamma_den is not None:
                inv_sig_den_t = torch.tensor(inv_gamma_den, dtype=torch.float64)
                den_val = den_out @ inv_sig_den_t @ den_out
            else:
                den_val = torch.sum(den_out**2)
                
            return -(num_val / (den_val + eps)).item()

        else:
            num_vec = numerator_model(dp)
            den_vec = denominator_model(dp)
            
            num_val = num_vec.T @ inv_gamma_num @ num_vec if inv_gamma_num is not None else np.sum(num_vec**2)
            den_val = den_vec.T @ inv_gamma_den @ den_vec if inv_gamma_den is not None else np.sum(den_vec**2)
            
            return -num_val / (den_val + eps)


        
    minimizer_kwargs = {"method": "SLSQP"}
    if bounds is not None:
        minimizer_kwargs["bounds"] = bounds


    if is_pytorch:
        minimizer_kwargs["jac"] = True

    optimal_niter = 20 if is_pytorch else 1000

    result = basinhopping(
        objective_fun,
        starting_pos,
        niter=optimal_niter,
        rng=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    return result.x, -result.fun
    

def evaluate_grad(
    dp,
    numerator_SensMatrix,
    numerator_CurvMatrix,
    denominator_SensMatrix,
    denominator_CurvMatrix,
    eps=0.0,
):
    """
    Computes the analytical gradient of the unweighted variance ratio with 
    respect to the parameter perturbation vector `dp` for Quadtune models.

    Parameters
    ----------
    dp : numpy.ndarray
        1D normalized parameter perturbation vector.
    numerator_SensMatrix, numerator_CurvMatrix : numpy.ndarray
        2D sensitivity (Jacobian) and curvature arrays for the numerator model.
    denominator_SensMatrix, denominator_CurvMatrix : numpy.ndarray
        2D sensitivity (Jacobian) and curvature arrays for the denominator model.
    eps : float, optional
        Numerical stability constant added to the denominator (default is 0.0).

    Returns
    -------
    numpy.ndarray
        1D gradient vector of the variance ratio at `dp`.
    """

    v_num = evaluate_Quadtune_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
    v_den = evaluate_Quadtune_model(dp, denominator_SensMatrix, denominator_CurvMatrix)

    num_val = np.sum(v_num**2)
    den_val = np.sum(v_den**2) + eps

    grad_num = 2 * (numerator_SensMatrix + numerator_CurvMatrix * dp).T @ v_num
    grad_den = 2 * (denominator_SensMatrix + denominator_CurvMatrix * dp).T @ v_den

    return (grad_num * den_val - num_val * grad_den) / (den_val**2)

def evaluate_variance(dp, SensMatrix, CurvMatrix, inv_gamma=None):
    """
    Evaluates a Quadtune model at `dp` and computes its quadratic variance 
    (weighted squared deviation or Euclidean sum of squares).

    Parameters
    ----------
    dp : numpy.ndarray
        1D normalized parameter perturbation vector.
    SensMatrix : numpy.ndarray
        2D sensitivity (Jacobian) matrix of shape (num_boxes, num_params).
    CurvMatrix : numpy.ndarray
        2D diagonal curvature matrix of shape (num_boxes, num_params).
    inv_gamma : numpy.ndarray, optional
        2D inverse noise covariance matrix of shape (num_boxes, num_boxes). 
        If None, computes the unweighted Euclidean sum of squares (default is None).

    Returns
    -------
    float
        Computed quadratic form value (N * V_m).
    """
    v = evaluate_Quadtune_model(dp, SensMatrix, CurvMatrix)
    
    if inv_gamma is not None:
        return v.T @ inv_gamma @ v
    else:
        return np.sum(v**2)
    

def maximize_constr_problem(
    base_model,
    numerator_model,
    constr_value,
    denominator_model = None,
    bounds=None,
    eps=1e-4,
    starting_pos=None,
    seed=None,
    inv_gamma_base=None,  
    inv_gamma_num=None,    
    inv_gamma_den=None
):
    """
    Maximizes either a numerator variance or a variance ratio subject to an 
    inequality constraint bounding the baseline present-day variance.

    Uses basin-hopping with Sequential Least Squares Programming (SLSQP) to 
    maximize `V_num(dp)` (if `denominator_model` is None) or 
    `V_num(dp) / (V_den(dp) + constr_value + eps)` subject to 
    `V_base(dp) <= constr_value`.

    Parameters
    ----------
    base_model : callable or torch.nn.Module
        Emulator model for the primary present-day constraint field.
    numerator_model : callable or torch.nn.Module
        Emulator model for the target field in the numerator.
    constr_value : float
        Maximum allowable threshold for the primary constraint quadratic form.
    denominator_model : callable or torch.nn.Module, optional
        Emulator model for the secondary constraint field in the denominator. 
        If None, only the numerator is maximized (default is None).
    bounds : sequence of tuple, optional
        Sequence of (min, max) bounds for each normalized parameter (default is None).
    eps : float, optional
        Small regularization constant added to the denominator (default is 1e-4).
    starting_pos : numpy.ndarray, optional
        Initial parameter guess for the optimizer. If None, initializes to a 
        zero vector (default is None).
    seed : int, optional
        Random seed for the stochastic basin-hopping algorithm (default is None).
    inv_gamma_base : numpy.ndarray, optional
        Inverse noise covariance matrix for the primary constraint field (default is None).
    inv_gamma_num : numpy.ndarray, optional
        Inverse noise covariance matrix for the numerator field (default is None).
    inv_gamma_den : numpy.ndarray, optional
        Inverse noise covariance matrix for the secondary constraint field (default is None).

    Returns
    -------
    tuple of (numpy.ndarray, float)
        A tuple containing:
        - 1D array of the optimal normalized parameter perturbation vector `dp`.
        - Maximum achieved objective value.
    """
    if starting_pos is None:
        starting_pos = np.zeros(len(bounds))

    is_pytorch = False
    if TORCH_AVAILABLE:
        is_pytorch = isinstance(numerator_model, torch.nn.Module)


    if is_pytorch:
        base_model.eval()
        numerator_model.eval()
        if denominator_model is not None:
            denominator_model.eval()

    def objective_fun(dp):
        if is_pytorch:
            # Gaussian Process Models (PyTorch) with autograd
            dp_tensor = torch.tensor(dp, dtype=torch.float64, requires_grad=True)
            num_out = numerator_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
            
            if denominator_model is not None:
                denom_out = denominator_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
                obj = -torch.sum(num_out**2) / (torch.sum(denom_out**2) + constr_value + eps)
            else:
                obj = -torch.sum(num_out**2)
                
            obj.backward()
            return obj.detach().numpy().astype(np.float64), dp_tensor.grad.numpy().astype(np.float64)
            
        else:
            # Quadtune Model
            num_vec = numerator_model(dp)
            num_val = num_vec.T @ inv_gamma_num @ num_vec if inv_gamma_num is not None else np.sum(num_vec**2)
            
            if denominator_model is not None:
                den_vec = denominator_model(dp)
                den_val = den_vec.T @ inv_gamma_den @ den_vec if inv_gamma_den is not None else np.sum(den_vec**2)
                return -num_val / (den_val + constr_value + eps)
            else:
                return -num_val

    def constraint_fun(dp): 
        if is_pytorch:
            dp_tensor = torch.tensor(dp, dtype=torch.float64)
            with torch.no_grad():
                base_out = base_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
                base_val = torch.sum(base_out**2)
            return (constr_value - base_val).numpy().astype(np.float64)
        else:
            base_vec = base_model(dp)
            base_val = base_vec.T @ inv_gamma_base @ base_vec if inv_gamma_base is not None else np.sum(base_vec**2)
            return constr_value - base_val

    def constraint_jac(dp):
        dp_tensor = torch.tensor(dp, dtype=torch.float64, requires_grad=True)
        base_out = base_model(dp_tensor.unsqueeze(0)).mean.squeeze(0)
        constr_obj = constr_value - torch.sum(base_out**2)
        constr_obj.backward()
        return dp_tensor.grad.numpy().astype(np.float64)

    constraints_dict = {"type": "ineq", "fun": constraint_fun}
    if is_pytorch:
        constraints_dict["jac"] = constraint_jac

    minimizer_kwargs = {
        "method": "SLSQP",
        "constraints": [constraints_dict]
    }



    if is_pytorch:
        minimizer_kwargs["jac"] = True
    if bounds is not None:
        minimizer_kwargs["bounds"] = bounds

    optimal_niter = 20 if is_pytorch else 1000

    result = basinhopping(
        objective_fun,
        starting_pos,
        niter=optimal_niter,
        rng=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    return result.x, -result.fun



def evaluate_Quadtune_ratio(
    dp,
    numerator_SensMatrix,
    numerator_CurvMatrix,
    denominator_SensMatrix,
    denominator_CurvMatrix,
    inv_gamma_num=None,
    inv_gamma_den=None,
    eps=0.0,
    reg_gamma=0,
):
    """
    Computes the variance ratio of two Quadtune models at a specific parameter 
    perturbation `dp`.

    Parameters
    ----------
    dp : numpy.ndarray
        1D normalized parameter perturbation vector.
    numerator_SensMatrix, numerator_CurvMatrix : numpy.ndarray
        2D sensitivity and curvature arrays for the numerator model.
    denominator_SensMatrix, denominator_CurvMatrix : numpy.ndarray
        2D sensitivity and curvature arrays for the denominator model.
    inv_gamma_num : numpy.ndarray, optional
        Inverse noise covariance matrix for the numerator. If None, uses the 
        identity matrix (default is None).
    inv_gamma_den : numpy.ndarray, optional
        Inverse noise covariance matrix for the denominator. If None, uses the 
        identity matrix (default is None).
    eps : float, optional
        Numerical stability constant added to the denominator (default is 0.0).
    reg_gamma : float, optional
        L2 regularization weight penalizing parameter norm `||dp||^2` in the 
        denominator (default is 0).

    Returns
    -------
    float
        The evaluated variance ratio.
    """

    if inv_gamma_num is None or inv_gamma_den is None:
        print("Warning: Inverse covariance matrices not provided. Using identity matrices for both numerator and denominator.")
        inv_gamma_num = np.eye(numerator_SensMatrix.shape[0])
        inv_gamma_den = np.eye(denominator_SensMatrix.shape[0])

    v_num = evaluate_Quadtune_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
    numerator = v_num.T @ inv_gamma_num @ v_num
    
    v_den = evaluate_Quadtune_model(dp, denominator_SensMatrix, denominator_CurvMatrix)
    denominator = (v_den.T @ inv_gamma_den @ v_den) + reg_gamma * np.sum(dp**2)
    
    return numerator / (denominator + eps)


def evaluate_Quadtune_model(dp, SensMatrix, CurvMatrix):
    """
    Evaluates a quadratic Quadtune emulator model at a specific parameter 
    perturbation `dp`.

    Parameters
    ----------
    dp : numpy.ndarray
        1D array of length P_n representing normalized parameter perturbations.
    SensMatrix : numpy.ndarray
        2D array of shape (N, P_n) representing the first-order sensitivity 
        (Jacobian) matrix.
    CurvMatrix : numpy.ndarray
        2D array of shape (N, P_n) representing the unmixed second-order 
        curvature (diagonal Hessian) matrix.

    Returns
    -------
    numpy.ndarray
        1D array of length N representing the emulated regional field perturbation.
    """
    return SensMatrix @ dp + 0.5 * CurvMatrix @ dp**2

def optimize_all(
        base_var_name,
        constr_var_name,
        PD_base_model,
        F_base_model,
        PD_constr_model,
        PD_combined_model,
        normlzd_param_bounds,
        eps=1e-4,
        seed=None,
        run_exact_constraint=False,
        exact_constr_value=None,
        inv_gamma_base=None,     
        inv_gamma_constr=None,   
        inv_gamma_combined=None  
):
    """
    Executes the suite of optimization scenarios for a given primary constraint 
    field and secondary constraint field.

    Computes the optimal parameter perturbations for single-constraint future 
    variance (`res_max_F_S`), double-constraint future variance (`res_max_F_SL`), 
    and present-day relative constraint strength (`res_max_L_S`, plus `res_min_L_S` 
    when `run_exact_constraint` is False).

    Parameters
    ----------
    base_var_name : str
        Identifier for the primary constraint and target field (e.g., 'SWCF').
    constr_var_name : str
        Identifier for the secondary constraint field (e.g., 'LWCF', 'PRECT', 'TMQ').
    PD_base_model : callable or torch.nn.Module
        Present-day emulator model for the primary field.
    F_base_model : callable or torch.nn.Module
        Future (+4K SST) emulator model for the target field.
    PD_constr_model : callable or torch.nn.Module
        Present-day emulator model for the secondary constraint field.
    PD_combined_model : callable or torch.nn.Module
        Present-day emulator model for the stacked primary and secondary fields.
    normlzd_param_bounds : sequence of tuple
        Sequence of (min, max) bounds for the normalized parameter space.
    eps : float, optional
        Numerical stability constant for ratio denominators (default is 1e-4).
    seed : int, optional
        Random seed for the stochastic global optimizer (default is None).
    run_exact_constraint : bool, optional
        If True, normalizes the primary constraint denominator to `exact_constr_value` 
        and performs constrained optimization instead of direct ratio maximization 
        (default is False).
    exact_constr_value : float, optional
        Threshold value for the primary constraint when `run_exact_constraint` 
        is True (default is None).
    inv_gamma_base : numpy.ndarray, optional
        Inverse noise covariance matrix for the primary field (default is None).
    inv_gamma_constr : numpy.ndarray, optional
        Inverse noise covariance matrix for the secondary constraint field (default is None).
    inv_gamma_combined : numpy.ndarray, optional
        Block-diagonal inverse noise covariance matrix for the combined fields 
        (default is None).

    Returns
    -------
    dict of {str: tuple of (numpy.ndarray, float)}
        Dictionary mapping scenario keys (e.g., 'res_max_F_S', 'res_max_F_SL', 
        'res_max_L_S') to tuples of (optimal_dp_vector, optimal_objective_value).
    """

    if constr_var_name == "TMQ":
        short_constr_name = "Q"
    else:
        short_constr_name = constr_var_name[0]

    if base_var_name == "TMQ":
        short_base_name = "Q"
    else:
        short_base_name = base_var_name[0]

    results = {}

    if run_exact_constraint:

        print(
            f"Maximizing {base_var_name} future with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_F_{short_base_name}"] = maximize_constr_problem(
            PD_base_model, F_base_model, constr_value=exact_constr_value, denominator_model=None, 
            bounds=normlzd_param_bounds, eps=eps, seed=seed,
            inv_gamma_base=inv_gamma_base, inv_gamma_num=inv_gamma_base, inv_gamma_den=None
        )

        print(
            f"Maximizing ratio ({base_var_name} / {constr_var_name}) with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_F_{short_base_name}{short_constr_name}"] = maximize_constr_problem(
            PD_base_model, F_base_model, constr_value=exact_constr_value, denominator_model=PD_constr_model, 
            bounds=normlzd_param_bounds, eps=eps, seed=seed,
            inv_gamma_base=inv_gamma_base, inv_gamma_num=inv_gamma_base, inv_gamma_den=inv_gamma_constr
        )

        print(
            f"Maximizing {constr_var_name} present-day with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_{short_constr_name}_{short_base_name}"] = maximize_constr_problem(
            PD_base_model, PD_constr_model, constr_value=exact_constr_value, denominator_model=None, 
            bounds=normlzd_param_bounds, eps=eps, seed=seed,
            inv_gamma_base=inv_gamma_base, inv_gamma_num=inv_gamma_constr, inv_gamma_den=None
        )

    else:

        # optimize for future over base
        print(f"Maximizing {base_var_name} future over {base_var_name} present-day ")
        results[f"res_max_F_{short_base_name}"] = maximize_ratio(
            F_base_model, PD_base_model, bounds=normlzd_param_bounds, eps=eps,
            inv_gamma_num=inv_gamma_base, inv_gamma_den=inv_gamma_base,seed=seed
        )

        # optimize for future over restricted base
        print(
            f"Maximizing {base_var_name} future over {base_var_name} and {constr_var_name} present-day "
        )
        results[f"res_max_F_{short_base_name}{short_constr_name}"] = maximize_ratio(
            F_base_model, PD_combined_model, bounds=normlzd_param_bounds, eps=eps,
            inv_gamma_num=inv_gamma_base, inv_gamma_den=inv_gamma_combined,seed=seed
        )

        # optimize for constraint over base
        print(
            f"Maximizing present-day {constr_var_name} over present-day {base_var_name}"
        )
        results[f"res_max_{short_constr_name}_{short_base_name}"] = maximize_ratio(
            PD_constr_model, PD_base_model, bounds=normlzd_param_bounds, eps=eps,
            inv_gamma_num=inv_gamma_constr, inv_gamma_den=inv_gamma_base, seed=seed
        )

        # optimize for base over constraint
        print(
            f"Minimizing present-day {constr_var_name} over present-day {base_var_name}"
        )
        temp_res = maximize_ratio(
            PD_base_model, PD_constr_model, bounds=normlzd_param_bounds, eps=eps,
            inv_gamma_num=inv_gamma_base, inv_gamma_den=inv_gamma_constr, seed=seed
        )

        results[f"res_min_{short_constr_name}_{short_base_name}"] = (
            temp_res[0],
            1 / temp_res[1],
        )

    return results


def normalize_metrics_data(metrics_data, default_data, global_average_obs):
    """
    Computes regional deviations from a default simulation normalized by 
    global reference averages.

    Parameters
    ----------
    metrics_data : numpy.ndarray
        1D or 2D array of regional metric values in physical units.
    default_data : numpy.ndarray
        1D array of default simulation metric values in physical units.
    global_average_obs : numpy.ndarray
        1D array of global reference averages for each variable.

    Returns
    -------
    numpy.ndarray
        Array of dimensionless, normalized metric perturbations of the same 
        shape as `metrics_data`.
    """
    num_features = len(default_data)
    
    factors = np.zeros(num_features)
    length = num_features // len(global_average_obs)
    
    for i in range(len(global_average_obs)):
        factors[i*length:(i+1)*length] = np.abs(global_average_obs[i])
        
    return (metrics_data - default_data) / factors


def denormalize_metrics_data(normalized_data, default_data, global_averages):
    """
    Reverts normalized metric deviations back to their original physical units.

    Parameters
    ----------
    normalized_data : numpy.ndarray
        1D array of normalized metric perturbations.
    default_data : numpy.ndarray
        1D array of default simulation data in physical units.
    global_averages : numpy.ndarray
        1D array of reference global averages used as scaling weights.

    Returns
    -------
    numpy.ndarray
        1D array of regional metric data projected back into physical units.
    """
    weights = np.ones_like(normalized_data)

    length = normalized_data.shape[0] // len(global_averages)

    for i, global_average in enumerate(global_averages):
        weights[i * length : (i + 1) * length] = np.abs(global_average)

    return (normalized_data * weights) + default_data


def get_H_at_dp(SensMatrix, CurvMatrix, dp, inv_gamma = None):
    """
    Computes the local P_n x P_n Gramian matrix H at a specific parameter 
    perturbation `dp`.

    Linearizes the quadratic Quadtune model about `dp` via the local Jacobian 
    `J(dp) = SensMatrix + CurvMatrix * dp` and evaluates `J(dp)^T @ inv_gamma @ J(dp)`.

    Parameters
    ----------
    SensMatrix : numpy.ndarray
        2D sensitivity matrix of shape (N, P_n).
    CurvMatrix : numpy.ndarray
        2D curvature matrix of shape (N, P_n).
    dp : numpy.ndarray
        1D parameter perturbation vector of length P_n about which to linearize.
    inv_gamma : numpy.ndarray, optional
        2D inverse noise covariance matrix of shape (N, N). If None, uses the 
        identity matrix (default is None).

    Returns
    -------
    numpy.ndarray
        2D symmetric positive semi-definite Gramian matrix of shape (P_n, P_n).
    """

    J = SensMatrix + CurvMatrix * dp
    if inv_gamma is None:
        return J.T @ J
    else:
        return J.T @ inv_gamma @ J


def calc_A_opt_metrics(H_base, H_constr):
    """
    Calculates analytical scale and shape decomposition metrics from the primary 
    and secondary Gramian matrices.

    Parameters
    ----------
    H_base : numpy.ndarray
        2D Gramian matrix H_S of the primary baseline constraint (P_n, P_n).
    H_constr : numpy.ndarray
        2D Gramian matrix H_L of the secondary constraint (P_n, P_n).

    Returns
    -------
    tuple of (float, float, float, float)
        - R_scale : Volumetric overall constraint strength ratio based on determinants.
        - E_RLS : Expected relative constraint strength (average eigenvalue of H_S^-1 H_L).
        - R_shape : Complementarity alignment factor (E_RLS / R_scale).
        - A_opt_over_n_slope : Reference slope `1 / (1 + E_RLS)`.
    """
    H_base_inv = np.linalg.pinv(H_base)

    R_scale = (np.linalg.det(H_constr) / np.linalg.det(H_base)) ** (1 / H_base.shape[0])

    A_opt = np.trace(H_base_inv @ H_constr)

    E_RLS = A_opt / H_base.shape[0]

    A_opt_over_n_slope = 1 / (1 + E_RLS)

    R_shape = E_RLS / (R_scale)

    return R_scale, E_RLS, R_shape, A_opt_over_n_slope

def calc_all_A_opt_metrics(all_optimizations, used_fields, base_field, all_fields_data, paramset_idx, inv_gamma_dict=None):
    """
    Calculates the analytical scale, shape, and expected relative constraint 
    metrics across all evaluated secondary constraint fields.

    Parameters
    ----------
    all_optimizations : dict
        Nested dictionary mapping field names to their optimization results.
    used_fields : list of str
        Names of all fields being evaluated (including `base_field`).
    base_field : str
        Name of the primary baseline constraint field (e.g., 'SWCF').
    all_fields_data : dict
        Dictionary mapping field names to lists of their sensitivity and 
        curvature matrices.
    paramset_idx : int
        Index of the optimization scenario in `all_optimizations[field]` about 
        which the Quadtune model is linearized.
    inv_gamma_dict : dict of {str: numpy.ndarray}, optional
        Dictionary mapping field names to their inverse noise covariance matrices 
        (default is None).

    Returns
    -------
    tuple of (list of float, list of float, list of float, list of float)
        - Scales_metrics : List of R_scale values for each secondary constraint.
        - E_RLS_allfields : List of expected relative constraint strengths E_RLS.
        - Shapes_metrics : List of R_shape complementarity values.
        - A_opt_over_n_slopes : List of `1 / (1 + E_RLS)` slopes.
    """
    Scales_metrics = []
    Shapes_metrics = []
    E_RLS_allfields= []
    A_opt_over_n_slopes = []


    BaseSensMatrix, BaseCurvMatrix = all_fields_data[base_field][:2]

    for field in used_fields:
        if field == base_field:
            continue
        
        opt_dict = all_optimizations[field]
        values_list = list(opt_dict.values())
        
        constrained_parameter_set = np.array(values_list[paramset_idx][0])

        # H is the P_n x P_n Gramian Matrix (S+C*diag(dp))^T (S+C*diag(dp)) where S is the sensitivity matrix and C the curvature matrix
        ConstrSensMatrix, ConstrCurvMatrix = all_fields_data[field][:2]
        inv_sig_base = inv_gamma_dict[base_field] if inv_gamma_dict is not None else None
        inv_sig_constr = inv_gamma_dict[field] if inv_gamma_dict is not None else None

        H_Base = get_H_at_dp(BaseSensMatrix, BaseCurvMatrix, constrained_parameter_set, inv_gamma=inv_sig_base)
        H_Constr = get_H_at_dp(ConstrSensMatrix, ConstrCurvMatrix, constrained_parameter_set, inv_gamma=inv_sig_constr)
        R_scale, E_RLS, R_shape, A_opt_over_n = calc_A_opt_metrics(H_Base, H_Constr)
        

        Scales_metrics.append(R_scale)
        Shapes_metrics.append(R_shape)
        E_RLS_allfields.append(E_RLS)
        A_opt_over_n_slopes.append(A_opt_over_n)
    
    return Scales_metrics, E_RLS_allfields, Shapes_metrics, A_opt_over_n_slopes


def calc_E_RR(H_base, H_constr):
    """
    Calculates the expected ratio of variance ratios (E_RR) by averaging over 
    the principal axes of the relative information matrix.

    Evaluates `trace((I + H_base^-1 @ H_constr)^-1) / P_n`.

    Parameters
    ----------
    H_base : numpy.ndarray
        2D Gramian matrix H_S of the primary baseline constraint (P_n, P_n).
    H_constr : numpy.ndarray
        2D Gramian matrix H_L of the secondary constraint (P_n, P_n).

    Returns
    -------
    float
        The expected ratio of variance ratios E_RR.
    """
    trace = np.trace(
        np.linalg.pinv(np.eye(H_base.shape[0]) + np.linalg.pinv(H_base) @ H_constr)
    )

    return trace / H_base.shape[0]

def calc_all_E_RR(all_optimizations, used_fields, base_field, all_fields_data, paramset_idx, inv_gamma_dict=None):
    """
    Calculates the expected ratio of variance ratios (E_RR) for all secondary 
    constraint fields.

    Parameters
    ----------
    all_optimizations : dict
        Nested dictionary mapping field names to their optimization results.
    used_fields : list of str
        Names of all fields being evaluated (including `base_field`).
    base_field : str
        Name of the primary baseline constraint field (e.g., 'SWCF').
    all_fields_data : dict
        Dictionary mapping field names to lists of their sensitivity and 
        curvature matrices.
    paramset_idx : int
        Index of the optimization scenario in `all_optimizations[field]` about 
        which the Quadtune model is linearized.
    inv_gamma_dict : dict of {str: numpy.ndarray}, optional
        Dictionary mapping field names to their inverse noise covariance matrices 
        (default is None).

    Returns
    -------
    list of float
        Computed E_RR values for each secondary constraint field.
    """
    E_RR_values = []

    BaseSensMatrix, BaseCurvMatrix = all_fields_data[base_field][:2]


    for field in used_fields:
        if field == base_field:
            continue
            
        opt_dict = all_optimizations[field]
        values_list = list(opt_dict.values())
        
        constrained_parameter_set = np.array(values_list[paramset_idx][0])


        ConstrSensMatrix, ConstrCurvMatrix = all_fields_data[field][:2]
        inv_sig_base = inv_gamma_dict[base_field] if inv_gamma_dict is not None else None
        inv_sig_constr = inv_gamma_dict[field] if inv_gamma_dict is not None else None

        H_B = get_H_at_dp(BaseSensMatrix, BaseCurvMatrix, constrained_parameter_set, inv_gamma=inv_sig_base)
        H_C = get_H_at_dp(ConstrSensMatrix, ConstrCurvMatrix, constrained_parameter_set, inv_gamma=inv_sig_constr)

        E_RR_values.append(calc_E_RR(H_B, H_C))

    return E_RR_values

def normalize_params(params, default_params):
    """
    Converts physical parameter values into dimensionless perturbations relative 
    to the default parameter set.

    Parameters
    ----------
    params : numpy.ndarray
        Array of parameter values in physical units.
    default_params : numpy.ndarray
        Array of default parameter values in physical units.

    Returns
    -------
    numpy.ndarray
        Normalized parameter perturbations `(params - default_params) / |default_params|`.
    """
    return (params - default_params) / np.abs(default_params)


def denormalize_params(normlzd_params, default_params):
    """
    Converts dimensionless parameter perturbations back into physical parameter values.

    Parameters
    ----------
    normlzd_params : numpy.ndarray
        Normalized parameter perturbations `dp`.
    default_params : numpy.ndarray
        Array of default parameter values in physical units.

    Returns
    -------
    numpy.ndarray
        Parameter values in physical units.
    """
    return normlzd_params * np.abs(default_params) + default_params
