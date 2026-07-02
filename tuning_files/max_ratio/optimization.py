import warnings

import numpy as np

from scipy.optimize import basinhopping

"""
All functions needed for the optimization of the different ratios
"""


def maximize_ratio(
    numerator_SensMatrix,
    numerator_CurvMatrix,
    denominator_SensMatrix,
    denominator_CurvMatrix,
    bounds,
    eps=1e-4,
    starting_pos=None,
    seed=None,
    use_grad=True,
    reg_gamma=0,
):
    """
    Maximizes the ratio of two Quadtune models using the Basin-Hopping algorithm.

    Parameters
    ----------
    numerator_SensMatrix, numerator_CurvMatrix : numpy.ndarray
        2D matrices defining the sensitivity and curvature of the target field to maximize.
    denominator_SensMatrix, denominator_CurvMatrix : numpy.ndarray
        2D matrices defining the sensitivity and curvature of the constraint field.
    bounds : sequence of tuples
        (min, max) bounds for the parameter space.
    eps : float, optional
        Numerical stability constant (default is 1e-4).
    starting_pos : numpy.ndarray, optional
        Initial guess for the parameter vector (default is zeros).
    seed : int, optional
        Random seed (default is None).
    use_grad : bool, optional
        If True, uses the analytical gradient function via SLSQP (default is True).
    reg_gamma : float, optional
        L2 regularization factor (default is 0).

    Returns
    -------
    tuple
        - x (numpy.ndarray): The optimized parameter vector.
        - fun (float): The maximized ratio value.
    """

    if starting_pos is None:
        starting_pos = np.zeros(numerator_SensMatrix.shape[1])

    def ratio_fun(dp):
        return -evaluate_ratio(
            dp,
            numerator_SensMatrix,
            numerator_CurvMatrix,
            denominator_SensMatrix,
            denominator_CurvMatrix,
            eps=eps,
            reg_gamma=reg_gamma,
        )

    def jac_fun(dp):
        return -evaluate_grad(
            dp,
            numerator_SensMatrix,
            numerator_CurvMatrix,
            denominator_SensMatrix,
            denominator_CurvMatrix,
            eps=eps,
        )

    with warnings.catch_warnings():
        # Catch SLSQP clipping to bounds warning to clean up the output
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")

    if use_grad:
        result = basinhopping(
            ratio_fun,
            starting_pos,
            niter=1000,
            rng=seed,
            minimizer_kwargs={"method": "SLSQP", "bounds": bounds, "jac": jac_fun},
        )
    else:
        result = basinhopping(
            ratio_fun,
            starting_pos,
            niter=1000,
            rng=seed,
            minimizer_kwargs={"method": "SLSQP", "bounds": bounds},
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
    Computes the analytical gradient of the variance ratio with respect to the parameter vector.

    Parameters
    ----------
    dp : numpy.ndarray
        1D parameter vector.
    numerator_SensMatrix, numerator_CurvMatrix : numpy.ndarray
        2D arrays for the numerator model.
    denominator_SensMatrix, denominator_CurvMatrix : numpy.ndarray
        2D arrays for the denominator model.
    eps : float, optional
        Numerical stability constant (default is 0.0).

    Returns
    -------
    numpy.ndarray
        1D gradient vector.
    """

    v_num = evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
    v_den = evaluate_model(dp, denominator_SensMatrix, denominator_CurvMatrix)

    num_val = np.sum(v_num**2)
    den_val = np.sum(v_den**2) + eps

    grad_num = 2 * (numerator_SensMatrix + numerator_CurvMatrix * dp).T @ v_num
    grad_den = 2 * (denominator_SensMatrix + denominator_CurvMatrix * dp).T @ v_den

    return (grad_num * den_val - num_val * grad_den) / (den_val**2)


def maximize_constr_problem(
    base_SensMatrix,
    base_CurvMatrix,
    numerator_SensMatrix,
    numerator_CurvMatrix,
    constr_value,
    denominator_SensMatrix=None,
    denominator_CurvMatrix=None,
    bounds=None,
    eps=1e-4,
    starting_pos=None,
    seed=None,
    use_grad=True,
    reg_gamma=0,
):
    """
    Maximizes either the numerator model OR the ratio of (numerator / denominator),
    subject to the base model's sum of squares being exactly constrained to `constr_value`.
    """
    if starting_pos is None:
        starting_pos = np.zeros(numerator_SensMatrix.shape[1])

    is_ratio = denominator_SensMatrix is not None

    def objective_fun(dp):
        if is_ratio:
            return -evaluate_ratio(
                dp,
                numerator_SensMatrix,
                numerator_CurvMatrix,
                denominator_SensMatrix,
                denominator_CurvMatrix,
                eps=constr_value + eps,
                reg_gamma=reg_gamma,
            )
        else:
            v_num = evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
            return -np.sum(v_num**2)

    def jac_fun(dp):
        if is_ratio:
            return -evaluate_grad(
                dp,
                numerator_SensMatrix,
                numerator_CurvMatrix,
                denominator_SensMatrix,
                denominator_CurvMatrix,
                eps=constr_value + eps,
            )
        else:
            v_num = evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix)
            grad_num = 2 * (numerator_SensMatrix + numerator_CurvMatrix * dp).T @ v_num
            return -grad_num

    def constraint_fun(dp):
        base_val = np.sum(evaluate_model(dp, base_SensMatrix, base_CurvMatrix) ** 2)
        return constr_value - base_val

    def constraint_jac(dp):
        v_base = evaluate_model(dp, base_SensMatrix, base_CurvMatrix)
        return -2 * (base_SensMatrix + base_CurvMatrix * dp).T @ v_base

    constraints = [{"type": "ineq", "fun": constraint_fun}]
    if use_grad:
        constraints[0]["jac"] = constraint_jac

    minimizer_kwargs = {"method": "SLSQP", "constraints": constraints}

    if bounds is not None:
        minimizer_kwargs["bounds"] = bounds
    if use_grad:
        minimizer_kwargs["jac"] = jac_fun

    result = basinhopping(
        objective_fun,
        starting_pos,
        niter=1000,
        rng=seed,
        minimizer_kwargs=minimizer_kwargs,
    )

    return result.x, -result.fun


def evaluate_ratio(
    dp,
    numerator_SensMatrix,
    numerator_CurvMatrix,
    denominator_SensMatrix,
    denominator_CurvMatrix,
    eps=0.0,
    reg_gamma=0,
):
    """
    Computes the variance ratio of two Quadtune models.

    Parameters
    ----------
    dp : numpy.ndarray
        1D parameter vector.
    numerator_SensMatrix, numerator_CurvMatrix : numpy.ndarray
        2D arrays for the numerator model.
    denominator_SensMatrix, denominator_CurvMatrix : numpy.ndarray
        2D arrays for the denominator model.
    eps : float, optional
        Numerical stability constant (default is 0.0).
    reg_gamma : float, optional
        L2 regularization weight (default is 0.0).

    Returns
    -------
    float
        The calculated ratio.
    """
    numerator = np.sum(
        evaluate_model(dp, numerator_SensMatrix, numerator_CurvMatrix) ** 2
    )
    denominator = np.sum(
        (evaluate_model(dp, denominator_SensMatrix, denominator_CurvMatrix)) ** 2
    ) + reg_gamma * np.sum(dp**2)
    return numerator / (denominator + eps)


def evaluate_model(dp, SensMatrix, CurvMatrix):
    """
    Evaluates a Quadtune model at a specific dp.

    Parameters
    ----------
    dp : numpy.ndarray
        1D array of length N representing parameter perturbations.
    SensMatrix : numpy.ndarray
        2D array of shape (M, N) representing the Jacobian matrix.
    CurvMatrix : numpy.ndarray
        2D array of shape (M, N) representing the diagonal of the Hessian matrix.

    Returns
    -------
    numpy.ndarray
        1D array of length M representing the evaluated model output.
    """
    return SensMatrix @ dp + 0.5 * CurvMatrix @ dp**2


def optimize_all(
    base_var_name,
    PD_base_SensMatrix,
    PD_base_CurvMatrix,
    F_base_SensMatrix,
    F_base_CurvMatrix,
    constr_var_name,
    PD_constr_SensMatrix,
    PD_constr_CurvMatrix,
    F_constr_SensMatrix,
    F_constr_CurvMatrix,
    normlzd_param_bounds,
    combined_future=False,
    eps=1e-4,
    use_grad=True,
    reg_gamma=0,
    seed=None,
    run_exact_constraint=False,
    exact_constr_value=None,
):
    """
    Run optimizations for the four different scenario with a base field and a constraining field (e.g., F_S, F_SL, max L_S, min L_S)
    Parameters
    ----------
    base_var_name : str
        Identifier for the primary target field.
    PD_base_SensMatrix, PD_base_CurvMatrix : numpy.ndarray
        Baseline sensitivity (Jacobian) and curvature (Hessian) matrices for the primary field.
    F_base_SensMatrix, F_base_CurvMatrix : numpy.ndarray
        Future sensitivity and curvature matrices for the primary field.
    constr_var_name : str
        Identifier for the secondary constraint field.
    PD_constr_SensMatrix, PD_constr_CurvMatrix : numpy.ndarray
        Baseline sensitivity and curvature matrices for the constraint field.
    F_constr_SensMatrix, F_constr_CurvMatrix : numpy.ndarray
        Future sensitivity and curvature matrices for the constraint field.
    normlzd_param_bounds : sequence of tuples
        Optimization bounds for each normalized parameter in the target space.
    combined_future : bool, optional
        If True, stacks the projected base and constraint matrices into a joint target (default is False).
    eps : float, optional
        Numerical stability constant (default is 1e-4).
    use_grad : bool, optional
        If True, utilizes analytical gradients for the optimization (default is True).
    reg_gamma : float, optional
        L2 regularization penalty weight applied to the objective function (default is 0).
    seed : int, optional
        Random seed for the stochastic global optimizer.

    Returns
    -------
    dict
        Dictionary mapping specific optimization scenarios to tuples containing the optimal parameter
        vector and the maximized objective value.
    """

    if combined_future:
        Future_SensMatrix = np.vstack((F_base_SensMatrix, F_constr_SensMatrix))
        Future_CurvMatrix = np.vstack((F_base_CurvMatrix, F_constr_CurvMatrix))
        future_var_name = base_var_name + constr_var_name
    else:
        Future_SensMatrix = F_base_SensMatrix
        Future_CurvMatrix = F_base_CurvMatrix
        future_var_name = base_var_name

    combined_PD_SensMatrix = np.vstack((PD_base_SensMatrix, PD_constr_SensMatrix))
    combined_PD_CurvMatrix = np.vstack((PD_base_CurvMatrix, PD_constr_CurvMatrix))

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
            f"Maximizing {future_var_name} future with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_F_{short_base_name}"] = maximize_constr_problem(
            base_SensMatrix=PD_base_SensMatrix,
            base_CurvMatrix=PD_base_CurvMatrix,
            numerator_SensMatrix=Future_SensMatrix,
            numerator_CurvMatrix=Future_CurvMatrix,
            constr_value=exact_constr_value,
            denominator_SensMatrix=None,
            denominator_CurvMatrix=None,
            bounds=normlzd_param_bounds,
            eps=eps,
            use_grad=use_grad,
            reg_gamma=reg_gamma,
            seed=seed,
        )

        print(
            f"Maximizing ratio ({future_var_name} / {constr_var_name}) with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_F_{short_base_name}{short_constr_name}"] = (
            maximize_constr_problem(
                base_SensMatrix=PD_base_SensMatrix,
                base_CurvMatrix=PD_base_CurvMatrix,
                numerator_SensMatrix=Future_SensMatrix,
                numerator_CurvMatrix=Future_CurvMatrix,
                constr_value=exact_constr_value,
                denominator_SensMatrix=PD_constr_SensMatrix,
                denominator_CurvMatrix=PD_constr_CurvMatrix,
                bounds=normlzd_param_bounds,
                eps=eps,
                use_grad=use_grad,
                reg_gamma=reg_gamma,
                seed=seed,
            )
        )

        print(
            f"Maximizing {constr_var_name} present-day with {base_var_name} constrained to {exact_constr_value}"
        )
        results[f"res_max_{short_constr_name}_{short_base_name}"] = (
            maximize_constr_problem(
                base_SensMatrix=PD_base_SensMatrix,
                base_CurvMatrix=PD_base_CurvMatrix,
                numerator_SensMatrix=PD_constr_SensMatrix,
                numerator_CurvMatrix=PD_constr_CurvMatrix,
                constr_value=exact_constr_value,
                denominator_SensMatrix=None,
                denominator_CurvMatrix=None,
                bounds=normlzd_param_bounds,
                eps=eps,
                use_grad=use_grad,
                reg_gamma=reg_gamma,
                seed=seed,
            )
        )
    else:

        # optimize for future over base
        print(f"Maximizing {future_var_name} future over {base_var_name} present-day ")
        results[f"res_max_F_{short_base_name}"] = maximize_ratio(
            numerator_SensMatrix=Future_SensMatrix,
            numerator_CurvMatrix=Future_CurvMatrix,
            denominator_SensMatrix=PD_base_SensMatrix,
            denominator_CurvMatrix=PD_base_CurvMatrix,
            bounds=normlzd_param_bounds,
            eps=eps,
            use_grad=use_grad,
            reg_gamma=reg_gamma,
            seed=seed,
        )

        # optimize for future over restricted base
        print(
            f"Maximizing {future_var_name} future over {base_var_name} and {constr_var_name} present-day "
        )
        results[f"res_max_F_{short_base_name}{short_constr_name}"] = maximize_ratio(
            numerator_SensMatrix=Future_SensMatrix,
            numerator_CurvMatrix=Future_CurvMatrix,
            denominator_SensMatrix=combined_PD_SensMatrix,
            denominator_CurvMatrix=combined_PD_CurvMatrix,
            bounds=normlzd_param_bounds,
            eps=eps,
            use_grad=use_grad,
            reg_gamma=reg_gamma,
            seed=seed,
        )

        # optimize for constraint over base
        print(
            f"Maximizing present-day {constr_var_name} over present-day {base_var_name}"
        )
        results[f"res_max_{short_constr_name}_{short_base_name}"] = maximize_ratio(
            numerator_SensMatrix=PD_constr_SensMatrix,
            numerator_CurvMatrix=PD_constr_CurvMatrix,
            denominator_SensMatrix=PD_base_SensMatrix,
            denominator_CurvMatrix=PD_base_CurvMatrix,
            bounds=normlzd_param_bounds,
            eps=eps,
            use_grad=use_grad,
            reg_gamma=reg_gamma,
            seed=seed,
        )

        # optimize for base over constraint
        print(
            f"Minimizing present-day {constr_var_name} over present-day {base_var_name}"
        )
        temp_res = maximize_ratio(
            numerator_SensMatrix=PD_base_SensMatrix,
            numerator_CurvMatrix=PD_base_CurvMatrix,
            denominator_SensMatrix=PD_constr_SensMatrix,
            denominator_CurvMatrix=PD_constr_CurvMatrix,
            bounds=normlzd_param_bounds,
            eps=eps,
            use_grad=use_grad,
            reg_gamma=reg_gamma,
            seed=seed,
        )

        results[f"res_min_{short_constr_name}_{short_base_name}"] = (
            temp_res[0],
            1 / temp_res[1],
        )

    return results


def normalize_metrics_data(metrics_data, default_data, global_averages):
    """
    Standardizes output metrics by their respective global observed average and default result.

    Parameters
    ----------
    metrics_data : numpy.ndarray
        1D array of flattened spatial data.
    default_data : numpy.ndarray
        1D array of flattened default data.
    global_averages : numpy.ndarray
        1D array of reference averages used as weighting factors.

    Returns
    -------
    numpy.ndarray
        Weighted, dimensionless differences from the baseline.
    """
    length = metrics_data.shape[0] // len(global_averages)

    weights = np.repeat(np.abs(global_averages), length)

    return (metrics_data - default_data) / weights


def denormalize_metrics_data(normalized_data, default_data, global_averages):
    """
    Reverts standardized data back to its original physical units.

    Parameters
    ----------
    normalized_data : numpy.ndarray
        1D array of normalized data.
    default_data : numpy.ndarray
        1D array of default data in physical units.
    global_averages : numpy.ndarray
        1D array of reference averages used as weights.

    Returns
    -------
    numpy.ndarray
        Data projected back into absolute units.
    """
    weights = np.ones_like(normalized_data)

    length = normalized_data.shape[0] // len(global_averages)

    for i, global_average in enumerate(global_averages):
        weights[i * length : (i + 1) * length] = np.abs(global_average)

    return (normalized_data * weights) + default_data


def get_H_at_dp(SensMatrix, CurvMatrix, dp):
    """
    Computes the approximated Hessian matrix
    at a specific evaluation dp.

    Parameters
    ----------
    SensMatrix : numpy.ndarray
        Sensitivity matrix (M, N).
    CurvMatrix : numpy.ndarray
        Curvature matrix (M, N).
    dp : numpy.ndarray
        Evaulation dp (N,).

    Returns
    -------
    numpy.ndarray
        2D array of shape (N, N) representing the localized Hessian approximation.
    """

    J = SensMatrix + CurvMatrix * dp
    return J.T @ J


def calc_A_opt_metrics(H_base, H_constr):
    """
    Calculates A-optimality metrics to quantify the restriction
    imposed by secondary constraints.

    Parameters
    ----------
    H_base : numpy.ndarray
        Hessian matrix of the unconstrained base system (N, N).
    H_constr : numpy.ndarray
        Hessian matrix of the constrained system (N, N).

    Returns
    -------
    tuple
        - R_scale (float): Volumetric scaling ratio based on matrix determinants.
        - E_RLS (float): Expected ratio (A-optimality).
        - R_shape (float):Shape alignment between base and constrained nullspaces.
        - A_opt_over_n_slope (float): Normalized slope of the A-optimality metric.
    """
    H_base_inv = np.linalg.pinv(H_base)

    R_scale = (np.linalg.det(H_constr) / np.linalg.det(H_base)) ** (1 / H_base.shape[0])

    A_opt = np.trace(H_base_inv @ H_constr)

    E_RLS = A_opt / H_base.shape[0]

    A_opt_over_n_slope = 1 / (1 + E_RLS)

    R_shape = E_RLS / (R_scale)

    return R_scale, E_RLS, R_shape, A_opt_over_n_slope

def calc_all_A_opt_metrics(all_optimizations,used_fields, base_field, all_fields_data, paramset_idx):
    """
    Calculate the A-optimality metrics for all given fields

    Parameters
    ----------
    all_optimizations: dict
        Nested dictionary mapping field names to their optimization results.
    used_fields : list of str
        Names of all fields being evaluated (e.g. LWCF).
    base_field: str
        Name of the base field (e.g. SWCF).
    all_fields_data : dict
        Dictionary mapping field names to their respective sensitivity and curvature matrices.
    paramset_idx: int
        Index of where in all_optimizations the reference parameterset about which the linearisation is done is stored.

    Returns
    -------
    tuple
        - Scales_metrics (list of floats): Volumetric scaling ratio based on matrix determinants.
        - E_RLS (list of floats): Expected ratios (A-optimality), with one for each field used in optimization.
        - Shapes_metrics (list of floats): Shape alignment between base and constrained nullspaces.
        - A_opt_over_n_slope (list of floats): Normalized slope of the A-optimality metric.
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


        ConstrSensMatrix, ConstrCurvMatrix = all_fields_data[field][:2]
        H_Base = get_H_at_dp(BaseSensMatrix, BaseCurvMatrix, constrained_parameter_set)
        H_Constr = get_H_at_dp(ConstrSensMatrix, ConstrCurvMatrix, constrained_parameter_set)


        R_scale, E_RLS, R_shape, A_opt_over_n = calc_A_opt_metrics(H_Base, H_Constr)
        

        Scales_metrics.append(R_scale)
        Shapes_metrics.append(R_shape)
        E_RLS_allfields.append(E_RLS)
        A_opt_over_n_slopes.append(A_opt_over_n)
    
    return Scales_metrics, E_RLS_allfields, Shapes_metrics, A_opt_over_n_slopes


def calc_E_RR(H_base, H_constr):
    """
    Calculates the expected ratio (E_RR) using the trace of the
    Hessian matrices.

    Parameters
    ----------
    H_base : numpy.ndarray
        Hessian matrix of the unconstrained base system (N, N).
    H_constr : numpy.ndarray
        Hessian matrix of the constrained system (N, N).

    Returns
    -------
    float
        The expected average ratio.
    """
    trace = np.linalg.trace(
        np.linalg.pinv(np.eye(H_base.shape[0]) + np.linalg.pinv(H_base) @ H_constr)
    )

    return trace / H_base.shape[0]

def calc_all_E_RR(all_optimizations, used_fields, base_field, all_fields_data, paramset_idx):
    """
    Calculates the expected ratio (E_RR) using the trace of the
    Hessian matrices.

    Parameters
    ----------
    all_optimizations: dict
        Nested dictionary mapping field names to their optimization results.
    used_fields : list of str
        Names of all fields being evaluated (e.g. LWCF).
    base_field: str
        Name of the base field.
    all_fields_data : dict
        Dictionary mapping field names to their respective sensitivity and curvature matrices.
    paramset_idx: int
        Index of where in all_optimizations the reference parameterset about which the linearisation is done is stored.
    
    Returns
    -------
    list
        E_RR values for all fields
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
        H_B = get_H_at_dp(BaseSensMatrix, BaseCurvMatrix, constrained_parameter_set)
        H_C = get_H_at_dp(ConstrSensMatrix, ConstrCurvMatrix, constrained_parameter_set)

        E_RR_values.append(calc_E_RR(H_B, H_C))

    return E_RR_values

def normalize_params(params, default_params):
    """
    Converts absolute parameter values into perturbations relative to the default.

    Parameters
    ----------
    params : numpy.ndarray
        Array of absolute parameters.
    default_params : numpy.ndarray
        Array of default parameters.

    Returns
    -------
    numpy.ndarray
        Normalized parameter perturbations.
    """
    return (params - default_params) / np.abs(default_params)


def denormalize_params(normlzd_params, default_params):
    """
    Converts parameter perturbations back into absolute values.

    Parameters
    ----------
    normlzd_params : numpy.ndarray
        Normalized parameter perturbations.
    default_params : numpy.ndarray
        Array of default parameters.

    Returns
    -------
    numpy.ndarray
        Absolute parameter values.
    """
    return normlzd_params * np.abs(default_params) + default_params
