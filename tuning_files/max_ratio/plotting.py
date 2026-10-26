from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import pandas as pd

import sys

import seaborn as sns



project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from optimization import (calc_all_A_opt_metrics, get_H_at_dp, calc_E_RR)



from create_nonbootstrap_figs import createMapPanel


def latexify_parameterset_name(name):
    """
    Parses a formatted optimization scenario identifier and converts it into a 
    LaTeX-formatted mathematical string.

    Parameters
    ----------
    name : str
        The input string formatted with underscores separating four components: 
        `<prefix>_<optimization>_<numerator>_<denominator>` (e.g., 'dp_max_F_SL').

    Returns
    -------
    str
        LaTeX string formatted as `$<prefix>_{<optimization>}^{(<numerator>,<denominator>)}$`.
    """
    parts = name.split('_')

    dp, optimization, numerator, denominator = parts

    return f"${dp}_{{{optimization}}}^{{({numerator},{denominator})}}$"    

def create_parameter_bar_chart(parametersets, parameter_names, parameter_set_names, ax =None):
    """
    Generates a horizontal bar chart comparing normalized parameter perturbations 
    across multiple optimized parameter sets.

    Parameters
    ----------
    parametersets : numpy.ndarray
        2D array of normalized parameter perturbation vectors of shape 
        (num_sets, num_params).
    parameter_names : sequence of str
        Labels for the individual tunable parameters (y-axis).
    parameter_set_names : list of str
        LaTeX-formatted labels for the different optimization runs (legend).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are 
        created (default is None).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the bar chart.
    """

    if ax == None:
        fig, ax = plt.subplots( figsize=(16,12))

    df = pd.DataFrame(parametersets.T, index=parameter_names,columns = parameter_set_names).iloc[::-1]

    ax = df.plot.barh(ax=ax,
                      width=0.65, fontsize=16)

    ax.set_xlabel("Normalized parameter value", fontsize=16)
    ax.yaxis.grid(False)

    ax.axvline(0, color='black', linewidth=1)

    ax.legend(
        title="Parameter sets", 
        fontsize=20,
        title_fontsize=20,
        loc='lower right', 
        frameon=True
    )

    ax.set_xlim(-1,1)

    return ax



def create_map_plots(
    all_optimizations: dict[str, dict[str, np.ndarray]], 
    fields_idxs: list[int], 
    used_fields: list[str], 
    base_field: str, 
    all_models: dict[str, Callable], 
    sst4k_models: dict[str, Callable], 
    global_averages_obs: np.ndarray, 
    box_size: int, 
    outdir: Path, 
    plot_width: int = 600, 
    num_sets_to_plot: int = 2
):
    """
    Generates and saves global regional map plots of present-day and future 
    emulated field perturbations for the optimized parameter sets.

    Colorbar limits are synchronized across all plotted parameter sets for a 
    given field to allow direct visual comparison of perturbation damping and 
    spatial mode-switching.

    Parameters
    ----------
    all_optimizations : dict of {str: dict}
        Nested dictionary containing the optimization results for each secondary 
        constraint field.
    fields_idxs : list of int
        Indices mapping `used_fields` to `ALL_FIELDS` and `global_averages_obs`.
    used_fields : list of str
        List of all climate fields evaluated in the run.
    base_field : str
        Name of the primary constraint and target field (e.g., 'SWCF').
    all_models : dict of {str: Callable}
        Present-day emulator callables for each field.
    sst4k_models : dict of {str: Callable}
        Future (+4K SST) emulator callables for each field.
    global_averages_obs : numpy.ndarray
        1D array of global reference averages used to rescale normalized outputs 
        back to physical units.
    box_size : int
        Regional grid box resolution in degrees.
    outdir : pathlib.Path
        Output directory where the generated PDF map figures are saved.
    plot_width : int, optional
        Width of the generated Plotly map panels in pixels (default is 600).
    num_sets_to_plot : int, optional
        Number of optimization scenarios per field to plot (default is 2, 
        corresponding to `dp_max^(F,S)` and `dp_max^(F,SL)`).

    Returns
    -------
    None
    """
    base_plot_done = False
    base_field_idx = fields_idxs[used_fields.index(base_field)]
    
    base_PD_model = all_models[base_field]
    base_F_model = sst4k_models[base_field]

    all_param_sets = []
    for field_dict in all_optimizations.values():
        for result in list(field_dict.values())[:num_sets_to_plot]:
            all_param_sets.append(result[0])

    global_base_min = np.inf
    global_base_max = -np.inf
    for p_set in all_param_sets:
        pd_eval = base_PD_model(p_set) * np.abs(global_averages_obs[base_field_idx])
        f_eval = base_F_model(p_set) * np.abs(global_averages_obs[base_field_idx])


        global_base_min = min(global_base_min, np.min(pd_eval), np.min(f_eval))
        global_base_max = max(global_base_max, np.max(pd_eval), np.max(f_eval))

    for idx, field in enumerate(used_fields):
        if field == base_field:
            continue

        parameter_sets_names = list(all_optimizations[field].keys())[:num_sets_to_plot]
        parameter_sets = np.array([result[0] for result in list(all_optimizations[field].values())[:num_sets_to_plot]])
        
        constr_PD_model = all_models[field]
        constr_F_model = sst4k_models[field]

        for set_idx, parameter_set in enumerate(parameter_sets):
            param_name = parameter_sets_names[set_idx].replace("res", "dp")
            latex_name = latexify_parameterset_name(param_name)
            math_content = latex_name.replace('$', '')

            global_constr_min = np.inf
            global_constr_max = -np.inf
            for p_set in all_param_sets:
                pd_eval = constr_PD_model(p_set) * np.abs(global_averages_obs[fields_idxs[idx]])
                f_eval = constr_F_model(p_set) * np.abs(global_averages_obs[fields_idxs[idx]])

                global_constr_min = min(global_constr_min, np.min(pd_eval), np.min(f_eval))
                global_constr_max = max(global_constr_max, np.max(pd_eval), np.max(f_eval))


            if set_idx != 0 or not base_plot_done:
                if set_idx == 0:
                    base_plot_done = True
                
                normalized_base_PD_plot_data = base_PD_model(parameter_set)
                normalized_base_F_plot_data = base_F_model(parameter_set)

                diff_base_PD_plot_data = normalized_base_PD_plot_data * np.abs(global_averages_obs[base_field_idx])
                diff_base_F_plot_data = normalized_base_F_plot_data * np.abs(global_averages_obs[base_field_idx])

                base_PD_plot = createMapPanel(diff_base_PD_plot_data, plot_width, rf"$\text{{{base_field} present-day perturbation for }} {math_content}$", box_size, minField=global_base_min, maxField=global_base_max)
                base_F_plot = createMapPanel(diff_base_F_plot_data, plot_width, rf"$\text{{{base_field} future perturbation for }} {math_content}$", box_size, minField=global_base_min, maxField=global_base_max)

                base_PD_plot.write_image(outdir / f"Diff_Map_PD_{param_name}_{base_field}.pdf")
                base_F_plot.write_image(outdir / f"Diff_Map_F_{param_name}_{base_field}.pdf")

            normalized_constr_PD_plot_data = constr_PD_model(parameter_set)
            normalized_constr_F_plot_data = constr_F_model(parameter_set)

            diff_constr_PD_plot_data = normalized_constr_PD_plot_data * np.abs(global_averages_obs[fields_idxs[idx]])
            diff_constr_F_plot_data = normalized_constr_F_plot_data * np.abs(global_averages_obs[fields_idxs[idx]])

            constr_PD_plot = createMapPanel(diff_constr_PD_plot_data, plot_width, rf"$\text{{{field} present-day perturbation for }} {math_content}$", box_size, minField=global_constr_min, maxField=global_constr_max)
            constr_F_plot = createMapPanel(diff_constr_F_plot_data, plot_width, rf"$\text{{{field} future perturbation for }} {math_content}$", box_size, minField=global_constr_min, maxField=global_constr_max)

            constr_PD_plot.write_image(outdir / f"Diff_Map_PD_{param_name}_{field}.pdf")
            constr_F_plot.write_image(outdir / f"Diff_Map_F_{param_name}_{field}.pdf")

            


def make_scatterplot(
    results: dict, 
    base_var_name: str, 
    base_PD_model: Callable, 
    base_F_model: Callable,
    constr_var_name: str, 
    constr_PD_model: Callable, 
    constr_F_model: Callable, 
    combined_future: bool = False, 
    additional_params: np.ndarray = None, 
    e3sm_PPE_results: tuple = None, 
    e3sm_optimized_results: tuple = None, 
    include_average_constr: bool = True, 
    constrained_opt: bool = False,
    constr_value: float = None, 
    ax: plt.Axes = None,
    # Optional matrices for E_RR calculation (Quadtune only)
    PD_base_SensMatrix=None, PD_base_CurvMatrix=None,
    PD_constr_SensMatrix=None, PD_constr_CurvMatrix=None,
    eps=1e-4,
    # Optional inverse covariance matrices
    inv_gamma_base=None,      
    inv_gamma_constr=None,    
    inv_gamma_combined=None   
):
    """
    Generates a diagnostic scatterplot evaluating initial constraint strength 
    and constraint evasion when a secondary constraint is introduced.

    Plots the double-constraint variance ratio against the single-constraint 
    variance ratio (both normalized by the maximum single-constraint ratio), 
    along with optional E3SM PPE members, verification runs, and the analytical 
    expected ratio slope E_RR.

    Parameters
    ----------
    results : dict
        Optimization results dictionary for the current secondary constraint field.
    base_var_name : str
        Name of the primary baseline constraint field (e.g., 'SWCF').
    base_PD_model, base_F_model : Callable
        Present-day and future emulator callables for the primary field.
    constr_var_name : str
        Name of the secondary constraint field (e.g., 'LWCF', 'PRECT', 'TMQ').
    constr_PD_model, constr_F_model : Callable
        Present-day and future emulator callables for the secondary field.
    combined_future : bool, optional
        If True, includes both primary and secondary fields in the future 
        numerator variance (default is False).
    additional_params : numpy.ndarray, optional
        2D array of additional parameter perturbation vectors to evaluate and 
        scatter via the emulator (default is None).
    e3sm_PPE_results : tuple or list of numpy.ndarray, optional
        Tuple of (PPE_deltas, PPE_deltas_sst4k) containing pre-computed E3SM 
        one-at-a-time PPE regional deviations (default is None).
    e3sm_optimized_results : tuple or numpy.ndarray, optional
        Tuple of (deltas, deltas_sst4k) from actual E3SM verification runs at 
        the optimized parameter sets (default is None).
    include_average_constr : bool, optional
        If True, plots the analytical expected ratio of ratios (E_RR) as a dotted 
        reference line (default is True).
    constrained_opt : bool, optional
        If True, evaluates ratios using the fixed primary constraint threshold 
        `constr_value` in the denominator (default is False).
    constr_value : float, optional
        Primary constraint threshold value used when `constrained_opt` is True 
        (default is None).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure and axes are 
        created (default is None).
    PD_base_SensMatrix, PD_base_CurvMatrix : numpy.ndarray, optional
        Present-day sensitivity and curvature matrices for the primary field, 
        required to compute E_RR (default is None).
    PD_constr_SensMatrix, PD_constr_CurvMatrix : numpy.ndarray, optional
        Present-day sensitivity and curvature matrices for the secondary field, 
        required to compute E_RR (default is None).
    eps : float, optional
        Small regularization constant added to denominators (default is 1e-4).
    inv_gamma_base : numpy.ndarray, optional
        Inverse noise covariance matrix for the primary field (default is None).
    inv_gamma_constr : numpy.ndarray, optional
        Inverse noise covariance matrix for the secondary field (default is None).
    inv_gamma_combined : numpy.ndarray, optional
        Block-diagonal inverse noise covariance matrix for the combined fields 
        (default is None).

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object containing the constraint scatterplot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    short_constr_name = 'Q' if constr_var_name == "TMQ" else constr_var_name[0]
    short_base_name = 'Q' if base_var_name == "TMQ" else base_var_name[0]


    def F_base_fun(dp):
        vec = base_F_model(dp)
        return vec.T @ inv_gamma_base @ vec if inv_gamma_base is not None else np.sum(vec**2)

    def F_combined_fun(dp):
        vec_base = base_F_model(dp)
        vec_constr = constr_F_model(dp)
        if inv_gamma_combined is not None:
            combined_vec = np.hstack((vec_base, vec_constr))
            return combined_vec.T @ inv_gamma_combined @ combined_vec
        else:
            return np.sum(vec_base**2) + np.sum(vec_constr**2)

    def PD_base_fun(dp):
        vec = base_PD_model(dp)
        return vec.T @ inv_gamma_base @ vec if inv_gamma_base is not None else np.sum(vec**2)

    def PD_combined_fun(dp):
        vec_base = base_PD_model(dp)
        vec_constr = constr_PD_model(dp)
        if inv_gamma_combined is not None:
            combined_vec = np.hstack((vec_base, vec_constr))
            return combined_vec.T @ inv_gamma_combined @ combined_vec
        else:
            return np.sum(vec_base**2) + np.sum(vec_constr**2)

    def PD_constr_fun(dp):
        vec = constr_PD_model(dp)
        return vec.T @ inv_gamma_constr @ vec if inv_gamma_constr is not None else np.sum(vec**2)


    if constrained_opt:
        if combined_future:
            R_F_B_fun = lambda dp: F_combined_fun(dp) / (constr_value + eps)
            R_F_BC_fun = lambda dp: F_combined_fun(dp) / (PD_constr_fun(dp) + constr_value + eps)
        else:
            R_F_B_fun = lambda dp: F_base_fun(dp) / (constr_value + eps)
            R_F_BC_fun = lambda dp: F_base_fun(dp) / (PD_constr_fun(dp) + constr_value + eps)

        R_max_F_B = R_F_B_fun(results[f"res_max_F_{short_base_name}"][0])
    else:
        if combined_future:
            R_F_B_fun = lambda dp: F_combined_fun(dp) / (PD_base_fun(dp) + eps)
            R_F_BC_fun = lambda dp: F_combined_fun(dp) / (PD_combined_fun(dp) + eps)
        else:
            R_F_B_fun = lambda dp: F_base_fun(dp) / (PD_base_fun(dp) + eps)
            R_F_BC_fun = lambda dp: F_base_fun(dp) / (PD_combined_fun(dp) + eps)
        R_max_F_B = results[f"res_max_F_{short_base_name}"][1]


    dp_max_F_B = results[f"res_max_F_{short_base_name}"][0]
    dp_max_F_BC = results[f"res_max_F_{short_base_name}{short_constr_name}"][0]
    dp_max_C_B = results[f"res_max_{short_constr_name}_{short_base_name}"][0]

    initial_x = R_F_B_fun(dp_max_F_B) / R_max_F_B
    initial_y = R_F_BC_fun(dp_max_F_B) / R_max_F_B
    initial_name = latexify_parameterset_name(f"res_max_F_{short_base_name}".replace("res","dp"))

    final_x = R_F_B_fun(dp_max_F_BC) / R_max_F_B
    final_y = R_F_BC_fun(dp_max_F_BC) / R_max_F_B
    final_name = latexify_parameterset_name(f"res_max_F_{short_base_name}{short_constr_name}".replace("res","dp"))


    sns.scatterplot(x=[initial_x], y=[initial_y], label=initial_name, color="red", s=200, linewidth=2, ax=ax)
    sns.scatterplot(x=[final_x], y=[final_y], label=final_name, color="red", marker="x", s=200, linewidth=2, ax=ax)
    ax.annotate("", xy=(final_x, final_y), xytext=(initial_x, initial_y), arrowprops=dict(arrowstyle="->,head_width=0.4", color="black", lw=2, shrinkA=12, shrinkB=12))


    if additional_params is not None:
        PPE_R_F_B_results = np.array([R_F_B_fun(dp) for dp in additional_params])
        PPE_R_F_BC_results = np.array([R_F_BC_fun(dp) for dp in additional_params])
        sns.scatterplot(x=PPE_R_F_B_results/R_max_F_B, y=PPE_R_F_BC_results/R_max_F_B, label="PPE samples", color="purple", alpha=0.5, ax=ax)


    if e3sm_PPE_results is not None and e3sm_PPE_results[0] is not None:
        PPE_deltas, PPE_deltas_sst4k = e3sm_PPE_results
        

        if inv_gamma_base is not None:
            base_pd_vals = np.einsum('ij,jk,ik->i', PPE_deltas[:, 0], inv_gamma_base, PPE_deltas[:, 0])
            base_f_vals = np.einsum('ij,jk,ik->i', PPE_deltas_sst4k[:, 0], inv_gamma_base, PPE_deltas_sst4k[:, 0])
        else:
            base_pd_vals = np.sum(PPE_deltas[:, 0]**2, axis=1)
            base_f_vals = np.sum(PPE_deltas_sst4k[:, 0]**2, axis=1)


        if inv_gamma_combined is not None:
            combined_pd_vecs = np.hstack((PPE_deltas[:, 0], PPE_deltas[:, 1]))
            combined_pd_vals = np.einsum('ij,jk,ik->i', combined_pd_vecs, inv_gamma_combined, combined_pd_vecs)
        else:
            combined_pd_vals = np.sum(PPE_deltas[:, 0]**2, axis=1) + np.sum(PPE_deltas[:, 1]**2, axis=1)
            

        if combined_future:
            if inv_gamma_combined is not None:
                combined_f_vecs = np.hstack((PPE_deltas_sst4k[:, 0], PPE_deltas_sst4k[:, 1]))
                combined_f_vals = np.einsum('ij,jk,ik->i', combined_f_vecs, inv_gamma_combined, combined_f_vecs)
            else:
                combined_f_vals = np.sum(PPE_deltas_sst4k[:, 0]**2, axis=1) + np.sum(PPE_deltas_sst4k[:, 1]**2, axis=1)

  
        if combined_future:
            PPE_F_B_E3SM_ratios = combined_f_vals / base_pd_vals
            PPE_F_BC_E3SM_ratios = combined_f_vals / combined_pd_vals
        else:
            PPE_F_B_E3SM_ratios = base_f_vals / base_pd_vals
            PPE_F_BC_E3SM_ratios = base_f_vals / combined_pd_vals
            
        sns.scatterplot(x=PPE_F_B_E3SM_ratios/R_max_F_B, y=PPE_F_BC_E3SM_ratios/R_max_F_B, label="PPE samples", color="purple", s=100, alpha=0.5, ax=ax)

    if e3sm_optimized_results is not None:
        PPE_deltas, PPE_deltas_sst4k = e3sm_optimized_results
        
        if inv_gamma_base is not None:
            base_pd_vals = np.einsum('ij,jk,ik->i', PPE_deltas[:, 0], inv_gamma_base, PPE_deltas[:, 0])
            base_f_vals = np.einsum('ij,jk,ik->i', PPE_deltas_sst4k[:, 0], inv_gamma_base, PPE_deltas_sst4k[:, 0])
        else:
            base_pd_vals = np.sum(PPE_deltas[:, 0]**2, axis=1)
            base_f_vals = np.sum(PPE_deltas_sst4k[:, 0]**2, axis=1)

        
        if inv_gamma_combined is not None:
            combined_pd_vecs = np.hstack((PPE_deltas[:, 0], PPE_deltas[:, 1]))
            combined_pd_vals = np.einsum('ij,jk,ik->i', combined_pd_vecs, inv_gamma_combined, combined_pd_vecs)
        else:
            combined_pd_vals = np.sum(PPE_deltas[:, 0]**2, axis=1) + np.sum(PPE_deltas[:, 1]**2, axis=1)
            
        
        if combined_future:
            if inv_gamma_combined is not None:
                combined_f_vecs = np.hstack((PPE_deltas_sst4k[:, 0], PPE_deltas_sst4k[:, 1]))
                combined_f_vals = np.einsum('ij,jk,ik->i', combined_f_vecs, inv_gamma_combined, combined_f_vecs)
            else:
                combined_f_vals = np.sum(PPE_deltas_sst4k[:, 0]**2, axis=1) + np.sum(PPE_deltas_sst4k[:, 1]**2, axis=1)

        
        if combined_future:
            PPE_F_B_E3SM_ratios = combined_f_vals / base_pd_vals
            PPE_F_BC_E3SM_ratios = combined_f_vals / combined_pd_vals
        else:
            PPE_F_B_E3SM_ratios = base_f_vals / base_pd_vals
            PPE_F_BC_E3SM_ratios = base_f_vals / combined_pd_vals
            
        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[0]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[0]/R_max_F_B], label=rf'E3SM $dp_{{max}}^{{F,{short_base_name}}}$', color="green", s=100, linewidth=2, ax=ax)
        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[1]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[1]/R_max_F_B], label=rf'E3SM $dp_{{max}}^{{F,{short_base_name}{short_constr_name}}}$', color="green", marker="x", s=100, linewidth=2, ax=ax)


    if include_average_constr:
        if PD_base_SensMatrix is not None and PD_base_CurvMatrix is not None:
            dp_ref = dp_max_C_B
            H_B = get_H_at_dp(PD_base_SensMatrix, PD_base_CurvMatrix, dp_ref, inv_gamma=inv_gamma_base)
            H_C = get_H_at_dp(PD_constr_SensMatrix, PD_constr_CurvMatrix, dp_ref, inv_gamma=inv_gamma_constr)
            E_RR = calc_E_RR(H_B, H_C)
            sns.lineplot(x=[0,1], y=[0,E_RR], label=r"Average ratio ($\mathcal{E}_{RR}$)", linestyle=":", linewidth=4, color="cyan", ax=ax)

    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(fr'$\frac{{R^{{(F,{short_base_name})}}}}{{R^{{(F,{short_base_name})}}_{{max}}}}$', fontsize=24)
    ax.set_ylabel(fr'$\frac{{R^{{(F,{short_base_name}{short_constr_name})}}}}{{R^{{(F,{short_base_name})}}_{{max}}}}$', fontsize=24, rotation=0, ha='right', va='center')
    ax.yaxis.set_label_coords(-0.05, 0.50)

    sns.lineplot(x=[0,1], y=[0,1], label="Upper bound", color="black", linestyle="--", alpha=0.5, ax=ax)
    ax.legend(framealpha=0.9, fontsize=20, title_fontsize=20)
    
    return ax



def create_cartoon_scatterplot(ax=None, scatter_size=200, text_size=12, arrow_pad = 12, arrow_size =4, head_width =0.6):
    """
    Generates a conceptual, annotated scatterplot to illustrate theoretical 
    optimization scenarios and boundary behaviors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot (default is None).
    scatter_size : int, optional
        Base size for the scatter markers (default is 200).
    text_size : int, optional
        Font size for the annotations and labels (default is 12).
    arrow_pad, arrow_size, head_width : float or int, optional
        Styling parameters for the arrows showcasing the shift from initial 
        to final parameter points.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object.
    """
    if ax is None:
        fig, ax  = plt.subplots(figsize=(6,6))
   
   
    sns.lineplot(x=[0,1],y=[0,1], label="Upper bound", color="black", linestyle="--",alpha=0.5, ax=ax)
    ax.set_aspect('equal', adjustable='box')

    ax.text(
    0.3, 0.3,                          
    "Upper bound for all points",            
    rotation=45,                       
    rotation_mode='anchor',            
    ha='center',                       
    va='center',                       
    color='black',                      
    fontsize=text_size,
    bbox=dict(facecolor='white', edgecolor='none', pad=3) 
)


    #Scenario 1
    sns.scatterplot(x = [0.7], y = [0.05],color="green", s=scatter_size*1.6, marker="X",ax=ax)
    ax.text(x=0.7, y=0.1,s="Guaranteed strong 2nd constraint",c="green", fontsize = text_size,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2), horizontalalignment='center')


    #Scenario 2
    sns.scatterplot(x = [1], y=[0.94], color="blue", s=scatter_size, ax=ax)
    ax.text(x=1,y=1,s="Guaranteed weak 2nd constraint", c="blue",fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')

    #Scenario 3
    sns.scatterplot(x=[1], y =[0.7], color="tab:orange",s=scatter_size, alpha=0.8, ax=ax)
    sns.scatterplot(x=[1], y =[0.7], color="tab:orange",s=scatter_size*1.6, alpha=0.5, marker="X", ax=ax)

    ax.text(x=1,y=0.75,s="Initial and final point coincide", c="tab:orange", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')

    #Scenario 4a
    sns.scatterplot(x=[1], y=[0.2], color="red", s=scatter_size, ax=ax)
    sns.scatterplot(x=[0.85], y=[0.2], color="red", s=scatter_size*1.6,marker="X",ax=ax)

    ax.annotate("", xy=(0.85,0.2), xytext = (1,0.2), arrowprops=dict(arrowstyle=f"->,head_width={head_width}",color="red",lw=arrow_size,shrinkA=arrow_pad, shrinkB=arrow_pad))


    ax.text(x=0.75,y=0.25,s="Unevaded 2nd constraint",c="red", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='center')


    #Scenario 4b
    sns.scatterplot(x=[1], y=[0.4], color="purple", s=scatter_size, ax=ax)
    sns.scatterplot(x=[1], y=[0.55], color="purple", s=scatter_size*1.6,marker="X",ax=ax)
    ax.text(x=0.95,y=0.45,s="Evaded 2nd constraint",c="purple", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')
    ax.annotate("", xy=(1,0.55), xytext = (1,0.4), arrowprops=dict(arrowstyle=f"->,head_width={head_width}",color="purple",lw=arrow_size,shrinkA=arrow_pad, shrinkB=arrow_pad))




    ax.set_title("Optimization scenarios when a second constraint is added", fontsize = text_size)

    ax.set_xlabel(r'$\frac{R^{(F,S)}}{R^{(F,S)}_{\text{max}}}$', fontsize = text_size*1.5)

    ax.set_ylabel(r'$\frac{R^{(F,SL)}}{R^{(F,S)}_{\text{max}}}$',fontsize = text_size*1.5, rotation =0,ha='right', va='center' )
    ax.yaxis.set_label_coords(-0.05, 0.50)


    ax.tick_params(axis='both', labelsize=text_size)

    marker_init = mlines.Line2D([], [], color='w', marker='o',
                                markerfacecolor='dimgrey', markeredgecolor='dimgrey',
                                markersize=9, label=r'Initial ($dp_{max}^{F,S}$)')
    marker_final = mlines.Line2D([], [], color='w', marker='X',
                                 markerfacecolor='dimgrey', markeredgecolor='dimgrey',
                                 markersize=9, label=r'Final ($dp_{max}^{F,SL}$)')

    ax.legend(handles=[marker_init, marker_final],
              title="Point Type", framealpha=0.9, fontsize=text_size, title_fontsize=text_size)

   
    return ax


def create_shape_vs_scale_plot(
    all_optimizations, base_field, used_fields, all_fields_data, 
    text_size=14, scatter_size=150, paramset_idx_to_plot=2, 
    colors=None, ax=None, inv_gamma_dict=None
):
    """
    Visualizes the decomposition of relative constraint strength into volumetric 
    scale (R_scale) and alignment shape (R_shape) factors across secondary constraints.

    Linearizes the Quadtune models about a reference parameter vector to evaluate 
    the local Gramian matrices and their generalized eigenvalues.

    Parameters
    ----------
    all_optimizations : dict
        Nested dictionary of optimization results for each field.
    base_field : str
        Name of the primary baseline constraint field (e.g., 'SWCF').
    used_fields : list of str
        List of all climate variables being evaluated.
    all_fields_data : dict
        Dictionary mapping field names to their respective sensitivity and 
        curvature matrices.
    text_size : int, optional
        Base font size for axis labels, ticks, and legend (default is 14).
    scatter_size : int, optional
        Marker size for the scatter points (default is 150).
    paramset_idx_to_plot : int, optional
        Index of the optimization result in `all_optimizations[field]` used as 
        the linearization point (default is 2, corresponding to `dp_max^(m,S)`).
    colors : list of str, optional
        Color palette to apply to the plotted secondary constraints. If None, 
        uses Seaborn's 'tab10' palette (default is None).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, creates a new figure and axes 
        (default is None).
    inv_gamma_dict : dict of {str: numpy.ndarray}, optional
        Dictionary mapping field names to their inverse noise covariance matrices 
        (default is None).

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object.
    """

    if ax is None:
        fig, ax  = plt.subplots(figsize=(8,8))
   
    fields_to_plot = used_fields.copy()
    fields_to_plot.remove(base_field)


    if colors is None:
        colors = sns.color_palette("tab10", n_colors=len(fields_to_plot))


    R_scales, _, R_shapes, _ = calc_all_A_opt_metrics(
        all_optimizations, used_fields, base_field, all_fields_data, 
        paramset_idx_to_plot, inv_gamma_dict=inv_gamma_dict
    )
    print(f"Fields to plot: {used_fields}")
    print(f"R_scales: {R_scales}")
    print(f"R_shapes: {R_shapes}")

    labels = []
    for idx, field in enumerate(fields_to_plot):
        

        constrained_parameter_set_name = list(all_optimizations[field].keys())[paramset_idx_to_plot]

        scatter_point_name = latexify_parameterset_name(constrained_parameter_set_name.replace("res","R"))

        labels.append(scatter_point_name)
        


    sns.scatterplot(x=R_scales, y=R_shapes, hue = labels, s=scatter_size,palette = colors[:len(labels)], ax=ax)

    

    ax.set_xlabel(r"$R_\text{scale}$",fontsize = text_size*3)
    ax.set_ylabel(r"$R_\text{shape}$",fontsize = text_size*3, rotation =0,ha='right', va='center')

    ax.axhline(y=1, label="lower bound \n" r"for $R_\text{shape}$", linestyle='--', linewidth = 3)
    ax.tick_params(axis='both', labelsize=text_size*2)

    ax.set_ylim(bottom=0.,top=2.2)
    ax.set_xlim(left=0.0, right = 2.2)

    ax.set_xticks([0.0,1.0,2.0])
    ax.legend(framealpha=0.9, fontsize=text_size*1.5, title_fontsize=text_size*1.5, loc='lower right')

    

    return ax