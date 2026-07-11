from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


import pandas as pd

import sys

import seaborn as sns



project_root = Path(__file__).resolve().parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from optimization import (calc_all_A_opt_metrics, evaluate_model, get_H_at_dp, evaluate_ratio, calc_A_opt_metrics, calc_E_RR)



from create_nonbootstrap_figs import createMapPanel


def latexify_parameterset_name(name):
    """
    Parses a specifically formatted identifier string and converts it into a 
    LaTeX-formatted mathematical representation.

    Parameters
    ----------
    name : str
        The input string, expected to be formatted with underscores separating 
        four components: (dp, optimization, numerator, denominator).

    Returns
    -------
    str
        The formatted LaTeX string formatted as dp_optimization^{(numerator,denominator)}
    """
    parts = name.split('_')

    dp, optimization, numerator, denominator = parts

    return f"${dp}_{{{optimization}}}^{{({numerator},{denominator})}}$"    

def create_parameter_bar_chart(parametersets, parameter_names, parameter_set_names, ax =None):
    """
    Generates a horizontal bar chart comparing normalized parameter values 
    across multiple optimized parameter sets.

    Parameters
    ----------
    parametersets : numpy.ndarray
        2D array of parameter vectors.
    parameter_names : list of str
        Labels for the individual parameters (y-axis).
    parameter_set_names : list of str
        Labels for the different optimization runs/sets (legend).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object containing the plot.
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



def create_map_plots(all_optimizations: dict[str, dict[str,np.ndarray]], fields_idxs:list[int], used_fields:list[str], base_field:str, all_fields_data : dict[str,np.ndarray], global_averages_obs: np.ndarray, box_size:int, outdir:Path, plot_width:int=600 ,num_sets_to_plot:int=2):
    """
    Generates and saves spatial maps for perturbed variables 
    using quadratic model evaluations from Quadtune.

    Computes global minimum and maximum bounds across a specified number of 
    parameter sets to ensure uniform color scaling, then evaluates and plots 
    both baseline and constrained target variables for SST and SST+4K.

    Parameters
    ----------
    all_optimizations : dict
        Nested dictionary mapping field names to their optimization results.
    fields_idxs : list of int
        Indices corresponding to the used fields within all_fields.
    used_fields : list of str
        Names of all fields being evaluated.
    base_field : str
        Name of the base field.
    all_fields_data : dict
        Dictionary mapping field names to their respective sensitivity and curvature matrices.
    global_averages_obs : numpy.ndarray
        1D array of global averages used to scale normalized data back to 
        absolute perturbation values.
    box_size : int
        Grid resolution constraint for the underlying map generation.
    outdir : pathlib.Path
        Output directory for the generated PDF files.
    plot_width : int, optional
        Width of the generated plot panels (default is 600).
    num_sets_to_plot : int, optional
        Number of parameter sets to evaluate and plot (default is 2).

    Returns
    -------
    None
    """
    
    base_plot_done = False

    base_field_idx = fields_idxs[used_fields.index(base_field)]
    
    
    base_PD_matrices = all_fields_data[base_field][:2]
    base_F_matrices = all_fields_data[base_field][2:]

    all_param_sets = []
    for field_dict in all_optimizations.values():
        for result in list(field_dict.values())[:num_sets_to_plot]:
            all_param_sets.append(result[0])

    global_base_min = np.inf
    global_base_max = -np.inf
    for p_set in all_param_sets:
        pd_eval = evaluate_model(p_set, *base_PD_matrices) * np.abs(global_averages_obs[base_field_idx])
        f_eval = evaluate_model(p_set, *base_F_matrices) * np.abs(global_averages_obs[base_field_idx])
        global_base_min = min(global_base_min, np.min(pd_eval), np.min(f_eval))
        global_base_max = max(global_base_max, np.max(pd_eval), np.max(f_eval))

    for idx, field in enumerate(used_fields): 

        if field == base_field:
            continue

        parameter_sets_names = list(all_optimizations[field].keys())[:num_sets_to_plot]
        parameter_sets  = np.array([result[0] for result in list(all_optimizations[field].values())[:num_sets_to_plot]])
        

        constr_PD_matrices = all_fields_data[field][:2]
        constr_F_matrices = all_fields_data[field][2:]

        for set_idx, parameter_set in enumerate(parameter_sets):
            
            param_name = parameter_sets_names[set_idx].replace("res","dp")
            latex_name = latexify_parameterset_name(param_name)

            math_content = latex_name.replace('$', '')


            global_constr_min = np.inf
            global_constr_max = -np.inf
            for p_set in all_param_sets:
                pd_eval = evaluate_model(p_set, *constr_PD_matrices) * np.abs(global_averages_obs[fields_idxs[idx]])
                f_eval = evaluate_model(p_set, *constr_F_matrices) * np.abs(global_averages_obs[fields_idxs[idx]])
                global_constr_min = min(global_constr_min, np.min(pd_eval), np.min(f_eval))
                global_constr_max = max(global_constr_max, np.max(pd_eval), np.max(f_eval))

            if set_idx != 0 or not base_plot_done:

                if set_idx == 0:
                    base_plot_done = True
                normalized_base_PD_plot_data = evaluate_model(parameter_set, *base_PD_matrices)
                normalized_base_F_plot_data = evaluate_model(parameter_set, *base_F_matrices)



                diff_base_PD_plot_data = normalized_base_PD_plot_data*np.abs(global_averages_obs[base_field_idx])
                diff_base_F_plot_data = normalized_base_F_plot_data*np.abs(global_averages_obs[base_field_idx])


                base_PD_plot = createMapPanel(diff_base_PD_plot_data,plot_width,rf"$\text{{{base_field} present-day perturbation for }} {math_content}$",box_size, minField=global_base_min,maxField=global_base_max)
                base_F_plot = createMapPanel(diff_base_F_plot_data,plot_width,rf"$\text{{{base_field} future perturbation for }} {math_content}$",box_size, minField=global_base_min, maxField=global_base_max)

                base_PD_plot.write_image(outdir / f"Diff_Map_PD_{param_name}_{base_field}.pdf")
                base_F_plot.write_image(outdir / f"Diff_Map_F_{param_name}_{base_field}.pdf")


            
             

            normalized_constr_PD_plot_data = evaluate_model(parameter_set, *constr_PD_matrices)
            normalized_constr_F_plot_data = evaluate_model(parameter_set, *constr_F_matrices)

            diff_constr_PD_plot_data = normalized_constr_PD_plot_data*np.abs(global_averages_obs[fields_idxs[idx]])
            diff_constr_F_plot_data = normalized_constr_F_plot_data*np.abs(global_averages_obs[fields_idxs[idx]])

            constr_PD_plot = createMapPanel(diff_constr_PD_plot_data, plot_width, rf"$\text{{{field} present-day perturbation for }} {math_content}$", box_size, minField=global_constr_min, maxField=global_constr_max)
            constr_F_plot = createMapPanel(diff_constr_F_plot_data, plot_width,rf"$\text{{{field} future perturbation for }} {math_content}$", box_size, minField=global_constr_min, maxField=global_constr_max)

            constr_PD_plot.write_image(outdir / f"Diff_Map_PD_{param_name}_{field}.pdf")
            constr_F_plot.write_image(outdir / f"Diff_Map_F_{param_name}_{field}.pdf")

            


def make_scatterplot(results, base_var_name, PD_base_SensMatrix, PD_base_CurvMatrix, F_base_SensMatrix,
                      F_base_CurvMatrix,constr_var_name, PD_constr_SensMatrix, PD_constr_CurvMatrix,
                        F_constr_SensMatrix, F_constr_CurvMatrix, combined_future = False, additional_params = None, e3sm_PPE_results=None, e3sm_optimized_results=None, include_average_constr =True, constrained_opt = False, ax =None):
    """
    Generates a scatterplot evaluating the trade-off between baseline results and constrained results.

    Plots initial versus final optimization states, draws relational vectors, 
    and overlays supplementary sample data to visualize the restriction imposed 
    by secondary constraints.

    Parameters
    ----------
    results : dict
        Dictionary of parameter vectors generated by the optimization routines.
    base_var_name, constr_var_name : str
        Identifiers for the base field and the secondary constraining field.
    PD_base_SensMatrix, PD_base_CurvMatrix : numpy.ndarray
        Baseline sensitivity and curvature matrices for the base field.
    F_base_SensMatrix, F_base_CurvMatrix : numpy.ndarray
        SST+4K sensitivity and curvature matrices for the base field.
    PD_constr_SensMatrix, PD_constr_CurvMatrix : numpy.ndarray
        Baseline matrices for the secondary constraining field.
    F_constr_SensMatrix, F_constr_CurvMatrix : numpy.ndarray
        SST+4K matrices for the secondary constraining field.
    combined_future : bool, optional
        If True, the future scenario to optimze consists of both the base field and the constraining field (default is False).
    additional_params : list or numpy.ndarray, optional
        Matrix of supplementary parameter sets to scatter as additional data.
    e3sm_PPE_results, e3sm_optimized_results : tuple of numpy.ndarray, optional
        Computed summed squared deviations for the underlying PPE and validation runs.
    include_average_constr : bool, optional
        If True, plots a reference line representing the expected ratio (default is True).
    constrained_opt : bool, optional
        Toggles between a ratio optimization and constrained optimization (default is False).
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    if constr_var_name == "TMQ":
        short_constr_name = 'Q'
    else:
        short_constr_name = constr_var_name[0]

    if base_var_name == "TMQ":
        short_base_name = 'Q'
    else:
        short_base_name = base_var_name[0]


    if combined_future:
        Future_SensMatrix = np.vstack((F_base_SensMatrix,F_constr_SensMatrix))
        Future_CurvMatrix = np.vstack((F_base_CurvMatrix,F_constr_CurvMatrix))
  
    else:
        Future_SensMatrix = F_base_SensMatrix
        Future_CurvMatrix = F_base_CurvMatrix


    combined_PD_SensMatrix = np.vstack((PD_base_SensMatrix,PD_constr_SensMatrix))
    combined_PD_CurvMatrix =  np.vstack((PD_base_CurvMatrix,PD_constr_CurvMatrix))

    if constrained_opt:
        R_F_B_fun = lambda dp: np.sum(evaluate_model(dp, Future_SensMatrix, Future_CurvMatrix)**2)
        R_F_BC_fun = lambda dp: evaluate_ratio(dp, Future_SensMatrix, Future_CurvMatrix, PD_constr_SensMatrix, PD_constr_CurvMatrix, eps=1)
    else:
        R_F_B_fun = lambda dp: evaluate_ratio(dp,Future_SensMatrix, Future_CurvMatrix,PD_base_SensMatrix, PD_base_CurvMatrix)
        R_F_BC_fun = lambda dp: evaluate_ratio(dp,Future_SensMatrix, Future_CurvMatrix,combined_PD_SensMatrix, combined_PD_CurvMatrix)

   

    R_max_F_B = results[f"res_max_F_{short_base_name}"][1]

    dp_max_F_B = results[f"res_max_F_{short_base_name}"][0]
    dp_max_F_BC = results[f"res_max_F_{short_base_name}{short_constr_name}"][0]
    dp_max_C_B = results[f"res_max_{short_constr_name}_{short_base_name}"][0]
    # dp_min_C_B = results[f"res_min_{short_constr_name}_{short_base_name}"][0]


    initial_x = R_F_B_fun(dp_max_F_B)/R_max_F_B
    initial_y = R_F_BC_fun(dp_max_F_B)/R_max_F_B

    initial_name= latexify_parameterset_name(f"res_max_F_{short_base_name}".replace("res","dp"))

    final_x = R_F_B_fun(dp_max_F_BC)/R_max_F_B
    final_y = R_F_BC_fun(dp_max_F_BC)/R_max_F_B

    final_name = latexify_parameterset_name(f"res_max_F_{short_base_name}{short_constr_name}".replace("res","dp"))

    sns.scatterplot(x=[initial_x], y=[initial_y],label=initial_name, color="red",s=200,linewidth=2,ax = ax)
    sns.scatterplot(x=[final_x], y=[final_y],label=final_name, color="red",marker="x",s=200,linewidth=2,ax = ax)
    # sns.scatterplot(x=[R_F_B_fun(dp_max_C_B)/R_max_F_B], y=[R_F_BC_fun(dp_max_C_B)/R_max_F_B],label=fr'$dp_{{max}}^{{{short_constr_name},{short_base_name}}}$', color="red",marker="+",s=100,linewidth=2,ax = ax)
    # sns.scatterplot(x=[R_F_B_fun(dp_min_C_B)/R_max_F_B], y=[R_F_BC_fun(dp_min_C_B)/R_max_F_B],label=fr'$dp_{{min}}^{{{short_constr_name},{short_base_name}}}$', color="red",marker="*",edgecolor="red",s=100,linewidth=2,ax = ax)

    ax.annotate("", xy=(final_x,final_y), xytext = (initial_x, initial_y), arrowprops=dict(arrowstyle=f"->,head_width={0.4}",color="black",lw=2,shrinkA=12, shrinkB=12))

    if additional_params is not None:

        PPE_R_F_B_results = np.apply_along_axis(R_F_B_fun, 1, np.array(additional_params))

        PPE_R_F_BC_results = np.apply_along_axis(R_F_BC_fun, 1, np.array(additional_params))

        sns.scatterplot(x=PPE_R_F_B_results/R_max_F_B, y=PPE_R_F_BC_results/R_max_F_B, label="PPE samples", color="purple",alpha=0.5,ax = ax)

    if e3sm_PPE_results is not None:

        if e3sm_PPE_results[0] is not None:
                   
            PPE_deltas = e3sm_PPE_results[0]
            PPE_deltas_sst4k = e3sm_PPE_results[1]

            if combined_future:
                PPE_F_B_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/PPE_deltas[:,0]
                PPE_F_BC_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/np.sum(PPE_deltas,axis=1)
            else:
                PPE_F_B_E3SM_ratios = PPE_deltas_sst4k[:,0]/PPE_deltas[:,0]
                PPE_F_BC_E3SM_ratios = PPE_deltas_sst4k[:,0]/np.sum(PPE_deltas,axis=1)


        sns.scatterplot(x=PPE_F_B_E3SM_ratios/R_max_F_B, y=PPE_F_BC_E3SM_ratios/R_max_F_B, label="PPE samples", color="purple",s=100,alpha=0.5,ax = ax)

    if e3sm_optimized_results is not None:
        PPE_deltas = e3sm_optimized_results[0]
        PPE_deltas_sst4k = e3sm_optimized_results[1]

        if combined_future:
            PPE_F_B_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = np.sum(PPE_deltas_sst4k,axis=1)/np.sum(PPE_deltas,axis=1)
        else:
            PPE_F_B_E3SM_ratios = PPE_deltas_sst4k[:,0]/PPE_deltas[:,0]
            PPE_F_BC_E3SM_ratios = PPE_deltas_sst4k[:,0]/np.sum(PPE_deltas,axis=1)


        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[0]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[0]/R_max_F_B],label=rf'E3SM $dp_{{max}}^{{F,{short_base_name}}}$', color="green",s=100,linewidth=2,ax = ax)
        sns.scatterplot(x=[PPE_F_B_E3SM_ratios[1]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[1]/R_max_F_B],label=rf'E3SM $dp_{{max}}^{{F,{short_base_name}{short_constr_name}}}$', color="green",marker="x",s=100,linewidth=2,ax = ax)
        # sns.scatterplot(x=[PPE_F_B_E3SM_ratios[2]/R_max_F_B], y=[PPE_F_BC_E3SM_ratios[2]/R_max_F_B],label=rf'E3SM $dp_{{max}}^{{{short_constr_name},{short_base_name}}}$', color="green",marker="+",s=100,linewidth=2,ax = ax)

    if include_average_constr:
        dp_ref = dp_max_C_B

        H_B = get_H_at_dp(PD_base_SensMatrix, PD_base_CurvMatrix, dp_ref)
        H_C = get_H_at_dp(PD_constr_SensMatrix, PD_constr_CurvMatrix, dp_ref)

        E_RR = calc_E_RR(H_B, H_C)

        sns.lineplot(x=[0,1],y =[0,E_RR], label=r"Average ratio ($\mathcal{E}_{RR}$)", linestyle=":",linewidth = 4,color="cyan", ax=ax)


    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel(fr'$\frac{{R^{{(F,{short_base_name})}}}}{{R^{{(F,{short_base_name})}}_{{max}}}}$',fontsize=24)
    ax.set_ylabel(fr'$\frac{{R^{{(F,{short_base_name}{short_constr_name})}}}}{{R^{{(F,{short_base_name})}}_{{max}}}}$',fontsize=24,rotation=0,ha='right', va='center')

    ax.yaxis.set_label_coords(-0.05, 0.50)

    sns.lineplot(x=[0,1],y=[0,1], label="Upper bound", color="black", linestyle="--",alpha=0.5,ax = ax)

    ax.tick_params(axis='both', labelsize=14)

    ax.legend(framealpha=0.9, fontsize=20, title_fontsize=20)
    return ax



def create_cartoon_scatterplot(ax=None, scatter_size=200, text_size=14, arrow_pad = 12, arrow_size =4, head_width =0.6):
    """
    Generates a conceptual, annotated scatterplot to illustrate theoretical 
    optimization scenarios and boundary behaviors.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot.
    scatter_size : int, optional
        Base size for the scatter markers (default is 200).
    text_size : int, optional
        Font size for the annotations (default is 14).
    arrow_pad, arrow_size, head_width : float/int, optional
        Styling parameters for the arrows showcasing the change from inital to final points.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object.
    """
    if ax == None:
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
    ax.text(x=0.7, y=0.1,s="Guaranteed strong constraint",c="green", fontsize = text_size,bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2), horizontalalignment='center')


    #Scenario 2
    sns.scatterplot(x = [1], y=[0.94], color="blue", s=scatter_size, ax=ax)
    ax.text(x=1,y=1,s="Guaranteed weak constraint", c="blue",fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')

    #Scenario 3
    sns.scatterplot(x=[1], y =[0.7], color="tab:orange",s=scatter_size, alpha=0.5, ax=ax)
    sns.scatterplot(x=[1], y =[0.7], color="tab:orange",s=scatter_size*1.6, alpha=0.5, marker="X", ax=ax)

    ax.text(x=1,y=0.75,s="Initial and final point coincide", c="tab:orange", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')

    #Scenario 4a
    sns.scatterplot(x=[1], y=[0.2], color="red", s=scatter_size, ax=ax)
    sns.scatterplot(x=[0.85], y=[0.2], color="red", s=scatter_size*1.6,marker="X",ax=ax)

    ax.annotate("", xy=(0.85,0.2), xytext = (1,0.2), arrowprops=dict(arrowstyle=f"->,head_width={head_width}",color="red",lw=arrow_size,shrinkA=arrow_pad, shrinkB=arrow_pad))


    ax.text(x=0.85,y=0.25,s="Unevaded constraint",c="red", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='center')


    #Scenario 4b
    sns.scatterplot(x=[1], y=[0.4], color="purple", s=scatter_size, ax=ax)
    sns.scatterplot(x=[1], y=[0.55], color="purple", s=scatter_size*1.6,marker="X",ax=ax)
    ax.text(x=0.95,y=0.475,s="Evaded constraint",c="purple", fontsize = text_size, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),horizontalalignment='right')
    ax.annotate("", xy=(1,0.55), xytext = (1,0.4), arrowprops=dict(arrowstyle=f"->,head_width={head_width}",color="purple",lw=arrow_size,shrinkA=arrow_pad, shrinkB=arrow_pad))






    ax.set_xlabel(r'$\frac{R^{(F,S)}}{R^{(F,S)}_{\text{max}}}$', fontsize = text_size*1.5)

    ax.set_ylabel(r'$\frac{R^{(F,SL)}}{R^{(F,S)}_{\text{max}}}$',fontsize = text_size*1.5, rotation =0,ha='right', va='center' )
    ax.yaxis.set_label_coords(-0.05, 0.50)


    ax.tick_params(axis='both', labelsize=text_size)

    marker_init = mlines.Line2D([], [], color='w', marker='o',
                                markerfacecolor='dimgrey', markeredgecolor='dimgrey',
                                markersize=9, label='Initial')
    marker_final = mlines.Line2D([], [], color='w', marker='X',
                                 markerfacecolor='dimgrey', markeredgecolor='dimgrey',
                                 markersize=9, label='Final')

    ax.legend(handles=[marker_init, marker_final],
              title="Point Type", framealpha=0.9, fontsize=text_size, title_fontsize=text_size)

   
    return ax


def create_shape_vs_scale_plot(all_optimizations, base_field, used_fields, all_fields_data, text_size = 14, scatter_size =150, paramset_idx_to_plot=2, colors = None, ax=None):
    """
    Visualizes the relationship between volumetric scaling metrics and matrix 
    shape metrics across different constrained optimizations.

    Evaluates the localized Hessian matrices of the base and constrained targets 
    at a specific parameter vector to perform linear analysis.

    Parameters
    ----------
    all_optimizations : dict
        Nested dictionary of optimization results.
    base_field : str
        Name of the base field.
    used_fields : list of str
        List of all variables being evaluated.
    all_fields_data : dict
        Dictionary mapping field names to their respective sensitvity and curvature matrices.
    text_size : int, optional
        Font size for axis labels and ticks (default is 14).
    scatter_size : int, optional
        Size of the scatter markers (default is 150).
    paramset_idx_to_plot : int, optional
        The specific index of the optimization result used for linearization (default is 2).
    colors : list, optional
        Color palette to apply to the plotted constraint sets.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes for the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The updated axes object.
    """

    if ax is None:
        fig, ax  = plt.subplots(figsize=(8,8))
   
    fields_to_plot = used_fields.copy()
    fields_to_plot.remove(base_field)

    BaseSensMatrix, BaseCurvMatrix = all_fields_data[base_field][:2]

    if colors is None:
        colors = sns.color_palette("tab10", n_colors=len(fields_to_plot))


    R_scales, _, R_shapes, _ = calc_all_A_opt_metrics(all_optimizations, used_fields, base_field, all_fields_data, paramset_idx_to_plot)

    labels = []
    for idx, field in enumerate(fields_to_plot):
        

        constrained_parameter_set_name = list(all_optimizations[field].keys())[paramset_idx_to_plot]
        constrained_parameter_set  = np.array([result[0] for result in list(all_optimizations[field].values())])[paramset_idx_to_plot]

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