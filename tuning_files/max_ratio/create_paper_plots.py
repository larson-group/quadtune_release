


from optimization import (optimize_all, normalize_params, denormalize_params)

from data_io import (get_metrics_names, get_filenames, get_params_from_files,
                        calc_metric_sum_delta_from_file, 
                        get_correct_model_runs_delta, 
                        extract_field_matrices_from_whole)

from plotting import (latexify_parameterset_name, create_parameter_bar_chart,
                       create_map_plots, make_scatterplot,
                         create_cartoon_scatterplot, create_ortho_vs_scale_plot)


import numpy as np
import xarray as xr
import argparse

from pathlib import Path

import seaborn as sns
sns.set_theme(context="paper",style="whitegrid")



SEED = 26052026

"""
General configuration
"""
ALL_PARAMS_NAMES = np.array(['clubb_c1','clubb_gamma_coef','zmconv_tau','zmconv_dmpdz','zmconv_ke','zmconv_micro_dcs','zmconv_auto_fac','zmconv_accr_fac','p3_nc_autocon_expon','p3_qc_accret_expon','p3_embryonic_rain_size','p3_mincdnc','cldfrc_dp1','nucleate_ice_subgrid'])
ALL_DEFAULT_PARAMS = np.array([2.4,0.12,3600,-0.7e-3,2.5e-6,150e-6,7.0,1.5,-1.2,1.15,25e-6,20e6,0.018,1.35]) 

ALL_LOW_PARAMS = np.array([1.0, 0.1, 1800, -2.0e-3, 0.5e-6, 100e-6, 3.0, 1.3, -1.79, 1.1, 15e-6, 5e6, 0.01, 1.0])
ALL_HIGH_PARAMS = np.array([5.0, 0.5, 14400, -0.1e-3, 10e-6, 400e-6, 7.5, 2.0, -0.7, 1.3, 40e-6, 30e6, 0.1, 1.4])

ALL_FIELDS = ['SWCF','LWCF', 'PRECT', 'TMQ']

GLOBAL_AVERAGES_OBS = np.array([-45.30901732406915,  25.81230998182771,   2.6904643040278655, 24.39980138451228 ])

USED_PARAMETER_NAMES = ALL_PARAMS_NAMES
# USED_PARAMETER_NAMES = ['clubb_c1','p3_nc_autocon_expon','p3_qc_accret_expon' ,'p3_embryonic_rain_size','nucleate_ice_subgrid' ]
# used_parameters_names = ["clubb_c1","clubb_gamma_coef"]

USED_FIELDS = ALL_FIELDS

COLORS = ['red','blue','green']

# used_fields = ["SWCF","LWCF","PRECT"]
# used_fields = ["SWCF","LWCF"]


BASE_FIELD = 'SWCF'

BOX_SIZE = 15

DATAPREFIX = '1520260310'

LOAD_DATA = True



def main(args):
    
    



    """
    Check if everything is correctly defined
    """
    if (args.ppe_data is None) != (args.ppe_data_sst4k is None):
        print("Provide either no path to a PPE or for both SST and SST4K!")
        return

    if (args.e3sm_results is not None) and (args.ppe_data is None):
        print("Plotting E3SM results requires the original PPE for SST and SST4K to ensure coherent normalization of the results")
        return


    """
    Define indices
    """
    fields_idxs = [list(ALL_FIELDS).index(field_name) for field_name in USED_FIELDS]

    parameter_idxs = [list(ALL_PARAMS_NAMES).index(param_name) for param_name in USED_PARAMETER_NAMES]
    
    """
    Read data
    """

    if args.outdir is None:
        outdir = Path("./")
    else:
        outdir = Path(args.outdir)


    datapath = Path(args.datapath)

    try:
        full_sens_matrix = np.load(datapath / "SensMatrix.npy")
        full_curv_matrix = np.load(datapath / "CurvMatrix.npy")
        full_sens_matrix_sst4k = np.load(datapath / "SensMatrixSST4K.npy")
        full_curv_matrix_sst4k = np.load(datapath / "CurvMatrixSST4K.npy")
    except FileNotFoundError:
        print("Sensitivity and curvature matrices not found")
        return

    if LOAD_DATA:
        try:
            PPE_params = np.load(datapath / "PPE_params.npy")[:,parameter_idxs]
        except FileNotFoundError:
            PPE_params = None

        try:
            PPE_deltas = np.load(datapath / "PPE_deltas.npy")
        except FileNotFoundError:
            PPE_deltas = None
        try:
            PPE_deltas_sst4k = np.load(datapath / "PPE_deltas_sst4k.npy")
        except FileNotFoundError:
            PPE_deltas_sst4k = None

    else:
        PPE_params = None

        PPE_deltas = None

        PPE_deltas_sst4k = None


    """
    Preprocess data
    """
    default_params = ALL_DEFAULT_PARAMS[parameter_idxs]
    low_params = ALL_LOW_PARAMS[parameter_idxs]
    high_params = ALL_HIGH_PARAMS[parameter_idxs]



    PD_Sens = extract_field_matrices_from_whole(full_sens_matrix, ALL_FIELDS, USED_FIELDS,parameter_idxs, BOX_SIZE)

    F_Sens = extract_field_matrices_from_whole(full_sens_matrix_sst4k, ALL_FIELDS, USED_FIELDS,parameter_idxs, BOX_SIZE)

    PD_Curv = extract_field_matrices_from_whole(full_curv_matrix, ALL_FIELDS, USED_FIELDS,parameter_idxs, BOX_SIZE)

    F_Curv = extract_field_matrices_from_whole(full_curv_matrix_sst4k, ALL_FIELDS, USED_FIELDS,parameter_idxs, BOX_SIZE)

    all_fields_data = {}
    for field in USED_FIELDS:
        field_data = [PD_Sens[field], PD_Curv[field], F_Sens[field], F_Curv[field]]
        all_fields_data[field] = field_data



    
    if args.ppe_data is not None:
        files = get_filenames(args.ppe_data, DATAPREFIX)

        sst4k_files = get_filenames(args.ppe_data_sst4k, DATAPREFIX)

        files.sort(key=lambda x: int(x.split('_')[-2]))
        sst4k_files.sort(key=lambda x: int(x.split('_')[-2]))


        default_file = files[0]
        sst4k_default_file = sst4k_files[0]

        default_dataset = xr.open_dataset(default_file, engine='netcdf4')
        default_dataset_sst4k = xr.open_dataset(sst4k_default_file, engine='netcdf4')

        default_params = default_dataset[USED_PARAMETER_NAMES].to_array().values.flatten()
        


        if PPE_deltas is None:
            metric_names = get_metrics_names(USED_FIELDS, BOX_SIZE)
            default_data = default_dataset[metric_names].to_array().values.flatten()
            PPE_deltas = np.array([calc_metric_sum_delta_from_file(file,default_data,GLOBAL_AVERAGES_OBS[fields_idxs],metric_names) for file in files[1:2*len(ALL_PARAMS_NAMES)+1]])
            # np.save(f"{datapath}/PPE_deltas",PPE_deltas)
        
        if PPE_deltas_sst4k is None:
            metric_names = get_metrics_names(USED_FIELDS, BOX_SIZE)
            default_data_sst4k = default_dataset_sst4k[metric_names].to_array().values.flatten()
            PPE_deltas_sst4k = np.array([calc_metric_sum_delta_from_file(file,default_data_sst4k,GLOBAL_AVERAGES_OBS[fields_idxs], metric_names) for file in sst4k_files[1:2*len(ALL_PARAMS_NAMES)+1]])
            # np.save(f"{datapath}/PPE_deltas_sst4k",PPE_deltas_sst4k)
        default_dataset.close()
        default_dataset_sst4k.close()
        


    if args.e3sm_results is not None:
        e3sm_result_files = get_filenames(args.e3sm_results,DATAPREFIX)
        sst4k_e3sm_result_files = get_filenames(args.e3sm_results_sst4k,DATAPREFIX)

        e3sm_result_files.sort(key=lambda x: int(x.split('_')[-2]))
        sst4k_e3sm_result_files.sort(key=lambda x: int(x.split('_')[-2]))

        params_E3SM_runs = np.array(get_params_from_files(e3sm_result_files,USED_PARAMETER_NAMES))

    normlzd_low_params = normalize_params(low_params, default_params)
    normlzd_high_params = normalize_params(high_params, default_params)

    optimizer_bounds = list(zip(normlzd_low_params, normlzd_high_params))


    """
    Optimize ratios
    """
    all_optimizations = {}

    for current_field in USED_FIELDS:
        if current_field == BASE_FIELD:
            continue

        all_optimizations[current_field] = optimize_all(BASE_FIELD, *all_fields_data[BASE_FIELD],current_field, *all_fields_data[current_field], optimizer_bounds)

    print(all_optimizations)

    ################ TODO: REMOVE THIS


    for idx,current_field in enumerate(USED_FIELDS):
        if current_field == BASE_FIELD:
            continue

        if current_field  == "TMQ":
            short_field_name = "Q"
        else:
            short_field_name = current_field[0]

        print(f"{current_field} F_S: {denormalize_params(all_optimizations[current_field]['res_max_F_S'][0], default_params)}")
        print(f"{current_field} F_SC: {denormalize_params(all_optimizations[current_field][f'res_max_F_S{short_field_name}'][0], default_params)}")



    #########################
    """
    Create cartoon scatter
    """
    print("Creating cartoon scatter plot")
    cartoon_ax = create_cartoon_scatterplot()
    cartoon_fig = cartoon_ax.get_figure()
    cartoon_fig.savefig(outdir / "Scatterplot_cartoon.pdf",bbox_inches="tight", dpi=300)

    """
    Create scale vs ortho plot
    """

    print("Creating scale vs ortho plot")
    scale_ortho_ax = create_ortho_vs_scale_plot(all_optimizations, BASE_FIELD, USED_FIELDS, all_fields_data, colors=COLORS)
    scale_ortho_fig = scale_ortho_ax.get_figure()
    scale_ortho_fig.savefig(outdir / "Scale_vs_Ortho_Scatter.pdf",bbox_inches="tight", dpi=300)

    ##################TODO: Remove this
    for i in range(4):
        scale_ortho_ax = create_ortho_vs_scale_plot(all_optimizations, BASE_FIELD, USED_FIELDS, all_fields_data, colors=COLORS,paramset_idx_to_plot=i)
        scale_ortho_fig = scale_ortho_ax.get_figure()
        scale_ortho_fig.savefig(outdir / f"Scale_vs_Ortho_Scatter_{i}.pdf",bbox_inches="tight", dpi=300)

    """
    Create mapplots
    """
    print("Creating mapplots")

    create_map_plots(all_optimizations, fields_idxs, USED_FIELDS, BASE_FIELD, all_fields_data, GLOBAL_AVERAGES_OBS, BOX_SIZE, outdir)

    """
    Create parameter barplot
    """
    print("Creating parameter bar plot")
    create_parameter_bar_plot(all_optimizations, USED_FIELDS, BASE_FIELD, USED_PARAMETER_NAMES, outdir)



    """Create Scatter plots"""
    print("Creating scatterplots")
    for idx, current_field in enumerate(ALL_FIELDS):

        if current_field == BASE_FIELD:
            continue

        if current_field not in USED_FIELDS:
            continue

        print(f"Scatter: {current_field}")

        optimized_results = all_optimizations[current_field]

        optimized_params = np.array([val[0] for val in list(optimized_results.values())])[:-1]

        denorm_optimized_params = [denormalize_params(paramset, default_params) for paramset in optimized_params]


        base_field_idx = ALL_FIELDS.index(BASE_FIELD)

        current_fields_idxs = [base_field_idx, idx]

        if args.e3sm_results is not None:
            e3sm_results = get_correct_model_runs_delta(denorm_optimized_params,params_E3SM_runs,np.array(e3sm_result_files),np.array(sst4k_e3sm_result_files), default_file, sst4k_default_file,np.array(ALL_FIELDS)[current_fields_idxs], np.array(GLOBAL_AVERAGES_OBS)[current_fields_idxs], BOX_SIZE)
        else:
            e3sm_results = None

        if PPE_deltas is not None:
            PPE_E3SM_res = [PPE_deltas[:,current_fields_idxs], PPE_deltas_sst4k[:,current_fields_idxs]]
        else:
            PPE_E3SM_res = None

        
        ax = make_scatterplot(optimized_results,BASE_FIELD, *all_fields_data[BASE_FIELD],current_field,
                  *all_fields_data[current_field], e3sm_PPE_results=PPE_E3SM_res, e3sm_optimized_results=e3sm_results)

        fig = ax.get_figure()

        fig.savefig(outdir / f"Scatter_{BASE_FIELD}_future_{BASE_FIELD}_constr_by_{current_field}.pdf",bbox_inches="tight", dpi=300)






def create_parameter_bar_plot(all_optimizations, used_fields, base_field, params_names, outdir, idx_to_plot =1):
    
    parametersets = []
    parameter_sets_names =[]

    for field in used_fields:

        if field == base_field:
            parameter_sets_names.append(fr'$dp_{{max}}^{{F,{field[0]}}}$')
            parametersets.append(all_optimizations[used_fields[idx_to_plot]][f'res_max_F_{field[0]}'][0])
            continue

        current_field_optimizations = all_optimizations[field]

        names = list(current_field_optimizations.keys())

        parametersets.append(current_field_optimizations[names[idx_to_plot]][0])
        parameter_sets_names.append(latexify_parameterset_name(names[idx_to_plot].replace("res","dp")))
    
    ax = create_parameter_bar_chart(np.array(parametersets), params_names, parameter_sets_names)

    fig = ax.get_figure()
    fig.savefig(outdir / "parameter_bar_plot.pdf",bbox_inches="tight", dpi=300)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--datapath",type=str,required=True,help="Path where the sensitivity and curvature matrices as well as stored computed values are located")

    parser.add_argument("-p","--ppe_data",type = str, required=False, help="Path where the PPE is stored, in case it should be plotted")
    parser.add_argument("--ppe_data_sst4k",type = str, required=False, help="Path where the SST4K PPE is stored, in case it should be plotted")

    parser.add_argument("-o","--outdir", type =str, required=False,help="Path where the plots will be stored" )

    parser.add_argument("--e3sm_results",type=str,required=False,help="Path where E3SM results of the optimized parameters are stored")

    parser.add_argument("--e3sm_results_sst4k",type=str,required=False,help="Path where E3SM results of the optimized parameters are stored")

    args = parser.parse_args()
    main(args)