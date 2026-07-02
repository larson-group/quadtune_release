"""
Create all plots used in the first submission of the Nobre-Wittwer et. al. 2026 paper "Effectiveness of observable constraints on the worst-case uncertainty of a global atmospheric model".
https://essopenarchive.org/doi/full/10.22541/essoar.15004390/v1
Runs multiple constrained optimizations and additional diagnostics (e.g. the expected average ratio of ratios).
"""


from concurrent.futures import ProcessPoolExecutor

from optimization import calc_all_A_opt_metrics, calc_all_E_RR, optimize_all, normalize_params, denormalize_params

from data_io import (
    get_metrics_names,
    get_filenames,
    get_params_from_files,
    calc_metric_sum_delta_from_file,
    get_correct_model_runs_delta,
    extract_field_matrices_from_whole,
    flatten_results,
)

from plotting import (
    latexify_parameterset_name,
    create_parameter_bar_chart,
    create_map_plots,
    make_scatterplot,
    create_cartoon_scatterplot,
    create_shape_vs_scale_plot,
)


import numpy as np
import xarray as xr
import argparse
import sys

from pathlib import Path

import seaborn as sns

sns.set_theme(context="paper", style="whitegrid")

engine = "h5netcdf" if sys.platform == "win32" else "netcdf4"

"""
General configuration
"""
# All parameter names available in the used files and sensitivity/curvature matrices
ALL_PARAMS_NAMES = np.array(
    [
        "clubb_c1",
        "clubb_gamma_coef",
        "zmconv_tau",
        "zmconv_dmpdz",
        "zmconv_ke",
        "zmconv_micro_dcs",
        "zmconv_auto_fac",
        "zmconv_accr_fac",
        "p3_nc_autocon_expon",
        "p3_qc_accret_expon",
        "p3_embryonic_rain_size",
        "p3_mincdnc",
        "cldfrc_dp1",
        "nucleate_ice_subgrid",
    ]
)
# The default values for all available parameters
ALL_DEFAULT_PARAMS = np.array(
    [
        2.4,
        0.12,
        3600,
        -0.7e-3,
        2.5e-6,
        150e-6,
        7.0,
        1.5,
        -1.2,
        1.15,
        25e-6,
        20e6,
        0.018,
        1.35,
    ]
)
# The low perturbed parameter values used for fitting Quadtune
ALL_LOW_PARAMS = np.array(
    [
        1.0,
        0.1,
        1800,
        -2.0e-3,
        0.5e-6,
        100e-6,
        3.0,
        1.3,
        -1.79,
        1.1,
        15e-6,
        5e6,
        0.01,
        1.0,
    ]
)
# The high perturbed parameter values used for fitting Quadtune
ALL_HIGH_PARAMS = np.array(
    [
        5.0,
        0.5,
        14400,
        -0.1e-3,
        10e-6,
        400e-6,
        7.5,
        2.0,
        -0.7,
        1.3,
        40e-6,
        30e6,
        0.1,
        1.4,
    ]
)

# All available fields in the provided sensitivity/curvature matrices
ALL_FIELDS = ["SWCF", "LWCF", "PRECT", "TMQ"]

# The 5 year global observed averages used for normalization
GLOBAL_AVERAGES_OBS = np.array(
    [-45.30901732406915, 25.81230998182771, 2.6904643040278655, 24.39980138451228]
)
# Seed for the randomized basinhopping algorithm to ensure reproducability
SEED = 26052026
# Safeguard to prevent division by 0
EPS = 1e-4
# The box size from the data used to fit quadtune and produce output plots
BOX_SIZE = 15


"""
Run specific configuration
"""

# The base field used as a reference to determine the impact of additional constraints
BASE_FIELD = "SWCF"
# The prefix of the files containing the model runs
DATAPREFIX = "1520260310"


# Parameters that should be considered in the following computations
USED_PARAMETER_NAMES = ALL_PARAMS_NAMES
# USED_PARAMETER_NAMES = ['clubb_c1','p3_nc_autocon_expon','p3_qc_accret_expon' ,'p3_embryonic_rain_size','nucleate_ice_subgrid' ]

# Index of parameter, for which a barplot should be created
PARAM_IDX_TO_BARPLOT = 1

# Index of parameterset to linearize about
# e.g. for LWCF: 0 -> R_F_S, 1-> R_F_SL, 2-> R_LS
PARAM_IDX_TO_LINEARIZE = 2 

# Fields that should be considered in the following computations
USED_FIELDS = ALL_FIELDS
# Colors used some plots for different fields
COLORS = ["red", "blue", "green"]

# Configure how many of the optimization results should be used to create map plots
NUM_PARAMS_TO_MAP_PLOT = 2


def main(argv = None):
    """
    Check if everything is correctly defined
    """
    args = parse_arguments(argv)

    if (args.ppe_data is None) != (args.ppe_data_sst4k is None):
        print("Provide either no path to a PPE or for both SST and SST4K!")
        return

    if (args.e3sm_results is not None) and (args.ppe_data is None):
        print(
            "Plotting E3SM results requires the original PPE for SST and SST4K to ensure coherent normalization of the results"
        )
        return

    """
    Define indices
    """
    fields_idxs = [list(ALL_FIELDS).index(field_name) for field_name in USED_FIELDS]

    parameter_idxs = [
        list(ALL_PARAMS_NAMES).index(param_name) for param_name in USED_PARAMETER_NAMES
    ]

    """
    Read data and configuration
    """

    LOAD_DATA = args.load_data
    CONSTR_OPT = args.constr_opt
    CONSTR_VALUE = args.constr_value

    if args.outdir is None:
        outdir = Path("./")
    else:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

    datapath = Path(args.datapath)

    try:
        full_sens_matrix = np.load(datapath / "SensMatrix.npy")
        full_curv_matrix = np.load(datapath / "CurvMatrix.npy")
        full_sens_matrix_sst4k = np.load(datapath / "SensMatrixSST4K.npy")
        full_curv_matrix_sst4k = np.load(datapath / "CurvMatrixSST4K.npy")
    except FileNotFoundError:
        print(f"Sensitivity and curvature matrices not found at {datapath}")
        return

    if LOAD_DATA:
        try:
            PPE_deltas = np.load(datapath / "PPE_deltas.npy")
        except FileNotFoundError:
            PPE_deltas = None
        try:
            PPE_deltas_sst4k = np.load(datapath / "PPE_deltas_sst4k.npy")
        except FileNotFoundError:
            PPE_deltas_sst4k = None

    else:

        PPE_deltas = None
        PPE_deltas_sst4k = None

    """
    Preprocess data
    """
    default_params = ALL_DEFAULT_PARAMS[parameter_idxs]
    low_params = ALL_LOW_PARAMS[parameter_idxs]
    high_params = ALL_HIGH_PARAMS[parameter_idxs]

    PD_Sens = extract_field_matrices_from_whole(
        full_sens_matrix, ALL_FIELDS, USED_FIELDS, parameter_idxs, BOX_SIZE
    )

    F_Sens = extract_field_matrices_from_whole(
        full_sens_matrix_sst4k, ALL_FIELDS, USED_FIELDS, parameter_idxs, BOX_SIZE
    )

    PD_Curv = extract_field_matrices_from_whole(
        full_curv_matrix, ALL_FIELDS, USED_FIELDS, parameter_idxs, BOX_SIZE
    )

    F_Curv = extract_field_matrices_from_whole(
        full_curv_matrix_sst4k, ALL_FIELDS, USED_FIELDS, parameter_idxs, BOX_SIZE
    )

    all_fields_data = {}
    for field in USED_FIELDS:
        field_data = [PD_Sens[field], PD_Curv[field], F_Sens[field], F_Curv[field]]
        all_fields_data[field] = field_data

    if args.ppe_data is not None:
        files = get_filenames(args.ppe_data, DATAPREFIX)

        sst4k_files = get_filenames(args.ppe_data_sst4k, DATAPREFIX)

        files.sort(key=lambda x: int(x.split("_")[-2]))
        sst4k_files.sort(key=lambda x: int(x.split("_")[-2]))

        default_file = files[0]
        sst4k_default_file = sst4k_files[0]
        default_dataset = xr.open_dataset(default_file, engine=engine)
        default_dataset_sst4k = xr.open_dataset(sst4k_default_file, engine=engine)
        default_params = (
            default_dataset[USED_PARAMETER_NAMES].to_array().values.flatten()
        )

        if PPE_deltas is None:
            metric_names = get_metrics_names(USED_FIELDS, BOX_SIZE)
            default_data = default_dataset[metric_names].to_array().values.flatten()
            with ProcessPoolExecutor() as executor:
                results = executor.map(
                    calc_metric_sum_delta_from_file,
                    files[1 : 2 * len(ALL_PARAMS_NAMES) + 1],
                    [default_data] * len(files),
                    [GLOBAL_AVERAGES_OBS[fields_idxs]] * len(files),
                    [metric_names] * len(files),
                )

            PPE_deltas = np.array(list(results))
            if LOAD_DATA:
                np.save(f"{datapath}/PPE_deltas", PPE_deltas)

        if PPE_deltas_sst4k is None:
            metric_names = get_metrics_names(USED_FIELDS, BOX_SIZE)
            default_data_sst4k = (
                default_dataset_sst4k[metric_names].to_array().values.flatten()
            )
            with ProcessPoolExecutor() as executor:
                results = executor.map(
                    calc_metric_sum_delta_from_file,
                    sst4k_files[1 : 2 * len(ALL_PARAMS_NAMES) + 1],
                    [default_data_sst4k] * len(sst4k_files),
                    [GLOBAL_AVERAGES_OBS[fields_idxs]] * len(sst4k_files),
                    [metric_names] * len(sst4k_files),
                )
            PPE_deltas_sst4k = np.array(list(results))

            if LOAD_DATA:
                np.save(f"{datapath}/PPE_deltas_sst4k", PPE_deltas_sst4k)

        default_dataset.close()
        default_dataset_sst4k.close()

    if args.e3sm_results is not None:
        e3sm_result_files = get_filenames(args.e3sm_results, DATAPREFIX)
        sst4k_e3sm_result_files = get_filenames(args.e3sm_results_sst4k, DATAPREFIX)

        e3sm_result_files.sort(key=lambda x: int(x.split("_")[-2]))
        sst4k_e3sm_result_files.sort(key=lambda x: int(x.split("_")[-2]))

        params_E3SM_runs = np.array(
            get_params_from_files(e3sm_result_files, USED_PARAMETER_NAMES)
        )

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

        all_optimizations[current_field] = optimize_all(
            BASE_FIELD,
            *all_fields_data[BASE_FIELD],
            current_field,
            *all_fields_data[current_field],
            optimizer_bounds,
            eps=EPS,
            run_exact_constraint=CONSTR_OPT,
            exact_constr_value=CONSTR_VALUE,
        )
    """
    In a testing configuration where plots are not needed, the code can be stopped early to minimize the compute time.
    """
    if args.testing:
        flattened_results_dict = flatten_results(all_optimizations)
        R_scales, E_RLS, R_shapes, _ = calc_all_A_opt_metrics(all_optimizations,USED_FIELDS, BASE_FIELD, all_fields_data, PARAM_IDX_TO_LINEARIZE)
        flattened_results_dict["R_scales"] = np.array(R_scales)
        flattened_results_dict["E_RLS"] = np.array(E_RLS)
        flattened_results_dict["R_shapes"] = np.array(R_shapes)

        #E_RR is defined as an expected ratio of variance ratios as described in Nobre-Wittwer et al. (2026)
        E_RR = calc_all_E_RR(all_optimizations,USED_FIELDS, BASE_FIELD, all_fields_data, PARAM_IDX_TO_LINEARIZE)

        flattened_results_dict["E_RR"] = np.array(E_RR)

        # np.savez("reference_max_ratio_results.npz",**flattened_results_dict)
        return flattened_results_dict
    

    """
    Create cartoon scatter
    """
    print("Creating cartoon scatter plot")
    cartoon_ax = create_cartoon_scatterplot()
    cartoon_fig = cartoon_ax.get_figure()
    cartoon_fig.savefig(
        outdir / "Scatterplot_cartoon.pdf", bbox_inches="tight", dpi=300
    )

    """
    Create scale vs shape plot
    """

    print("Creating scale vs shape plot")
    scale_shape_ax = create_shape_vs_scale_plot(
        all_optimizations, BASE_FIELD, USED_FIELDS, all_fields_data, colors=COLORS,paramset_idx_to_plot=PARAM_IDX_TO_LINEARIZE
    )
    scale_shape_fig = scale_shape_ax.get_figure()
    scale_shape_fig.savefig(
        outdir / "Scale_vs_Shape_Scatter.pdf", bbox_inches="tight", dpi=300
    )

    """
    Create mapplots
    """
    print("Creating mapplots")

    create_map_plots(
        all_optimizations,
        fields_idxs,
        USED_FIELDS,
        BASE_FIELD,
        all_fields_data,
        GLOBAL_AVERAGES_OBS,
        BOX_SIZE,
        outdir,
        num_sets_to_plot=NUM_PARAMS_TO_MAP_PLOT,
    )

    """
    Create parameter barplot
    """
    print("Creating parameter bar plot")
    parametersets = []
    parameter_sets_names = []

    for field in USED_FIELDS:
        if field == BASE_FIELD:
            parameter_sets_names.append(rf"$dp_{{max}}^{{F,{field[0]}}}$")
            parametersets.append(
                all_optimizations[USED_FIELDS[PARAM_IDX_TO_BARPLOT]][
                    f"res_max_F_{field[0]}"
                ][0]
            )
            continue

        current_field_optimizations = all_optimizations[field]

        names = list(current_field_optimizations.keys())

        parametersets.append(
            current_field_optimizations[names[PARAM_IDX_TO_BARPLOT]][0]
        )
        parameter_sets_names.append(
            latexify_parameterset_name(names[PARAM_IDX_TO_BARPLOT].replace("res", "dp"))
        )

    parameter_bar_ax = create_parameter_bar_chart(
        np.array(parametersets), USED_PARAMETER_NAMES, parameter_sets_names
    )
    parameter_bar_fig = parameter_bar_ax.get_figure()
    parameter_bar_fig.savefig(
        outdir / "parameter_bar_plot.pdf", bbox_inches="tight", dpi=300
    )

    """
    Create Scatter plots
    """
    print("Creating scatterplots")
    for idx, current_field in enumerate(ALL_FIELDS):

        if current_field == BASE_FIELD:
            continue

        if current_field not in USED_FIELDS:
            continue

        print(f"Scatter: {current_field}")

        optimized_results = all_optimizations[current_field]

        optimized_params = np.array(
            [val[0] for val in list(optimized_results.values())]
        )[:-1]

        denorm_optimized_params = [
            denormalize_params(paramset, default_params)
            for paramset in optimized_params
        ]

        base_field_idx = ALL_FIELDS.index(BASE_FIELD)

        current_fields_idxs = [base_field_idx, idx]

        if args.e3sm_results is not None:
            e3sm_results = get_correct_model_runs_delta(
                denorm_optimized_params,
                params_E3SM_runs,
                np.array(e3sm_result_files),
                np.array(sst4k_e3sm_result_files),
                default_file,
                sst4k_default_file,
                np.array(ALL_FIELDS)[current_fields_idxs],
                np.array(GLOBAL_AVERAGES_OBS)[current_fields_idxs],
                BOX_SIZE,
            )
        else:
            e3sm_results = None

        if PPE_deltas is not None:
            PPE_E3SM_res = [
                PPE_deltas[:, current_fields_idxs],
                PPE_deltas_sst4k[:, current_fields_idxs],
            ]
        else:
            PPE_E3SM_res = None

        ax = make_scatterplot(
            optimized_results,
            BASE_FIELD,
            *all_fields_data[BASE_FIELD],
            current_field,
            *all_fields_data[current_field],
            e3sm_PPE_results=PPE_E3SM_res,
            e3sm_optimized_results=e3sm_results,
            constrained_opt=CONSTR_OPT,
        )

        fig = ax.get_figure()

        fig.savefig(
            outdir
            / f"Scatter_{BASE_FIELD}_future_{BASE_FIELD}_constr_by_{current_field}.pdf",
            bbox_inches="tight",
            dpi=300,
        )

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(argv)

    parser.add_argument(
        "-d",
        "--datapath",
        type=str,
        required=True,
        help="Path where the sensitivity and curvature matrices as well as stored computed values are located",
    )

    parser.add_argument(
        "--ppe_data",
        type=str,
        required=False,
        help="Path where the PPE is stored, in case it should be plotted",
    )
    parser.add_argument(
        "--ppe_data_sst4k",
        type=str,
        required=False,
        help="Path where the SST4K PPE is stored, in case it should be plotted",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        required=False,
        help="Path where the plots will be stored",
    )

    parser.add_argument(
        "--e3sm_results",
        type=str,
        required=False,
        help="Path where E3SM results of the optimized parameters are stored",
    )

    parser.add_argument(
        "--e3sm_results_sst4k",
        type=str,
        required=False,
        help="Path where E3SM results of the optimized parameters are stored",
    )

    parser.add_argument(
        "--load_data",
        action="store_true",
        default=False,
        required=False,
        help="Load and store computed data (WARNING: can combine configurations, resulting in incorrect plots)",
    )

    parser.add_argument(
        "--constr_opt",
        action="store_true",
        default=False,
        required=False,
        help="Use constrained optimization instead of ratio maximization",
    )

    parser.add_argument(
        "--constr_value",
        type=float,
        default=1.1937,
        required=False,
        help="Value used to constrain the present-day base field result",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        default=False,
        required=False,
        help="Only run the optimizations and return the results for automatic testing"
    )

    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    main()
