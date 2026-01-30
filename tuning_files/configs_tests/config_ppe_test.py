# -*- coding: utf-8 -*-

# Run this app with `python3 quadtune_driver.py` and
# view the plots at http://127.0.0.1:8050/ in your web browser.
# (To open a web browser on a larson-group computer,
# login to malan with `ssh -X` and then type `firefox &`.)

"""
In this file, users may specify input data to quadtune_driver.
This includes assigning filenames for input netcdf files,
regional metric weights, and observed values of parameters.
"""

import os
import sys
from typing import Callable
import numpy as np

def config_core():
    """
    Mandatory configuration to run quadtune using PPE data
    """



    thisfile_abspath = os.path.dirname(os.path.abspath(__file__))

    '''
    Configure Filenames
    '''
    PPE_metrics_filename = thisfile_abspath + "/../../tests/files_for_ppe_test/PPEMetricsData.nc"

    PPE_params_filename = thisfile_abspath + "/../../tests/files_for_ppe_test/PPEParamsData.nc"

    '''
    Set flags to enable optional functionality
    '''

    # Flag for using bootstrap sampling
    doBootstrapSampling = False

    # doPiecewise = True if using a piecewise linear emulator
    doPiecewise = False

    # Flag for enabling additional output of multiple functions
    beVerbose = False

    # L1 regularization coefficient, i.e., penalty on param perturbations in objFnc
    # Increase this value to 0.1 or 0.5 or so if you want to eliminate
    # unimportant parameters.
    reglrCoef = 0.0

    # Non-dimensional pre-factor of penalty term in loss function that penalizes when
    #   the tuner leaves a global-mean bias, i.e., when the residuals don't sum to zero.
    #   Set to 1.0 for a "medium" penalty, and set to 0.0 for no penalty.
    penaltyCoef = np.array([0.0])

    # Use these flags to determine whether or not to create specific plots
    #    in create_nonbootstrap_figs.py
    doCreatePlots = False

    # Flag to enable reading SST4K regional files
    doMaximizeRatio = False

    # Set debug level
    debug_level = 0
    # Set perturbation for the recovery test
    recovery_test_dparam = 0.5


    # Flag for whether to bound quadtune parameter values to remain within
    # the range spanned by the default and sensitivity runs.
    doSensParamBounds = False


    '''
    Configure the resolution of the Data and which metrics and parameters to use
    '''

    varPrefixes = ['SWCF']

    boxSize = 20

    numBoxesInMap = (360//boxSize)*(180//boxSize)

    numMetricsToTune = len(varPrefixes) * numBoxesInMap

    # Flag for whether to weight certain regions more (or less) than others
    doWeightRegions = False

    # Dictionary for custom-weighted regions. The entries should follow the
    # pattern 'region':factor, where 'region' is of the form '2_10' (latitude
    # index followed by longitude index) and the factor is a multiplicative
    # factor that will be applied to the weight for that region (i.e. factor=1.0 
    # would make no difference).
    weightedRegionsDict = {'5_14':2.0}


    allparamsNamesInFile = ['cldfrc_dp2',
 'clubb_altitude_threshold',
 'clubb_c11',
 'clubb_c7',
 'clubb_c8',
 'clubb_c_invrs_tau_bkgnd',
 'clubb_c_invrs_tau_n2',
 'clubb_c_invrs_tau_n2_clear_wp3',
 'clubb_c_invrs_tau_n2_wp2',
 'clubb_c_invrs_tau_n2_wpxp',
 'clubb_c_invrs_tau_n2_xp2',
 'clubb_c_invrs_tau_sfc',
 'clubb_c_invrs_tau_shear',
 'clubb_c_invrs_tau_wpxp_n2_thresh',
 'clubb_c_k10',
 'clubb_c_k8',
 'clubb_c_wp2_splat',
 'clubb_gamma_coef',
 'clubb_nu1',
 'clubb_nu2',
 'clubb_z_displace',
 'micro_mg_autocon_lwp_exp',
 'micro_mg_dcs']
    
    paramsNamesAndScales = np.array(
    [
        ['clubb_c8',1.0e0],
        ['clubb_c_invrs_tau_n2',1.0e0],
        ['clubb_c_invrs_tau_n2_wp2',1.0e0],
        ['clubb_c_invrs_tau_sfc',1.0e0],
        ['clubb_c_invrs_tau_wpxp_n2_thresh',1.e3],

    ]
    )

    

    """
    Configure additional necessary information
    """

    interactParamsNamesAndFilenames = []
    interactParamsNamesAndFilenamesType = np.dtype([('jParamName', object),
                                                    ('kParamName', object),
                                                    ('filename',   object)])
    interactParamsNamesAndFilenames = np.array(interactParamsNamesAndFilenames,
                                               dtype=interactParamsNamesAndFilenamesType)



    return (numMetricsToTune,
         varPrefixes, boxSize,
         doCreatePlots,
         PPE_metrics_filename, PPE_params_filename,
         doMaximizeRatio,
         doPiecewise,
         interactParamsNamesAndFilenames,
         reglrCoef, penaltyCoef, doBootstrapSampling,
         paramsNamesAndScales, allparamsNamesInFile,
         debug_level, recovery_test_dparam,
         doSensParamBounds,
         doWeightRegions, weightedRegionsDict,
         beVerbose)




def config_plots(beVerbose: bool, varPrefixes:list[str], paramsNames:list[str]) -> tuple[dict[str, bool], np.ndarray, int, Callable]:
    """
    Configure settings for creating plots.
    For example, specify which plots to create.
    
    :param beVerbose: Boolean flag to make output more verbose.
    """

    def abbreviateParamsNames(paramsNames):
        """
        Abbreviate parameter names so that they fit on plots.
        This is handled manually with the lines of code below.
        """

        paramsAbbrv = np.char.replace(paramsNames, 'clubb_', '')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'c_invrs_tau_', '')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'wpxp_n2', 'n2')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'altitude', 'alt')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'threshold', 'thres')
        paramsAbbrv = np.char.replace(paramsAbbrv, 'thresh', 'thres')

        return paramsAbbrv

    # Use these flags to determine whether or not to create specific plots
    # in create_nonbootstrap_figs.py
    createPlotType = {
        'paramsErrorBarsFig': False,               # Parameter values with error bars
        'biasesOrderedArrowFig': False,            # Predicted vs. actual global-model bias removal
        'threeDotFig': False,                       # Quadratic fnc for each metric and parameter
        'metricsBarChart': True,                   # Visualization of tuning matrix eqn
        'paramsIncrsBarChart': True,               # Mean parameter contributions to removal of biases
        'paramsAbsIncrsBarChart': True,            # Squared parameter contributions to bias removal
        'paramsTotContrbBarChart': False,          # Linear + nonlinear contributions to bias removal
        'biasesVsDiagnosticScatterplot': False,    # Scatterplot of biases vs. other fields
        'dpMin2PtFig': False,                      # Min param perturbation needed to simultaneously remove 2 biases
        'dpMinMatrixScatterFig': False,            # Scatterplot of min param perturbation for 2-bias removal
        'projectionMatrixFigs': False,             # Color-coded projection matrix
        'biasesVsSensMagScatterplot': True,        # Biases vs. parameter sensitivities
        'biasesVsSvdScatterplot': False,           # Left SV1*bias vs. left SV2*bias
        'paramsCorrArrayFig': True,                # Color-coded matrix showing correlations among parameters
        'sensMatrixAndBiasVecFig': False,          # Color-coded matrix equation
        'PcaBiplot': False,                        # Principal components biplot
        'PcSensMap': True,                         # Maps showing sensitivities to parameters and left singular vectors
        'vhMatrixFig': True,                       # Color-coded matrix of right singular vectors
        'lossFncVsParamFig': True,                 # 2D loss function plots
        'SST4KPanelGallery': True                  # Maps showing metrics perturbation for parameters from Generalized Eigenvalue problem
    }

    if beVerbose:
        print(f"Creating {sum(createPlotType.values())} types of plots.")



    # mapVarIdx is the field is plotted in the 20x20 maps created by PcSensMap.
    mapVar = 'SWCF'
    mapVarIdx = varPrefixes.index(mapVar)

    # These are a selected subset of the tunable metrics that we want to include
    # in the metrics bar-chart, 3-dot plot, etc.
    # They must be a subset of metricsNames
    highlightedRegionsToPlot = np.array(['1_6', '1_14', '3_6', '3_14',
                                         '6_14', '6_18', '8_13'])
    mapVarPlusUnderscore = mapVar + '_'
    highlightedMetricsToPlot = np.char.add(mapVarPlusUnderscore, highlightedRegionsToPlot)      


    return createPlotType, highlightedMetricsToPlot, mapVarIdx, abbreviateParamsNames
