
"""
In this file, users may specify input data to quadtune_driver.
This includes assigning filenames for input netcdf files,
regional metric weights, and observed values of parameters.
"""

import os
from typing import Callable
import numpy as np
import pandas as pd
import sys



def config_core():

    from set_up_inputs import (
        setUp_x_MetricsList,
        setUpDefaultMetricValsCol, setUp_x_ObsMetricValsDict,
        calcObsGlobalAvgCol
    )

    thisfile_abspath = os.path.dirname(os.path.abspath(__file__))

    # Flag for using bootstrap sampling
    doBootstrapSampling = False


    # Flag for enabling additional output of multiple functions
    beVerbose = False

    # L1 regularization coefficient, i.e., penalty on param perturbations in objFnc
    # Increase this value to 0.1 or 0.5 or so if you want to eliminate
    # unimportant parameters.
    reglrCoef = 0.0

    # Non-dimensional pre-factor of penalty term in loss function that penalizes when
    #   the tuner leaves a global-mean bias, i.e., when the residuals don't sum to zero.
    #   Set to 1.0 for a "medium" penalty, and set to 0.0 for no penalty.
    penaltyCoef = 0.0


    doCreatePlots = False

    # doPiecewise = True if using a piecewise linear emulator
    doPiecewise = False

    # Flag to enable reading SST4K regional files
    doMaximizeRatio = False
    

    # Make varPrefixes a global variable so that it can be used in config_plots
    varPrefixes = ['SWCF']

    """ 
    Enable tests for debugging:
    
    - debug_level 0: Run no additional tests
    - debug_level 1: Run additional tests and stop upon failure
    
    """
    debug_level = 1

    # If debug_level > 0: Set delta_param for test
    # This will be applied to all parameters
    chosen_delta_param = 0.5 
    
    

    doObsOffset = False
    obsOffset = np.array([0])
    if ( len(obsOffset) != len(varPrefixes) ):
        sys.exit("Error: obsOffset must be the same size as the number of variables to tune.")

    # Number of metrics to tune.
    # If there are more metrics than this, then
    #   the metrics in the list beyond this number
    #   will appear in plots but not be counted in the tuning.
    boxSize = 20
    numBoxesInMap = np.rint( (360/boxSize) * (180/boxSize) )

    # numMetricsToTune includes all (e.g., 20x20 regions) and as many
    #   variables as we want to tune, up to all varPrefixes.
    numMetricsToTune = numBoxesInMap * len(varPrefixes)
    numMetricsToTune = numMetricsToTune.astype(int)

    obsOffsetCol = (obsOffset[:, np.newaxis] * np.ones((1, numBoxesInMap.astype(int)))).reshape(-1, 1)
                         

    # Directory where the regional files are stored (plus possibly a filename prefix)
    folder_name = thisfile_abspath  + '/../../tests/files_for_v1_paper/20.0sens1022_'


    # Netcdf file containing metric and parameter values from the default simulation
    defaultNcFilename = \
        (
            folder_name + '1_Regional.nc'
        )


    # Metrics from the global simulation that uses the tuner-recommended parameter values
    globTunedNcFilename = \
        (
            folder_name + '69_Regional.nc'
        )

    # Parameters are tunable model parameters, e.g. clubb_C8.
    # The float listed below after the parameter name is a factor that is used below for scaling plots.
    #   It is not a weight and doesn't affect optimized values; it just makes the plots more readable.
    # Each parameter is associated with two sensitivity simulations; in one, the parameter is perturbed
    #    up and in the other, it is perturbed down.
    #    The output from each sensitivity simulation is expected to be stored in its own netcdf file.
    #    Each netcdf file contains metric values and parameter values for a single simulation.
    paramsNamesScalesAndSuffixes = \
        [
        ['clubb_c8', 1.0e0,
         '14_Regional.nc',
         '15_Regional.nc'],
        ['clubb_c_invrs_tau_n2', 1.0,
         '10_Regional.nc',
         '11_Regional.nc'],
        ['clubb_c_invrs_tau_sfc', 1.0,
         '6_Regional.nc',
         '7_Regional.nc'],
        ['clubb_c_invrs_tau_wpxp_n2_thresh', 1.e3,
         '8_Regional.nc',
         '9_Regional.nc'],
        ['clubb_c_invrs_tau_n2_wp2', 1.0,
         '4_Regional.nc',
         '5_Regional.nc'],
        ]

    interactParamsNamesAndFilenames = \
    [
        ('clubb_c_invrs_tau_wpxp_n2_thresh', 'clubb_c8',
         'Regional_files/20241022_2yr_20x20regs/20sens1022_75_Regional.nc'),
    ]
    interactParamsNamesAndFilenames = []
    interactParamsNamesAndFilenamesType = np.dtype([('jParamName', object),
                                                    ('kParamName', object),
                                                    ('filename',   object)])
    interactParamsNamesAndFilenames = np.array(interactParamsNamesAndFilenames,
                                               dtype=interactParamsNamesAndFilenamesType)


    # Below we designate the subset of paramsNames that vary from [0,1] (e.g., C5)
    #    and hence will be transformed to [0,infinity] in order to make
    #    the relationship between parameters and metrics more linear:
    #transformedParamsNames = np.array(['clubb_c8','clubb_c_invrs_tau_n2', 'clubb_c_invrs_tau_n2_clear_wp3'])
    transformedParamsNames = np.array([''])

    prescribedParamsNamesScalesAndValues = \
        [
            
        ]

    
    
    # Read observed values of regional metrics on regular tiled grid into a Python dictionary
    (obsMetricValsDict, obsWeightsDict) = \
        (
    setUp_x_ObsMetricValsDict(varPrefixes, suffix='_[0-9]+_',
                      obsPathAndFilename=thisfile_abspath + '/../../tests/files_for_v1_paper/'
                             + '20.0_OBS.nc')
        )

#    # Add on RESTOM separately, since we typically want to prescribe its "observed" value

    # Set metricsNorms to be a global average

    obsGlobalAvgCol, obsGlobalStdCol, obsWeightsCol = \
    calcObsGlobalAvgCol(varPrefixes,
                        obsMetricValsDict, obsWeightsDict)

    # Warning: Using a global average as the constant weight produces little normalized
    #     sensitivity for PSL
    metricsNorms = np.copy(obsGlobalAvgCol)

    # Any special "custom" regions, e.g. DYCOMS, will be tacked onto the end of
    #     the usual metrics vectors.  But we exclude those regions from numMetricsNoCustom.
    

    # These are metrics from customized regions that differ from the standard 20x20 degree tiles.
    # Metrics are observed quantities that we want a tuned simulation to match.
    #    The first column is the metric name.
    #    The order of metricNames determines the order of rows in sensMatrix.
    # The second column is a vector of (positive) weights.  A small value de-emphasizes
    #   the corresponding metric in the fitting process.
    #   Use a large weight for global (GLB) metrics.
    # The third column is a vector of normalization values for metrics.
    #   If a value in the 3rd column is set to -999, then the metric is simply normalized by the observed value.
    #   Otherwise, the value in the 3rd column is itself the normalization value for the metric.
    metricsNamesWeightsAndNormsCustom = \
        [
            
        ]


    return ( numMetricsToTune,
     varPrefixes, boxSize,
     doCreatePlots, metricsNorms,
     obsMetricValsDict,
     obsOffsetCol, obsGlobalAvgCol, doObsOffset,
     obsWeightsCol,
     transformedParamsNames,
     defaultNcFilename, globTunedNcFilename,
     interactParamsNamesAndFilenames,
     doMaximizeRatio,
     doPiecewise,
     reglrCoef, penaltyCoef, doBootstrapSampling,
     paramsNamesScalesAndSuffixes, folder_name,
     prescribedParamsNamesScalesAndValues,
     metricsNamesWeightsAndNormsCustom,
     debug_level, chosen_delta_param, beVerbose)

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
    createPlotType: dict[str, bool] = {
        'paramsErrorBarsFig': False,               # Parameter values with error bars
        'biasesOrderedArrowFig': False,            # Predicted vs. actual global-model bias removal
        'threeDotFig': False,                      # Quadratic fnc for each metric and parameter
        'metricsBarChart': False,                  # Visualization of tuning matrix eqn
        'paramsIncrsBarChart': False,              # Mean parameter contributions to removal of biases
        'paramsAbsIncrsBarChart': False,           # Squared parameter contributions to bias removal
        'paramsTotContrbBarChart': False,          # Linear + nonlinear contributions to bias removal
        'biasesVsDiagnosticScatterplot': False,    # Scatterplot of biases vs. other fields
        'dpMin2PtFig': False,                      # Min param perturbation needed to simultaneously remove 2 biases
        'dpMinMatrixScatterFig': False,            # Scatterplot of min param perturbation for 2-bias removal
        'projectionMatrixFigs': False,             # Color-coded projection matrix
        'biasesVsSensMagScatterplot': False,       # Biases vs. parameter sensitivities
        'biasesVsSvdScatterplot': False,           # Left SV1*bias vs. left SV2*bias
        'paramsCorrArrayFig': False,               # Color-coded matrix showing correlations among parameters
        'sensMatrixAndBiasVecFig': False,          # Color-coded matrix equation
        'PcaBiplot': False,                        # Principal components biplot
        'PcSensMap': False,                        # Maps showing sensitivities to parameters and left singular vectors
        'vhMatrixFig': False,                      # Color-coded matrix of right singular vectors
        'lossFncVsParamFig': False,                # 2D loss function plots
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



def config_bootstrap(beVerbose: bool) -> tuple[int,str,str]:
    """
    Configure settings for bootstrap sampling.
    For example specify how many bootstrap samples to create.
    
    :param beVerbose: Boolean flag to make output more verbose.
    """
    numBootstrapSamples: int = 100

    # Directory where the SST4K regional files are stored (plus possibly a filename prefix)
    folder_name_SST4K = 'Regional_files/20241022_1yr_sst4k_20x20/20p4k1022_'

    defaultSST4KNcFilename = \
    (
        folder_name_SST4K + '1_Regional.nc'
    )

    return numBootstrapSamples, folder_name_SST4K, defaultSST4KNcFilename


def config_additional(beVerbose:bool) -> tuple[str, str]:
    """
    Configure additional settings.
    For example, specify SST4K filenames.
    """

    # Directory where the SST4K regional files are stored (plus possibly a filename prefix)
    folder_name_SST4K = ""

    defaultSST4KNcFilename = \
    (
        folder_name_SST4K + ""
    )

    return folder_name_SST4K, defaultSST4KNcFilename



