# -*- coding: utf-8 -*-

"""
Run this app with something like:

 `python3 quadtune_driver.py --config_filename config_default.py`

or

 `python3 quadtune_driver.py -c config_example.py`  
 
and view the plots at http://127.0.0.1:8050/ in your web browser.
(To open a web browser on a larson-group computer,
login to malan with `ssh -X` and then type `firefox &`.)
"""


import importlib
import os
import sys
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.optimize import Bounds
from scipy.linalg import eigh
from scipy.interpolate import UnivariateSpline
from sklearn import linear_model

import argparse
import sys


import matplotlib.pyplot as plt

def main(args):
    """
    Main driver for QuadTune.
    It calls routines to feeds in input configuration data,
    find optimal parameter values, and
    create diagnostic plots.
    """

    """
    The following line adds the current directory (quadtune/tuning_files) to the python path, so that imports from, e.g., set_up_inputs work,
    even if quadtune_driver.py is called from tests/ or another directory.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

    from set_up_inputs \
        import setUpColAndRowVectors, \
               setUpDefaultMetricValsCol, \
               createInteractIdxs, \
               calcNormlzdInteractBiasesCols, \
               readDnormlzdParamsInteract, \
               calcInteractDerivs, \
               checkInteractParamVals, \
               printInteractDiagnostics, \
               checkInteractDerivs, \
               checkPiecewiseLeftRightPoints

    from create_nonbootstrap_figs import createFigs
    from create_bootstrap_figs import bootstrapPlots
    from do_bootstrap_calcs import bootstrapCalculations

    import process_config_info

    #Parse the argument to get the config filename and import setUpConfig from that file | !! Potentially unsafe -> Import arbitrary function !!
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--config_filename", type=str,required=True,help="Please provide the filename of your config file, e.g., config_default.py")
    parser.add_argument("--ppe", action="store_true", help="Indicate that the data provided is PPE data.")

    args = parser.parse_args(args)
    config_file = importlib.import_module(args.config_filename.replace('.py',''))
    doUsePPEdata = args.ppe

    print("Set up inputs . . .")


    if doUsePPEdata:

        import read_PPE_files

        from set_up_inputs import calcObsGlobalAvgCol

        (numMetricsToTune,
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
         beVerbose) = \
            config_file.config_core()
        

        defaultParamValsOrigRow, PPE_params, paramsNames, paramsIndices, paramsScales, magParamValsRow, normlzdOrdDparamsMinFlat, normlzdOrdDparamsMaxFlat =\
            read_PPE_files.process_PPE_params_file(PPE_params_filename, paramsNamesAndScales, allparamsNamesInFile)
        

        defaultMetricValsCol, PPE_metrics, metricsNames, metricsWeights, obsMetricValsCol, obsWeightsCol =\
              read_PPE_files.process_ppe_metrics_file(PPE_metrics_filename, varPrefixes, boxSize)
        

        normlzdOrdDparamsMin = np.tile(normlzdOrdDparamsMinFlat,(len(metricsNames),1))
        normlzdOrdDparamsMax = np.tile(normlzdOrdDparamsMaxFlat,(len(metricsNames),1))
        
        metricGlobalAvgs = np.diag(np.dot(metricsWeights.reshape(-1, len(varPrefixes), order='F').T,
                                      defaultMetricValsCol.reshape(-1, len(varPrefixes), order='F')))


        obsMetricValsDict, obsWeightsDict =\
              read_PPE_files.setUp_x_ObsMetricValsDictPPE(obsMetricValsCol.flatten(),obsWeightsCol.flatten(),metricsNames)

        obsGlobalAvgCol, obsGlobalStdCol, obsWeightsCol = \
        calcObsGlobalAvgCol(varPrefixes,
                        obsMetricValsDict, obsWeightsDict)
        
        
        metricsNorms = np.copy(obsGlobalAvgCol)

        defaultBiasesCol = defaultMetricValsCol - obsMetricValsCol


        normMetricValsCol = np.copy(metricsNorms)
        for idx in np.arange(len(metricsNorms)):
            if np.isclose(metricsNorms[idx],-999.0): 
                normMetricValsCol[idx] = obsMetricValsCol[idx]

        

        

        sensParamValsRow = defaultParamValsOrigRow
        sensParamValsRowExt = defaultParamValsOrigRow
        transformedParamsNames = ['']

        normlzd_dpMid = np.zeros_like(sensParamValsRow)

        normlzdLeftSensMatrix = None
        normlzdRightSensMatrix = None
        globTunedNcFilename = None

        if doCreatePlots:
            numMetricsNoCustom = len(metricsNames)
            metricsNamesNoprefix = metricsNames
            
            sensNcFilenames = None
            sensNcFilenamesExt = None
            defaultNcFilename = None

    else:

        # The user should input all tuning configuration info into file set_up_inputs.py
        (numMetricsToTune,
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
        paramsNamesScalesAndFilenames, folder_name,
        prescribedParamsNamesScalesAndValues,
        metricsNamesWeightsAndNormsCustom, 
        debug_level, recovery_test_dparam, doSensParamBounds,
        doWeightRegions, weightedRegionsDict, beVerbose) \
        = \
            config_file.config_core()

        # Process configuration

        (paramsNames, paramsScales,
        sensNcFilenames,sensNcFilenamesExt) = \
            process_config_info.process_paramsnames_scales_and_filesuffixes(paramsNamesScalesAndFilenames, folder_name)

        (prescribedParamsNames, prescribedParamsScales,
        prescribedParamValsRow, prescribedSensNcFilenames,
        prescribedSensNcFilenamesExt,
        prescribedTransformedParamsNames) = \
            process_config_info.process_prescribed_paramsnames(prescribedParamsNamesScalesAndValues, folder_name)

        (metricsNames, metricsWeights, metricGlobalAvgs, numMetricsNoCustom) = \
            process_config_info.process_metrics_names_weights_norms(defaultNcFilename, varPrefixes)



        (metricsNames, metricsWeights,metricsNorms, metricsNamesNoprefix) = \
            process_config_info.process_metrics_names_weights_norms_custom(metricsNamesWeightsAndNormsCustom, metricsNames,
                                                                                metricsWeights, metricsNorms)
        
        # print the global mean metric values for convenient comparison to obs values
    # metricGlobalAvgs is based on the default test (defaultNcFilename) for the ensemble 
    # (see the lines of code immediately above this)
    for prefix_idx in range(len(varPrefixes)):
        print(f"metricGlobalAvgs for {varPrefixes[prefix_idx]} = ", metricGlobalAvgs[prefix_idx])

    # recalculate metricsWeights if we are using custom weighting on certain regions (i.e. doWeightRegions = True)
    # ***metricsWeights SHOULD NOT be altered after this point***
    if doWeightRegions:
        metricsWeights = reweight_regions_by_weightedRegionsDict(boxSize, numMetricsToTune, weightedRegionsDict,
                                                                 metricsWeights)
    if doCreatePlots:
        createPlotType, highlightedMetricsToPlot, mapVarIdx, abbreviateParamsNames  = \
            config_file.config_plots(beVerbose, varPrefixes = varPrefixes, paramsNames = paramsNames)

        paramsAbbrv = process_config_info.abbreviate_params_names(paramsNames, abbreviateParamsNames)

    if doBootstrapSampling:
        numBootstrapSamples =\
              config_file.config_bootstrap(beVerbose)
        
    if doMaximizeRatio or doBootstrapSampling:
        folder_name_SST4K, defaultSST4KNcFilename = config_file.config_additional(beVerbose)

        assert folder_name_SST4K != '', "folder_name_SST4K and defaultSST4KNcFilename must be provided if doCalcGenEig or doBootstrapSampling is True"
        _, _ , sensSST4KNcFilenames, sensSST4KNcFilenamesExt  =\
              process_config_info.process_paramsnames_scales_and_filesuffixes(paramsNamesScalesAndFilenames, folder_name_SST4K)


    # Number of regional metrics, including all of varPrefixes including the metrics we're not tuning, plus custom regions.
    numMetrics = len(metricsNames)

    # Save the original metricsWeights for global-avg diagnostics
    metricsWeightsDiagnostic = np.copy(metricsWeights)

    # We apply a tiny weight to the final metrics.
    #    Those metrics will appear in the diagnostics
    #    but their errors will not be accounted for in tuning.
    metricsWeights[numMetricsToTune:] = 1e-12

    print("Set up preliminaries . . .")

    if doUsePPEdata:
        normlzdSensMatrixPoly,normlzdCurvMatrix =\
            read_PPE_files.construct_sensitivity_curvature_matrices_from_PPE_data(\
                PPE_metrics, defaultMetricValsCol, PPE_params, defaultParamValsOrigRow, normMetricValsCol, magParamValsRow)
        normlzdConstMatrix = np.zeros_like(normlzdCurvMatrix)

    else:

        obsMetricValsCol, normMetricValsCol, \
        defaultBiasesCol, \
        defaultParamValsOrigRow, \
        sensParamValsRow, sensParamValsRowExt, \
        dnormlzdSensParams, \
        magParamValsRow, \
        dnormlzdPrescribedParams, \
        magPrescribedParamValsRow, \
        = setUpColAndRowVectors(metricsNames, metricsNorms,
                                obsMetricValsDict,
                                obsOffsetCol, obsGlobalAvgCol, doObsOffset,
                                paramsNames, transformedParamsNames, prescribedParamsNames, prescribedParamValsRow,
                                prescribedTransformedParamsNames,
                                sensNcFilenames, sensNcFilenamesExt,
                                defaultNcFilename
                                )


        # Construct numMetrics x numParams matrix of second derivatives, d2metrics/dparams2.
        #     The derivatives are normalized by observed metric values and max param values.
        # Also construct a linear sensitivity matrix, dmetrics/dparams.
        normlzdCurvMatrix, normlzdSensMatrixPoly, normlzdConstMatrix, \
        normlzdOrdDparamsMin, normlzdOrdDparamsMax, \
        normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix, \
        normlzd_pLeftRow, normlzd_pMidRow, normlzd_pRightRow = \
            constructNormlzdSensCurvMatrices(metricsNames, paramsNames, transformedParamsNames,
                                    normMetricValsCol, magParamValsRow,
                                    sensNcFilenames, sensNcFilenamesExt, defaultNcFilename)
        

        # For prescribed parameters, construct numMetrics x numParams matrix of second derivatives, d2metrics/dparams2.
        # The derivatives are normalized by observed metric values and max param values.
        normlzdPrescribedCurvMatrix, normlzdPrescribedSensMatrixPoly, normlzdPrescribedConstMatrix, \
        normlzdPrescribedOrdDparamsMin, normlzdPrescribedOrdDparamsMax, \
        normlzdPrescribed_dpMid, normlzdPrescribedLeftSensMatrix, normlzdPrescribedRightSensMatrix, \
        normlzdPrescribed_pLeftRow, normlzdPrescribed_pMidRow, normlzdPrescribed_pRightRow = \
            constructNormlzdSensCurvMatrices(metricsNames, prescribedParamsNames, prescribedTransformedParamsNames,
                                    normMetricValsCol, magPrescribedParamValsRow,
                                    prescribedSensNcFilenames, prescribedSensNcFilenamesExt, defaultNcFilename)

        # This is the prescribed correction to the metrics that appears on the left-hand side of the Taylor equation.
        #   It is not a bias from the obs.  It is a correction to the simulated default metric values
        #   based on prescribed param values.

        ### THIS CALL DOESN'T ACCOUNT FOR INTERACTIONS!!!
        normlzdPrescribedBiasesCol = \
            fwdFnc(dnormlzdPrescribedParams, normlzdPrescribedSensMatrixPoly, normlzdPrescribedCurvMatrix,
                    doPiecewise, normlzdPrescribed_dpMid,
                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                    numMetrics,
                    normlzdInteractDerivs= np.empty(0), interactIdxs = np.empty(0))

        prescribedBiasesCol = normlzdPrescribedBiasesCol * np.abs(normMetricValsCol)

        # defaultBiasesCol + prescribedBiasesCol = -fwdFnc_tuned_params  (see lossFnc).
        #     This lumps the prescribed-parameter adjustment into defaultBiasesCol.
        #        but it may be clearer to separate them out.
        # defaultBiasesCol = default simulation - observations
        defaultBiasesCol = defaultBiasesCol + prescribedBiasesCol

        
    obsMetricValsAvgs = np.diag(np.dot(obsWeightsCol.reshape(-1, len(varPrefixes), order='F').T,
                            obsMetricValsCol.reshape(-1, len(varPrefixes), order='F')))
    print(f"\nobsMetricValsAvgs (including any offsets) = {obsMetricValsAvgs}")

    # In order to weight certain metrics, multiply each row of normlzdSensMatrixPoly
    # by metricsWeights
    normlzdWeightedSensMatrixPoly = np.diag(np.transpose(metricsWeights)[0]) @ normlzdSensMatrixPoly

    # interactIdxs = array of numInteractTerms (j,k) tuples of parameter indices of interaction terms
    interactIdxs = createInteractIdxs(interactParamsNamesAndFilenames, paramsNames)

    checkInteractParamVals(
        interactIdxs, interactParamsNamesAndFilenames,
        sensParamValsRow, sensParamValsRowExt,
        defaultParamValsOrigRow,
        paramsNames, transformedParamsNames,
        len(paramsNames))

    defaultMetricValsCol = obsMetricValsCol + defaultBiasesCol

    # normlzdInteractBiasesCols = numMetrics x numInteractTerms array
    normlzdInteractBiasesCols = \
              calcNormlzdInteractBiasesCols( defaultMetricValsCol,
                              normMetricValsCol,
                              metricsNames,
                              interactParamsNamesAndFilenames)

    # dnormlzdParamsInteract = array of numInteractTerms tuples of parameter *values*
    dnormlzdParamsInteract = \
        readDnormlzdParamsInteract(interactParamsNamesAndFilenames, interactIdxs,
                               defaultParamValsOrigRow, magParamValsRow,
                               paramsNames, transformedParamsNames, len(paramsNames))

    # normlzdInteractDerivs = numMetrics x numInteractTerms array
    normlzdInteractDerivs = calcInteractDerivs(interactIdxs,
                       dnormlzdParamsInteract,
                       normlzdInteractBiasesCols,
                       normlzdCurvMatrix, normlzdSensMatrixPoly,
                       doPiecewise, normlzd_dpMid,
                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
                       numMetrics)

    checkInteractDerivs(normlzdInteractBiasesCols,
                        dnormlzdParamsInteract,
                        len(paramsNames),
                        normlzdSensMatrixPoly, normlzdCurvMatrix,
                        doPiecewise, normlzd_dpMid,
                        normlzdLeftSensMatrix, normlzdRightSensMatrix,
                        numMetrics,
                        normlzdInteractDerivs, interactIdxs)

    if debug_level > 0 :
        recovery_test_dparam = recovery_test_dparam *  np.ones((len(paramsNames),1))
        check_recovery_of_param_vals(debug_level, recovery_test_dparam, normlzdCurvMatrix, normlzdSensMatrixPoly,\
                                doPiecewise, normlzd_dpMid, normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                                  normlzdLeftSensMatrix, normlzdRightSensMatrix, numMetrics, normlzdInteractDerivs, interactIdxs,\
                                    metricsNames, metricsWeights, normMetricValsCol, magParamValsRow, defaultParamValsOrigRow, reglrCoef, penaltyCoef, beVerbose)
    normlzdDefaultBiasesCol = defaultBiasesCol / np.abs(normMetricValsCol)

    print("Optimizing parameter values . . . ")

    # sValsRatio = a threshold ratio of largest singular value
    #              to the smallest retained singular value.
    # If sValsRatio is large enough, then all singular vectors will be kept.
    # If sValsRatio is 1, then only the first singular vector will be kept.
    sValsRatio = 800.
    normlzdSensMatrixPolySvd = \
        approxMatrixWithSvd(normlzdSensMatrixPoly, sValsRatio, sValsNumToKeep=None, beVerbose=beVerbose)
    normlzdCurvMatrixSvd = \
        approxMatrixWithSvd(normlzdCurvMatrix, sValsRatio, sValsNumToKeep=None, beVerbose=beVerbose)

    if not doUsePPEdata:
        # Check whether piecewise-linear and quadratic emulators agree at lo/hi parameter values
        dLeftRightParams = ( normlzd_pLeftRow - defaultParamValsOrigRow * np.reciprocal(magParamValsRow) ).T
        checkPiecewiseLeftRightPoints(dLeftRightParams,
                                    normlzdSensMatrixPolySvd, normlzdCurvMatrix,
                                    normlzd_dpMid,
                                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                    numMetrics)
        dLeftRightParams = ( normlzd_pRightRow - defaultParamValsOrigRow * np.reciprocal(magParamValsRow) ).T
        checkPiecewiseLeftRightPoints(dLeftRightParams,
                                    normlzdSensMatrixPolySvd, normlzdCurvMatrix,
                                    normlzd_dpMid,
                                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                    numMetrics)
    

    if doMaximizeRatio or doBootstrapSampling:
        # SST4K:  call constructNormlzdSensCurvMatrices with SST4K sensFiles.

        # For SST4K runs,
        #     construct numMetrics x numParams matrix of second derivatives, d2metrics/dparams2.
        #     The derivatives are normalized by observed metric values and max param values.
        # Also construct a linear sensitivity matrix, dmetrics/dparams.
        normlzdCurvMatrixSST4K, normlzdSensMatrixPolySST4K, normlzdConstMatrixSST4K, \
        normlzdOrdDparamsMinSST4K, normlzdOrdDparamsMaxSST4K, \
        normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K, \
        normlzd_pLeftRowSST4K, normlzd_pMidRowSST4K, normlzd_pRightRowSST4K = \
            constructNormlzdSensCurvMatrices(metricsNames, paramsNames, transformedParamsNames,
                                             normMetricValsCol, magParamValsRow,
                                             sensSST4KNcFilenames, sensSST4KNcFilenamesExt, defaultSST4KNcFilename)

    #######################################################################################################
    #
    # Calculate an ensemble of parameter values by doing bootstrap sampling of the regional metrics.
    #
    #######################################################################################################

    if doBootstrapSampling:

        print("Starting bootstrap sampling . . .")

        

        # SST4K: Here feed normlzdSensMatrixPoly and normlzdCurvMatrix from SST4K runs into bootstrapCalculations.

        ( paramsBoot, paramsTuned, residualsDefaultCol, residualsTunedCol,
          residualsBootstrapMatrix,
          residualsBootstrapMatrixSST4K,
          defaultBiasesApproxNonlinMatrix,
          defaultBiasesApproxNonlinMatrixSST4K,
          dDefaultBiasesApproxNonlinMatrixSST4K,
          paramBoundsBoot,
          normResidualPairsMatrix, tradeoffBinaryMatrix ) = \
        bootstrapCalculations(numBootstrapSamples,
                              metricsWeights,
                              metricsNames,
                              paramsNames,
                              numMetrics,  # numMetrics is redundant, given that we're feeding in metricsNames
                              numMetricsToTune,
                              normMetricValsCol,
                              magParamValsRow,
                              defaultParamValsOrigRow,
                              normlzdOrdDparamsMin,
                              normlzdOrdDparamsMax,
                              normlzdSensMatrixPoly,
                              normlzdSensMatrixPolySST4K,
                              normlzdDefaultBiasesCol,
                              normlzdCurvMatrix,
                              normlzdCurvMatrixSST4K,
                              doPiecewise, normlzd_dpMid,
                              normlzdLeftSensMatrix, normlzdRightSensMatrix,
                              reglrCoef, penaltyCoef,
                              defaultBiasesCol,
                              normlzdInteractDerivs, interactIdxs
                              )

        print(f"Sample avg of paramsBoot = {np.mean(paramsBoot, axis=0)}")

        bootstrapPlots(numMetricsToTune,  # Should we feed in numMetrics instead??
                       boxSize,
                       metricsNames,
                       residualsBootstrapMatrix,
                       residualsBootstrapMatrixSST4K,
                       defaultBiasesApproxNonlinMatrix,
                       defaultBiasesApproxNonlinMatrixSST4K,
                       dDefaultBiasesApproxNonlinMatrixSST4K,
                       residualsTunedCol,
                       residualsDefaultCol,
                       paramsNames,
                       paramsBoot,
                       paramsTuned,
                       defaultParamValsOrigRow,
                       paramBoundsBoot,
                       normResidualPairsMatrix,
                       tradeoffBinaryMatrix)
    else:
        paramBoundsBoot = None

    #end if doBootstrapSampling

    ########################################
    #
    # Resume non-bootstrap calculations
    #
    #########################################


    defaultBiasesApproxNonlin, \
    dnormlzdParamsSolnNonlin, paramsSolnNonlin, \
    dnormlzdParamsSolnLin, paramsSolnLin, \
    defaultBiasesApproxNonlin2x, \
    defaultBiasesApproxNonlinNoCurv, defaultBiasesApproxNonlin2xCurv = \
        solveUsingNonlin(metricsNames,
                         metricsWeights, normMetricValsCol, magParamValsRow,
                         defaultParamValsOrigRow,
                         normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                         normlzdSensMatrixPolySvd, normlzdDefaultBiasesCol,
                         normlzdCurvMatrixSvd,
                         doPiecewise, normlzd_dpMid,
                         normlzdLeftSensMatrix, normlzdRightSensMatrix,
                         normlzdInteractDerivs, interactIdxs,
                         reglrCoef, penaltyCoef,
                         doSensParamBounds, beVerbose)

    y_hat_i = defaultBiasesApproxNonlin + defaultBiasesCol + obsMetricValsCol

    tunedMetricGlobalAvgs = np.diag(np.dot(metricsWeightsDiagnostic.reshape(-1, len(varPrefixes), order='F').T,
                                (defaultBiasesApproxNonlin + defaultBiasesCol).reshape(-1, len(varPrefixes), order='F'))) \
                            + obsMetricValsAvgs
    print(f"\ntunedMetricGlobalAvgs = {tunedMetricGlobalAvgs}")

    #print("Tuned parameter perturbation values (dnormzldParamsSolnNonlin)")
    #for idx in range(0,len(paramsNames)): \
    #    print("{:33s} {:7.7g}".format(paramsNames[idx], dnormlzdParamsSolnNonlin[idx][0] ) )
    print("\nTuned parameter values (paramsSolnNonlin)          Default value")
    for idx in range(0,len(paramsNames)): \
        print("{:33s} {:15.7f} {:15.7f}".format(paramsNames[idx], paramsSolnNonlin[idx][0], defaultParamValsOrigRow[0][idx] ) )

    # Check whether the minimizer actually reduces chisqd
    # Initial value of chisqd, which assumes parameter perturbations are zero
    #normlzdWeightedDefaultBiasesCol = metricsWeights * normlzdDefaultBiasesCol
    chisqdZero = lossFnc(np.zeros_like(defaultParamValsOrigRow),
                         normlzdSensMatrixPoly, normlzdDefaultBiasesCol, metricsWeights,
                         normlzdCurvMatrix,
                         doPiecewise, normlzd_dpMid,
                         normlzdLeftSensMatrix, normlzdRightSensMatrix,
                         numMetrics,
                         normlzdInteractDerivs, interactIdxs)  # Should I feed in numMetricsToTune instead??
                                                                    #   But metricsWeights is already set to eps for un-tuned metrics.
    # Optimized value of chisqd, which uses optimal values of parameter perturbations
    chisqdMin = lossFnc(dnormlzdParamsSolnNonlin.T,
                        normlzdSensMatrixPoly, normlzdDefaultBiasesCol, metricsWeights,
                        normlzdCurvMatrix,
                        doPiecewise, normlzd_dpMid,
                        normlzdLeftSensMatrix, normlzdRightSensMatrix,
                        numMetrics,
                        normlzdInteractDerivs, interactIdxs)  # Should I feed in numMetricsToTune instead??

    print("chisqdMinRatio (all metrics, non-unity metricsWeights) =", chisqdMin/chisqdZero)

    chisqdUnweightedZero = lossFnc(np.zeros_like(defaultParamValsOrigRow),
                                   normlzdSensMatrixPoly, normlzdDefaultBiasesCol, np.ones_like(metricsWeights),
                                   normlzdCurvMatrix,
                                   doPiecewise, normlzd_dpMid,
                                   normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                   numMetrics,
                                   normlzdInteractDerivs, interactIdxs)  # Should I feed in numMetricsToTune instead??
    # Optimized value of chisqd, which uses optimal values of parameter perturbations
    chisqdUnweightedMin = lossFnc(dnormlzdParamsSolnNonlin.T,
                                  normlzdSensMatrixPoly, normlzdDefaultBiasesCol, np.ones_like(metricsWeights),
                                  normlzdCurvMatrix,
                                  doPiecewise, normlzd_dpMid,
                                  normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                  numMetrics,
                                  normlzdInteractDerivs, interactIdxs)  # Should I feed in numMetricsToTune instead??

    print("chisqdUnweightedMinRatio (all metrics, metricsWeights=1) =", chisqdUnweightedMin/chisqdUnweightedZero)

    # Set up a column vector of metric values from the global simulation based on optimized
    #     parameter values.

    if globTunedNcFilename is None:
        globTunedBiasesCol = np.zeros_like(obsMetricValsCol)
    else:
        globTunedMetricValsCol = setUpDefaultMetricValsCol(metricsNames, globTunedNcFilename)

        # Store biases in default simulation, ( global_model - obs )
        globTunedBiasesCol = np.subtract(globTunedMetricValsCol, obsMetricValsCol)
        #globTunedBiasesCol = globTunedBiasesCol + prescribedBiasesCol

    # Check whether the minimizer actually reduces chisqd
    # Initial value of chisqd, which assumes parameter perturbations are zero
    normlzdGlobTunedBiasesCol = globTunedBiasesCol/np.abs(normMetricValsCol)
    chisqdGlobTunedMin = lossFnc(np.zeros_like(defaultParamValsOrigRow),
                                 normlzdSensMatrixPoly, normlzdGlobTunedBiasesCol, metricsWeights,
                                 normlzdCurvMatrix,
                                 doPiecewise, normlzd_dpMid,
                                 normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                 numMetrics, # Should I feed in numMetricsToTune instead??
                                 normlzdInteractDerivs, interactIdxs)

    print("chisqdGlobTunedMinRatio =", chisqdGlobTunedMin/chisqdZero)

    chisqdUnweightedGlobTunedMin = lossFnc(np.zeros_like(defaultParamValsOrigRow),
                                           normlzdSensMatrixPoly, normlzdGlobTunedBiasesCol, np.ones_like(metricsWeights),
                                           normlzdCurvMatrix,
                                           doPiecewise, normlzd_dpMid,
                                           normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                           numMetrics,  # Should I feed in numMetricsToTune instead??
                                           normlzdInteractDerivs, interactIdxs)

    print("chisqdUnweightedGlobTunedMinRatio =", chisqdUnweightedGlobTunedMin/chisqdUnweightedZero)
    print("-----------------------------------------------------")

    if True:
        printInteractDiagnostics(interactIdxs,
                       normlzdInteractDerivs,
                       dnormlzdParamsSolnNonlin,
                       normlzdCurvMatrix, normlzdSensMatrixPoly,
                       paramsNames, numMetrics)

    ##############################################
    #
    #    Create plots
    #
    ##############################################

    # Find best-fit params by use of the Elastic Net algorithm
    defaultBiasesApproxElastic, defaultBiasesApproxElasticNonlin, \
    dnormlzdParamsSolnElastic, paramsSolnElastic = \
        findParamsUsingElastic(normlzdSensMatrixPoly, normlzdWeightedSensMatrixPoly,
                     defaultBiasesCol, normMetricValsCol, metricsWeights,
                     magParamValsRow, defaultParamValsOrigRow,
                     normlzdCurvMatrix,
                     beVerbose)
    #defaultBiasesApproxElasticCheck = ( normlzdWeightedSensMatrixPoly @ dnormlzdParamsSolnElastic ) \
    #                        * np.reciprocal(metricsWeights) * np.abs(normMetricValsCol)
    #print("defaultBiasesApproxElastic = ", defaultBiasesApproxElastic)
    #print("defaultBiasesApproxElasticCheck = ", defaultBiasesApproxElasticCheck)

    if doPiecewise:
        normlzdLinplusSensMatrixPoly = \
            normlzdPiecewiseLinMatrixFnc(dnormlzdParamsSolnNonlin, normlzd_dpMid,
                                         normlzdLeftSensMatrix, normlzdRightSensMatrix)
    else:
        normlzdLinplusSensMatrixPoly = \
            normlzdSemiLinMatrixFnc(dnormlzdParamsSolnNonlin, normlzdSensMatrixPoly,
                                    normlzdCurvMatrix, numMetrics)
    #normlzdWeightedLinplusSensMatrixPoly = np.diag(np.transpose(metricsWeights)[0]) \
    #                                          @ normlzdLinplusSensMatrixPoly
    if doMaximizeRatio:
        print("-----------------Generalized Eigenvalue Problem and Ratio maximizing--------------------------\n")

        # normlzdLinplusSensMatrixPolySST4K = \
        #     normlzdSemiLinMatrixFnc(dnormlzdParamsSolnNonlin, normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K, numMetrics)
        
        # normlzdLinplusSensMatrixPolySST4K = normlzdSensMatrixPolySST4K
        normlzdWeightedSensMatrixPolySST4K  = np.diag(metricsWeights.T[0]) @ normlzdSensMatrixPolySST4K

        eigenvals, eigenvecs = eigh(a=normlzdWeightedSensMatrixPolySST4K.T @ normlzdWeightedSensMatrixPolySST4K ,\
                                     b=normlzdWeightedSensMatrixPoly.T @ normlzdWeightedSensMatrixPoly )
        
        ratios = []
        print(f"ParamsNames: {' '.join(paramsNames)}")
        for idx, eigenval in enumerate(eigenvals):
            eigenvec =  eigenvecs[:,idx]

            print(f"Eigenvalue {idx}: {eigenval}, Eigenvector: {eigenvec}")
            
            ratios.append((eigenvec.T @ normlzdWeightedSensMatrixPolySST4K.T @ normlzdWeightedSensMatrixPolySST4K @ eigenvec) \
                            / (eigenvec.T @ normlzdWeightedSensMatrixPoly.T @ normlzdWeightedSensMatrixPoly @ eigenvec))
        
        print(f"Ratios:",ratios)
        assert np.allclose(ratios, eigenvals), "Ratios do not match eigenvalues!"

        dnormlzdParamsMaxSST4K = eigenvecs[:,-1] 
        print(f"Maximizing parameter perturbations: {dnormlzdParamsMaxSST4K}")   
        print(f"Maximizing parameter values: {calc_dimensional_param_vals(dnormlzdParamsMaxSST4K,magParamValsRow,defaultParamValsOrigRow)}")


        dnormlzdMetricsGenEig = fwdFnc(dnormlzdParamsMaxSST4K.reshape((-1,1)), normlzdWeightedSensMatrixPoly, normlzdCurvMatrix*0, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs)
        
        dnormlzdMetricsGenEigSST4K = fwdFnc(dnormlzdParamsMaxSST4K.reshape((-1,1)), normlzdWeightedSensMatrixPolySST4K, normlzdCurvMatrixSST4K*0, \
                           doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                           numMetrics, normlzdInteractDerivs, interactIdxs)
        
        assert np.allclose(dnormlzdMetricsGenEigSST4K,normlzdWeightedSensMatrixPolySST4K @ dnormlzdParamsMaxSST4K.reshape((-1,1)) ),\
              "Sanity check for fwdFnc with maximizing parameters failed for SST4K data"
        
        assert np.allclose(dnormlzdMetricsGenEig,normlzdWeightedSensMatrixPoly @ dnormlzdParamsMaxSST4K.reshape((-1,1)) ),\
              "Sanity check for fwdFnc with maximizing parameters failed for PD data"
        
        
        
        def calc_SST4K_ratio(eigenvec: np.ndarray, doNonLin: bool):

            normal= fwdFnc(eigenvec.reshape((-1,1)),normlzdSensMatrixPoly, normlzdCurvMatrix * doNonLin, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs)*metricsWeights 
            sst4k   = fwdFnc(eigenvec.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K * doNonLin, \
                           doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                           numMetrics, normlzdInteractDerivs, interactIdxs)*metricsWeights 
            return -1. * (sst4k.T@sst4k)/(normal.T@normal)
        
    
        

        # Define the initial guess for the minimization
        initial_optimization_guess = ((normlzdOrdDparamsMin[0] + normlzdOrdDparamsMax[0])/2)

        # The iterable needs to be converted to a list, so that we can use bounds for both minimizations
        bounds = list(zip(normlzdOrdDparamsMin[0], normlzdOrdDparamsMax[0]))

        # Optimize the ratio using only the sensitivity matrix
        doNonLin = False
        res_lin = minimize(calc_SST4K_ratio, initial_optimization_guess,args=(doNonLin)\
                       , method='COBYLA',bounds=bounds,options={'maxiter':40000,'tol':1e-18})
        
        

        # Optimize the ratio using the sensitivity and curvature matrix
        doNonLin = True
        res_nonlin = minimize(calc_SST4K_ratio, res_lin.x,args=(doNonLin)\
                       , method='COBYLA',bounds=bounds,options={'maxiter':40000,'tol':1e-18})


        # Optimize the ratio using the sensitivity matrix using the basinhopping global optimizer
        doNonLin = False
        res_lin_basin = basinhopping(calc_SST4K_ratio,initial_optimization_guess,niter=10,
                                     minimizer_kwargs={"method":"COBYLA","bounds":bounds,"options":{"maxiter":10000}, "args":(doNonLin),"tol":1e-16})

        # Optimize the ratio using the sensitivity and curvature matrix using the basinhopping global optimizer
        doNonLin = True
        res_nonlin_basin = basinhopping(calc_SST4K_ratio,initial_optimization_guess,niter=10,
                                     minimizer_kwargs={"method":"COBYLA","bounds":bounds,"options":{"maxiter":10000}, "args":(doNonLin),"tol":1e-16})
        
        



        print(f"Result of linear optimization with COBYLA: {res_lin.x}, function value: {-1.*res_lin.fun}")
        print(f"True Parameter: {calc_dimensional_param_vals(res_lin.x,magParamValsRow,defaultParamValsOrigRow).flatten()}")

        print(f"Result of non-linear optimization with COBYLA: {res_nonlin.x}, function value: {-1.*res_nonlin.fun}")
        print(f"True Parameter: {calc_dimensional_param_vals(res_nonlin.x,magParamValsRow,defaultParamValsOrigRow).flatten()}")

        print(f"Result of linear optimization with basinhopping + COBYLA: {res_lin_basin.x}, function value: {-1.*res_lin_basin.fun} ")
        print(f"True Parameter: {calc_dimensional_param_vals(res_lin_basin.x,magParamValsRow,defaultParamValsOrigRow).flatten()}")

        print(f"Result of non-linear optimization with basinhopping + COBYLA: {res_nonlin_basin.x}, function value: {-1.*res_nonlin_basin.fun}")
        print(f"True Parameter: {calc_dimensional_param_vals(res_nonlin_basin.x,magParamValsRow,defaultParamValsOrigRow).flatten()}")
        


        # Check if a larger maximum can be found if we perturb only one parameter
        def check_for_minimum_across_one_axis(dParams,percentages=np.linspace(0.0,2,21),doNonLin=False):
            for paramIdx in range(len(dParams)):
                for percentage in percentages:
                    current_params = np.copy(dParams)
                    current_params[paramIdx] *= percentage

                    assert (newMaximum := -1*calc_SST4K_ratio(current_params,doNonLin)) <= -1 * calc_SST4K_ratio(dParams,doNonLin), \
                    f"Found new maximum with Parameter {paramIdx+1} multiplied with {percentage}. New maximum is: {newMaximum} \n"


        check_for_minimum_across_one_axis(res_lin.x, doNonLin=False)
        check_for_minimum_across_one_axis(res_nonlin.x, doNonLin=True)

                

        # Create values for plotting
        """
        These arrays contain the data for all plots.
         - First index: 0 -> non-linear, 1 -> linear
         - Second Index: 0 -> all parameters, 1-numParams -> Only using the parameter at index-1
         - Third Index: Contains the actual data

         Example: [0,2,:] contains the data for the linear problem using only the second parameter
        """
        MetricsSST4KMaxRatioParams = np.zeros((2,len(res_lin.x)+1,len(dnormlzdMetricsGenEig)))
        MetricsMaxRatioParams = np.zeros((2,len(res_lin.x)+1,len(dnormlzdMetricsGenEig)))



        MetricsSST4KMaxRatioParams[0,0,:] = fwdFnc(res_lin.x.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K * 0, \
                            doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                            numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
        
        MetricsMaxRatioParams[0,0,:]= fwdFnc(res_lin.x.reshape((-1,1)), normlzdSensMatrixPoly, normlzdCurvMatrix * 0, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs).flatten() 
        
        MetricsSST4KMaxRatioParams[1,0,:] = fwdFnc(res_nonlin.x.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K, \
                            doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                            numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
        
        MetricsMaxRatioParams[1,0,:]= fwdFnc(res_nonlin.x.reshape((-1,1)), normlzdSensMatrixPoly, normlzdCurvMatrix, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
        
        MetricsSST4KMaxRatioParams[0,0,:]*=metricsWeights.flatten()
        MetricsSST4KMaxRatioParams[1,0,:]*=metricsWeights.flatten()

        MetricsMaxRatioParams[0,0,:]*=metricsWeights.flatten()
        MetricsMaxRatioParams[1,0,:]*=metricsWeights.flatten()

        

        for paramIdx in range(len(res_lin.x)):
            single_parameter_vector = np.zeros_like(res_nonlin.x)
            single_parameter_vector[paramIdx] = res_lin.x[paramIdx]

            MetricsSST4KMaxRatioParams[0,paramIdx+1,:] = fwdFnc(single_parameter_vector.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K*0, \
                            doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                            numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
            
            MetricsMaxRatioParams[0,paramIdx+1,:]= fwdFnc(single_parameter_vector.reshape((-1,1)),normlzdSensMatrixPoly, normlzdCurvMatrix * 0, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
            
            single_parameter_vector[paramIdx] = res_nonlin.x[paramIdx]
            
            MetricsSST4KMaxRatioParams[1,paramIdx+1,:] = fwdFnc(single_parameter_vector.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K, \
                            doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                            numMetrics, normlzdInteractDerivs, interactIdxs).flatten()
            
            MetricsMaxRatioParams[1,paramIdx+1,:]= fwdFnc(single_parameter_vector.reshape((-1,1)),normlzdSensMatrixPoly, normlzdCurvMatrix, \
                           doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                           numMetrics, normlzdInteractDerivs, interactIdxs).flatten() 
            
            MetricsSST4KMaxRatioParams[0,paramIdx+1,:]*=metricsWeights.flatten()
            MetricsSST4KMaxRatioParams[1,paramIdx+1,:]*=metricsWeights.flatten()

            MetricsMaxRatioParams[0,paramIdx+1,:]*=metricsWeights.flatten()
            MetricsMaxRatioParams[1,paramIdx+1,:]*=metricsWeights.flatten()


        normalization_factor = res_lin.x[0]/dnormlzdParamsMaxSST4K[0]
        assert np.allclose(dnormlzdParamsMaxSST4K * normalization_factor,res_lin.x), "Results from generalized Eigenvalue problem and COBYLA maxmization differ"

        # Sanity checks
        assert np.allclose(np.sum(MetricsMaxRatioParams[1,1:,:],axis=0),MetricsMaxRatioParams[1,0,:]), "fwdFnc with all parameters does not match sum over fwdFnc with one parameter at a time"
        assert np.allclose(np.sum(MetricsSST4KMaxRatioParams[1,1:,:],axis=0),MetricsSST4KMaxRatioParams[1,0,:]), "fwdFnc with all parameters does not match sum over fwdFnc with one parameter at a time"


        print("----------------------------------------------------------------------")
    # Create empty variables for the call of createFigs for the case where doMaximazeRatio = False and doCreatePlots=True.
    else: 
        dnormlzdMetricsGenEig = None
        dnormlzdMetricsGenEigSST4K = None
        normlzdSensMatrixPolySST4K = None
        MetricsSST4KMaxRatioParams = None
        MetricsMaxRatioParams = None

    

    


    
    if doCreatePlots:
        if  createPlotType["SST4KPanelGallery"] and not doMaximizeRatio:
            print("Warning: createPlotType['SST4KPanelGallery'] is True but doCalcGenEig is False. Setting createPlotType['SST4KPanelGallery'] to False.")
            createPlotType["SST4KPanelGallery"] = False
            
        createFigs(numMetricsNoCustom, metricsNames, metricsNamesNoprefix,
                numMetricsToTune,
                varPrefixes, mapVarIdx, boxSize,
                highlightedMetricsToPlot,
                paramsNames, paramsAbbrv, transformedParamsNames, paramsScales,
                metricsWeights, obsMetricValsCol, normMetricValsCol, magParamValsRow,
                defaultBiasesCol, defaultBiasesApproxNonlin, defaultBiasesApproxElastic,
                defaultBiasesApproxNonlinNoCurv, defaultBiasesApproxNonlin2xCurv,
                normlzdDefaultBiasesCol,
                normlzdCurvMatrix, normlzdSensMatrixPoly, normlzdConstMatrix,
                doPiecewise, normlzd_dpMid,
                normlzdLeftSensMatrix, normlzdRightSensMatrix,
                normlzdInteractDerivs, interactIdxs,
                normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                normlzdWeightedSensMatrixPoly,
                dnormlzdParamsSolnNonlin,
                defaultParamValsOrigRow,
                normlzdGlobTunedBiasesCol, normlzdLinplusSensMatrixPoly,
                paramsSolnLin, dnormlzdParamsSolnLin,
                paramsSolnNonlin,
                paramsSolnElastic, dnormlzdParamsSolnElastic,
                sensNcFilenames, sensNcFilenamesExt, defaultNcFilename,
                MetricsMaxRatioParams, MetricsSST4KMaxRatioParams,
                createPlotType,
                reglrCoef, penaltyCoef, numMetrics,
                beVerbose,
                useLongTitle=False, paramBoundsBoot=paramBoundsBoot)

    return paramsSolnNonlin, paramsNames

def normlzdSemiLinMatrixFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix, numMetrics):
    """Calculate semi-linear matrix, sensMatrix + curvMatrix*dp, for use in forward solution"""

    normlzdSemiLinMatrix = \
        normlzdSensMatrix \
        + 0.5 * normlzdCurvMatrix * ( np.ones((numMetrics,1)) @ dnormlzdParams.T ).reshape(numMetrics,len(dnormlzdParams))

    return normlzdSemiLinMatrix

def normlzdPiecewiseLinMatrixFnc(dnormlzdParams, normlzd_dpMid,
                                 normlzdLeftSensMatrix, normlzdRightSensMatrix):
    """Calculate piecewise-linear matrix for use in forward solution"""

    normlzdPiecewiseLinMatrix = np.zeros((normlzdLeftSensMatrix.shape[0],dnormlzdParams.shape[0]))

    for col in np.arange(len(dnormlzdParams)):
        if ( dnormlzdParams[col,0].item() >= ( normlzd_dpMid[col,0].item() ) ):
            normlzdPiecewiseLinMatrix[:,col] = normlzdRightSensMatrix[:,col]
        else:
            normlzdPiecewiseLinMatrix[:, col] = normlzdLeftSensMatrix[:, col]

    return normlzdPiecewiseLinMatrix

def fwdFncNoInteract(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
                     doPiecewise, normlzd_dpMid,
                     normlzdLeftSensMatrix, normlzdRightSensMatrix,
                     numMetrics):
    """
    Calculate the sum of all the forward nonlinear terms except the interaction terms.
    Return a forward sum of terms that is normalized but not weighted
    """

    #print("")

    if doPiecewise:
        normlzdDefaultBiasesApproxNonlin = \
            normlzdPiecewiseLinMatrixFnc(dnormlzdParams, normlzd_dpMid,
                                     normlzdLeftSensMatrix, normlzdRightSensMatrix)  \
                    @ dnormlzdParams
    else:
        normlzdDefaultBiasesApproxNonlin = \
            normlzdSemiLinMatrixFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
                                    numMetrics) \
                    @ dnormlzdParams

    return normlzdDefaultBiasesApproxNonlin

def fwdFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
           doPiecewise, normlzd_dpMid,
           normlzdLeftSensMatrix, normlzdRightSensMatrix,
           numMetrics,
           normlzdInteractDerivs, interactIdxs):
    """Calculate forward nonlinear solution, normalized but not weighted"""

    normlzdDefaultBiasesApproxNonlin = \
        fwdFncNoInteract(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
                         doPiecewise, normlzd_dpMid,
                         normlzdLeftSensMatrix, normlzdRightSensMatrix,
                         numMetrics)

    # dnormlzd_dpj_dpk = ( dp_j * dp_k ) for each interaction term
    dnormlzd_dpj_dpk = calc_dnormlzd_dpj_dpk(dnormlzdParams, interactIdxs)
    interactTerms = normlzdInteractDerivs @ dnormlzd_dpj_dpk

    normlzdDefaultBiasesApproxNonlin += interactTerms

    return normlzdDefaultBiasesApproxNonlin

def calc_dnormlzd_dpj_dpk(dnormlzdParams, interactIdxs):

    '''
    Input:
    dnormlzdParams = A numParams numpy row array of dp
    Output:
    dnormlzd_dpj_dpk = A numInteract numpy col array = [ dp_k * dp_j, ... , ]
    '''
    dnormlzd_dpj_dpk = np.zeros((len(interactIdxs),1))
    for idx, jkTuple in np.ndenumerate(interactIdxs):
        dnormlzd_dpj_dpk[idx,0] = dnormlzdParams[jkTuple[0]][0] * dnormlzdParams[jkTuple[1]][0]

    return dnormlzd_dpj_dpk

def lossFncMetricsKernel(dnormlzdParams, normlzdSensMatrix,
                   normlzdDefaultBiasesCol, metricsWeights,
                   normlzdCurvMatrix,
                   doPiecewise, normlzd_dpMid,
                   normlzdLeftSensMatrix, normlzdRightSensMatrix,
                   numMetrics,
                   normlzdInteractDerivs, interactIdxs):
    """Each regional component of loss function (including squares)"""

    weightedBiasDiffSqdCol = \
        (-normlzdDefaultBiasesCol
                    - fwdFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
                             doPiecewise, normlzd_dpMid,
                             normlzdLeftSensMatrix, normlzdRightSensMatrix,
                             numMetrics,
                             normlzdInteractDerivs, interactIdxs)
         ) * metricsWeights

    return weightedBiasDiffSqdCol

def lossFncMetrics(dnormlzdParams, normlzdSensMatrix,
                   normlzdDefaultBiasesCol, metricsWeights,
                   normlzdCurvMatrix,
                   doPiecewise, normlzd_dpMid,
                   normlzdLeftSensMatrix, normlzdRightSensMatrix,
                   numMetrics,
                   normlzdInteractDerivs, interactIdxs):
    """Each regional component of loss function (including squares)"""

    weightedBiasDiffSqdCol = \
        np.square(
            lossFncMetricsKernel(dnormlzdParams, normlzdSensMatrix,
                   normlzdDefaultBiasesCol, metricsWeights,
                   normlzdCurvMatrix,
                   doPiecewise, normlzd_dpMid,
                   normlzdLeftSensMatrix, normlzdRightSensMatrix,
                   numMetrics,
                   normlzdInteractDerivs, interactIdxs)
        )

    return weightedBiasDiffSqdCol

def lossFnc(dnormlzdParams, normlzdSensMatrix, normlzdDefaultBiasesCol, metricsWeights,
            normlzdCurvMatrix,
            doPiecewise, normlzd_dpMid,
            normlzdLeftSensMatrix, normlzdRightSensMatrix,
            numMetrics,
            normlzdInteractDerivs, interactIdxs):
    """Define objective function (a.k.a. loss function) that is to be minimized."""

    dnormlzdParams = np.atleast_2d(dnormlzdParams).T # convert from 1d row array to 2d column array
    weightedBiasDiffSqdCol = \
        lossFncMetrics(dnormlzdParams, normlzdSensMatrix,
                       normlzdDefaultBiasesCol, metricsWeights,
                       normlzdCurvMatrix,
                       doPiecewise, normlzd_dpMid,
                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
                       numMetrics,
                       normlzdInteractDerivs, interactIdxs)
    #weightedBiasDiffSqdCol = \
    #    np.square( (-normlzdDefaultBiasesCol \
    #     - fwdFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix, numMetrics) \
    #     ) * metricsWeights )
    # This is the chisqd fnc listed in Eqn. (15.2.2) of Numerical Recipes, 1992.
    # It is like MSE (not RMSE), except that it sums the squares rather than averaging them.
    chisqd = np.sum(weightedBiasDiffSqdCol)

    return chisqd

def lossFncWithPenalty(dnormlzdParams, normlzdSensMatrix, normlzdDefaultBiasesCol, metricsWeights,
            normlzdCurvMatrix,
            doPiecewise, normlzd_dpMid,
            normlzdLeftSensMatrix, normlzdRightSensMatrix,
            reglrCoef, penaltyCoef, numMetrics,
            normlzdInteractDerivs, interactIdxs):
    """Define objective function (a.k.a. loss function) that is to be minimized."""

    dnormlzdParams = np.atleast_2d(dnormlzdParams).T # convert from 1d row array to 2d column array
#    weightedBiasDiffSqdCol = \
#        lossFncMetrics(dnormlzdParams, normlzdSensMatrix,
#                       normlzdDefaultBiasesCol, metricsWeights,
#                       normlzdCurvMatrix,
#                       doPiecewise, normlzd_dpMid,
#                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
#                       numMetrics,
#                       normlzdInteractDerivs, interactIdxs)
    weightedBiasDiffCol = \
        lossFncMetricsKernel(dnormlzdParams, normlzdSensMatrix,
                       normlzdDefaultBiasesCol, metricsWeights,
                       normlzdCurvMatrix,
                       doPiecewise, normlzd_dpMid,
                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
                       numMetrics,
                       normlzdInteractDerivs, interactIdxs)
    #weightedBiasDiffSqdCol = \
    #    np.square( (-normlzdDefaultBiasesCol \
    #     - fwdFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix, numMetrics) \
    #     ) * metricsWeights )

    # multiplying weightedBiasDiffCol by penaltyCoef values
    numMetricFields = len(penaltyCoef)
    numBoxesInMap_tmp = int(numMetrics/numMetricFields)
    lastDim = weightedBiasDiffCol.shape[1]
    weightedBiasDiffCol_reshaped = np.reshape(weightedBiasDiffCol,(numMetricFields,numBoxesInMap_tmp,lastDim))
    penalized_weightedBiasDiffCol_reshaped = penaltyCoef[:,None,None] * weightedBiasDiffCol_reshaped
    penalized_weightedBiasDiffCol = np.reshape(penalized_weightedBiasDiffCol_reshaped,(numMetrics,lastDim))

    # This is the chisqd fnc listed in Eqn. (15.2.2) of Numerical Recipes, 1992.
    # It is like MSE (not RMSE), except that it sums the squares rather than averaging them.
    #chisqd = #np.sum(weightedBiasDiffSqdCol) \
    #chisqd = np.sum( np.abs( weightedBiasDiffCol ) ) \
    chisqd = np.sum(np.square(weightedBiasDiffCol)) \
             + reglrCoef * np.linalg.norm(dnormlzdParams, ord=1) \
               + np.square( np.sum ( penalized_weightedBiasDiffCol
#                     (-normlzdDefaultBiasesCol
#                    - fwdFnc(dnormlzdParams, normlzdSensMatrix, normlzdCurvMatrix,
#                             doPiecewise, normlzd_dpMid,
#                             normlzdLeftSensMatrix, normlzdRightSensMatrix,
#                             numMetrics,
#                             normlzdInteractDerivs, interactIdxs)
#                 ) * metricsWeights
                 ) )
    #chisqd = np.sqrt(np.sum(weightedBiasDiffSqdCol)) \
    #         + reglrCoef * np.linalg.norm(dnormlzdParams, ord=1)
    #chisqd = np.linalg.norm( weightedBiasDiffCol, ord=2 )**1  \
    #            + reglrCoef * np.linalg.norm( dnormlzdParams, ord=1 )

    return chisqd

def solveUsingNonlin(metricsNames,
                     metricsWeights, normMetricValsCol, magParamValsRow,
                     defaultParamValsOrigRow,
                     normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                     normlzdSensMatrix, normlzdDefaultBiasesCol,
                     normlzdCurvMatrix,
                     doPiecewise, normlzd_dpMid,
                     normlzdLeftSensMatrix, normlzdRightSensMatrix,
                     normlzdInteractDerivs = np.empty(0), interactIdxs = np.empty(0),
                     reglrCoef = 0.0,
                     penaltyCoef = 0.0,
                     doSensParamBounds = False,
                     beVerbose = False):
    """Find optimal parameter values by minimizing quartic loss function"""

    numMetrics = len(metricsNames)


    # Don't let parameter values go negative
#    lowerBoundsCol =  -defaultParamValsOrigRow[0]/magParamValsRow[0]

    if doSensParamBounds: #don't let quadtune find solutions outside range spanned by default and sensitivity runs 
      lowerBoundsCol = normlzdOrdDparamsMin[0,:]
      upperBoundsCol = normlzdOrdDparamsMax[0,:]
    else: #no bounds on quadtune solutions
      lowerBoundsCol = -999999999*np.ones_like(normlzdOrdDparamsMin[0,:])
      upperBoundsCol =  999999999*np.ones_like(normlzdOrdDparamsMin[0,:])   

    lower_and_upper_bounds = list(zip(lowerBoundsCol.flatten(),upperBoundsCol.flatten()))

    #x0TwoYr = np.array([-0.1400083, -0.404022, 0.2203307, -0.9838958, 0.391993, -0.05910007, 1.198831])
    #x0TwoYr = np.array([0.5805136, -0.1447917, -0.2722521, -0.8183079, 0.3150205, -0.4794127, 0.1104284])
    x0TwoYr = np.array([0.5805136, -0.2722521, -0.8183079, 0.3150205, -0.4794127])
    x0Tuned = np.array([0.6466893, 0.5392086, 0.1818572, 0.0004418074, 0.5])
    # Perform nonlinear optimization
    #normlzdDefaultBiasesCol = defaultBiasesCol/np.abs(normMetricValsCol)
    #dnormlzdParamsSolnNonlin = (minimize(lossFncWithPenalty, x0=x0Tuned,
    dnormlzdParamsSolnNonlin = (minimize(lossFncWithPenalty, x0=np.zeros_like(np.transpose(defaultParamValsOrigRow[0])),
                                         #dnormlzdParamsSolnNonlin = minimize(lossFnc,x0=x0TwoYr, \
                                         #dnormlzdParamsSolnNonlin = minimize(lossFnc,dnormlzdParamsSoln, \
                                         args=(normlzdSensMatrix, normlzdDefaultBiasesCol, metricsWeights,
                                               normlzdCurvMatrix,
                                               doPiecewise, normlzd_dpMid,
                                               normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                               reglrCoef, penaltyCoef, numMetrics,
                                               normlzdInteractDerivs, interactIdxs),
                                         method='Powell', tol=1e-12,
                                         bounds=lower_and_upper_bounds))

    dnormlzdParamsSolnNonlin = np.atleast_2d(dnormlzdParamsSolnNonlin.x).T
    paramsSolnNonlin = calc_dimensional_param_vals(dnormlzdParamsSolnNonlin,magParamValsRow,defaultParamValsOrigRow)
    if beVerbose:
        print("paramsSolnNonlin.T=", paramsSolnNonlin.T)
        print("normlzdSensMatrix@dnPS.x.T=", normlzdSensMatrix @ dnormlzdParamsSolnNonlin)
        print("normlzdDefaultBiasesCol.T=", normlzdDefaultBiasesCol.T)
        print("normlzdSensMatrix=", normlzdSensMatrix)

    normlzdWeightedDefaultBiasesApproxNonlin = \
             fwdFnc(dnormlzdParamsSolnNonlin, normlzdSensMatrix, normlzdCurvMatrix,
                    doPiecewise, normlzd_dpMid,
                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                    numMetrics,
                    normlzdInteractDerivs, interactIdxs) \
             * metricsWeights

    ### WHAT IS THIS METRIC USED FOR???
    scale = 2
    normlzdWeightedDefaultBiasesApproxNonlin2x = \
             fwdFnc(scale * dnormlzdParamsSolnNonlin, normlzdSensMatrix, 1 * normlzdCurvMatrix,
                    doPiecewise, normlzd_dpMid,
                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                    numMetrics,
                    normlzdInteractDerivs, interactIdxs) \
             * metricsWeights

    # Relationship between QuadTune variable names and math symbols:
    # defaultBiasesApproxNonlin = (       forward model soln       - default soln )
    #                           = ( f0 +      fwdFnc               - default soln )
    #                           = ( f0 + df/dp*dp + 0.5d2f/dp2*dp2 -       f0     )
    # residual = (   y_i -                y_hat_i                        )
    #          = (   y_i - ( f0    +   df/dp_i*dp + 0.5d2f/dp2_i*dp2 )   )
    #          =   ( y_i -   f0 )  - ( df/dp_i*dp + 0.5d2f/dp2_i*dp2 )
    #          = -defaultBiasesCol - (   defaultBiasesApproxNonlin   )
    #          = -defaultBiasesCol -              fwdFnc
    #          = normlzdResid * abs(normMetricValsCol)
    #  where f0 = defaultBiasesCol + obsMetricValsCol,
    #        y_i = obsMetricValsCol.
    #  globTunedBiases = forward global model soln - obs
    #                =                    -global_resid
    defaultBiasesApproxNonlin = normlzdWeightedDefaultBiasesApproxNonlin \
                                * np.reciprocal(metricsWeights) * np.abs(normMetricValsCol)

    defaultBiasesApproxNonlin2x = normlzdWeightedDefaultBiasesApproxNonlin2x \
                                * np.reciprocal(metricsWeights) * np.abs(normMetricValsCol)

    # To provide error bars, calculate solution with no nonlinear term and double the nonlinear term
    defaultBiasesApproxNonlinNoCurv = \
             fwdFnc(dnormlzdParamsSolnNonlin, normlzdSensMatrix, 0 * normlzdCurvMatrix,
                    doPiecewise, normlzd_dpMid,
                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                    numMetrics,
                    normlzdInteractDerivs, interactIdxs) \
             * np.abs(normMetricValsCol)

    defaultBiasesApproxNonlin2xCurv = \
             fwdFnc(dnormlzdParamsSolnNonlin, normlzdSensMatrix, 2 * normlzdCurvMatrix,
                    doPiecewise, normlzd_dpMid,
                    normlzdLeftSensMatrix, normlzdRightSensMatrix,
                    numMetrics,
                    normlzdInteractDerivs, interactIdxs) \
             * np.abs(normMetricValsCol)



    dnormlzdParamsSolnLin = minimize(lossFnc, x0=np.zeros_like(np.transpose(defaultParamValsOrigRow[0])),
                                     args=(normlzdSensMatrix, normlzdDefaultBiasesCol, metricsWeights,
                                           0*normlzdCurvMatrix,
                                           doPiecewise, normlzd_dpMid,
                                           normlzdLeftSensMatrix, normlzdRightSensMatrix,
                                           numMetrics,
                                           normlzdInteractDerivs, interactIdxs),
                                     method='Powell')
    dnormlzdParamsSolnLin = np.atleast_2d(dnormlzdParamsSolnLin.x).T

    paramsSolnLin = calc_dimensional_param_vals(dnormlzdParamsSolnLin,magParamValsRow,defaultParamValsOrigRow)


    return (defaultBiasesApproxNonlin,
            dnormlzdParamsSolnNonlin, paramsSolnNonlin,
            dnormlzdParamsSolnLin, paramsSolnLin,
            defaultBiasesApproxNonlin2x,
            defaultBiasesApproxNonlinNoCurv, defaultBiasesApproxNonlin2xCurv
           )


def constructNormlzdSensCurvMatrices(metricsNames, paramsNames, transformedParamsNames,
                               normMetricValsCol, magParamValsRow,
                               sens1NcFilenames, sens2NcFilenames, defaultNcFilename):
    """
    For nonlinear 2nd-order term of Taylor series: 0.5*dp^2*d2m/dp2+...,
    construct a numMetrics x numParams matrix of 2nd-order derivatives, d2m/dp2.
    Each row is a different metric.  Each column is a different parameter.
    The matrix is nondimensionalized by the observed values of metrics
    and maximum values of parameters.
    """


    from set_up_inputs import setupSensArrays
    from set_up_inputs import setupDefaultParamVectors, \
                              setUpDefaultMetricValsCol


    if ( len(paramsNames) != len(sens1NcFilenames)   ):
        print("Number of parameters does not equal number of netcdf files.")
        quit()

    # Number of tunable parameters
    numParams = len(paramsNames)

    # Number of metrics
    numMetrics = len(metricsNames)

    # For use in normalizing metrics matrices
    invrsObsMatrix = np.reciprocal(np.abs(normMetricValsCol)) @ np.ones((1,numParams))

    # Set up a column vector of metric values from the default simulation
    defaultMetricValsCol = \
        setUpDefaultMetricValsCol(metricsNames, defaultNcFilename)
    defaultMetricValsMatrix = defaultMetricValsCol @ np.ones((1,numParams))
    normlzdDefaultMetricValsMatrix = defaultMetricValsMatrix * invrsObsMatrix

    # Based on the default simulation,
    #    set up a column vector of metrics and a row vector of parameter values.
    defaultParamValsRow, defaultParamValsOrigRow = \
        setupDefaultParamVectors(paramsNames, transformedParamsNames,
                                 numParams,
                                 defaultNcFilename)
    normlzdDefaultParamValsRow = defaultParamValsRow * np.reciprocal(magParamValsRow)

    # Based on the numParams sensitivity simulations,
    #    set up a row vector of modified parameter values.
    # Also set up numMetrics x numParams matrix,
    #    each column of which lists the metrics
    #    from one of the sensitivity simulations
    sens1MetricValsMatrix, sens1ParamValsRow, sens1ParamValsOrigRow = \
        setupSensArrays(metricsNames, paramsNames, transformedParamsNames,
                        numMetrics, numParams,
                        sens1NcFilenames,
                        beVerbose=False)
    normlzdSens1ParamValsRow = sens1ParamValsRow * np.reciprocal(magParamValsRow)
    normlzdSens1MetricValsMatrix = sens1MetricValsMatrix * invrsObsMatrix

    # Set up sensitivity-simulation matrices from the extended sensitivity simulation
    sens2MetricValsMatrix, sens2ParamValsRow, sens2ParamValsOrigRow = \
        setupSensArrays(metricsNames, paramsNames, transformedParamsNames,
                        numMetrics, numParams,
                        sens2NcFilenames,
                        beVerbose=False)
    normlzdSens2ParamValsRow = sens2ParamValsRow * np.reciprocal(magParamValsRow)
    normlzdSens2MetricValsMatrix = sens2MetricValsMatrix * invrsObsMatrix

    # Initialize matrix to store second derivatives of metrics w.r.t. parameters
    normlzdCurvMatrixSpline = np.zeros_like(sens1MetricValsMatrix)
    normlzdCurvMatrixPoly = np.zeros_like(sens1MetricValsMatrix)  # 3rd way of calculating derivs
    normlzdSensMatrixPoly = np.zeros_like(sens1MetricValsMatrix)  # Approx of linear sensitivity
    normlzdConstMatrixPoly = np.zeros_like(sens1MetricValsMatrix)  # Approx of linear sensitivity
    normlzdOrdDparamsMin = np.zeros_like(sens1MetricValsMatrix)
    normlzdOrdDparamsMax = np.zeros_like(sens1MetricValsMatrix)

    #pdb.set_trace()

    # Compute quadratic coefficients using a polynomial fit to metric and parameters
    # normlzdOrdMetrics and normlzdOrdParams are length-3 python lists
    for arrayCol in np.arange(numParams):
        for arrayRow in np.arange(numMetrics):

            # Set up three (x,y) points whose 2nd-order derivative we wish to calculate.
            # For the spline code below, the x points need to be ordered from least to greatest.
            if normlzdSens1ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdSens1ParamValsRow[0,arrayCol],
                       normlzdDefaultParamValsRow[0,arrayCol],
                       normlzdSens2ParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdSens1MetricValsMatrix[arrayRow,arrayCol],
                        normlzdDefaultMetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens2MetricValsMatrix[arrayRow,arrayCol] ]
            elif normlzdSens2ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdSens2ParamValsRow[0,arrayCol],
                       normlzdDefaultParamValsRow[0,arrayCol],
                       normlzdSens1ParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdSens2MetricValsMatrix[arrayRow,arrayCol],
                        normlzdDefaultMetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens1MetricValsMatrix[arrayRow,arrayCol] ]
            elif normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdDefaultParamValsRow[0,arrayCol],
                       normlzdSens1ParamValsRow[0,arrayCol],
                       normlzdSens2ParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdDefaultMetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens1MetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens2MetricValsMatrix[arrayRow,arrayCol] ]
            elif normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdDefaultParamValsRow[0,arrayCol],
                       normlzdSens2ParamValsRow[0,arrayCol],
                       normlzdSens1ParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdDefaultMetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens2MetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens1MetricValsMatrix[arrayRow,arrayCol] ]
            elif normlzdSens1ParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdSens1ParamValsRow[0,arrayCol],
                       normlzdSens2ParamValsRow[0,arrayCol],
                       normlzdDefaultParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdSens1MetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens2MetricValsMatrix[arrayRow,arrayCol],
                        normlzdDefaultMetricValsMatrix[arrayRow,arrayCol] ]
            elif normlzdSens2ParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol]:
                normlzdOrdParams = [ normlzdSens2ParamValsRow[0,arrayCol],
                       normlzdSens1ParamValsRow[0,arrayCol],
                       normlzdDefaultParamValsRow[0,arrayCol] ]
                normlzdOrdDparams = normlzdOrdParams - normlzdDefaultParamValsRow[0,arrayCol]
                normlzdOrdMetrics = [ normlzdSens2MetricValsMatrix[arrayRow,arrayCol],
                        normlzdSens1MetricValsMatrix[arrayRow,arrayCol],
                        normlzdDefaultMetricValsMatrix[arrayRow,arrayCol] ]
            else:
                print("Error: Sensitivity parameter values are equal to each other or the default value in constructNormlzdSensCurvMatrices.")
                print( "normlzdSens1ParamValsRow=   ",
                      np.array2string(normlzdSens1ParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
                print( "normlzdSens2ParamValsRow=   ",
                      np.array2string(normlzdSens2ParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
                print( "normlzdDefaultParamValsRow= ",
                      np.array2string(normlzdDefaultParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
                quit()

            normlzdOrdDparamsMin[arrayRow,arrayCol] = np.min(normlzdOrdDparams)
            normlzdOrdDparamsMax[arrayRow,arrayCol]  = np.max(normlzdOrdDparams)

            # Calculate second-order spline based on three given (x,y) points.
            metricValsSpline = UnivariateSpline(normlzdOrdParams,normlzdOrdMetrics,s=0,k=2)
            # Based on spline, find 2nd derivative at arbitrary point (1).
            # I hope that the derivative has the same value at all points,
            #    since it is a parabola.
            normlzdCurvMatrixSpline[arrayRow,arrayCol] = metricValsSpline.derivative(n=2)(1)

            # Check results using a 3rd calculation
            polyCoefs = np.polyfit(normlzdOrdDparams, normlzdOrdMetrics, 2)
            # The curvature matrix is d2m/dp2, not 0.5*d2m/dp2
            #     because of this 2:
            normlzdCurvMatrixPoly[arrayRow][arrayCol] = 2. * polyCoefs[0]
            normlzdSensMatrixPoly[arrayRow,arrayCol] = polyCoefs[1]
            normlzdConstMatrixPoly[arrayRow,arrayCol] = polyCoefs[2]

    if (not np.allclose(normlzdCurvMatrixSpline, normlzdCurvMatrixPoly) ):
        print(f"\nnormlzdCurvMatrixSpline = {normlzdCurvMatrixSpline}")
        print(f"\nnormlzdCurvMatrixPoly = {normlzdCurvMatrixPoly}")
        sys.exit("Error: Spline and polynomial fits of curvature matrices disagree.")

    # Read in information for piecewise linear forward function

    # Value of parameter between the high and low parameter values
    #     (usually, but not always, the default value)
    normlzd_pMidRow = np.zeros_like(defaultParamValsRow)
    # Sensitivity (slope) of metrics to the left of normlzd_pMidRow
    normlzdLeftSensMatrix = np.zeros_like(sens1MetricValsMatrix)
    # Sensitivity (slope) of metrics to the left of normlzd_pMidRow
    normlzdRightSensMatrix = np.zeros_like(sens1MetricValsMatrix)

    # Just as a diagnostic, output low and high parameter values
    normlzd_pLeftRow = np.zeros_like(defaultParamValsRow)
    normlzd_pRightRow = np.zeros_like(defaultParamValsRow)

    for arrayCol in np.arange(numParams):

        # normlzd_pMidRow, LeftSens, and RightSens depend on relative values of parameters
        if normlzdSens1ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdDefaultParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdSens1ParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdSens2ParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdDefaultMetricValsMatrix[:,arrayCol] - normlzdSens1MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdDefaultParamValsRow[0,arrayCol] - normlzdSens1ParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdSens2MetricValsMatrix[:,arrayCol] - normlzdDefaultMetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens2ParamValsRow[0,arrayCol] - normlzdDefaultParamValsRow[0,arrayCol] )

        elif normlzdSens2ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdDefaultParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdSens2ParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdSens1ParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdDefaultMetricValsMatrix[:,arrayCol] - normlzdSens2MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdDefaultParamValsRow[0,arrayCol] - normlzdSens2ParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdSens1MetricValsMatrix[:,arrayCol] - normlzdDefaultMetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens1ParamValsRow[0,arrayCol] - normlzdDefaultParamValsRow[0,arrayCol] )

        elif normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdSens1ParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdDefaultParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdSens2ParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdSens1MetricValsMatrix[:,arrayCol] - normlzdDefaultMetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens1ParamValsRow[0,arrayCol] - normlzdDefaultParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdSens2MetricValsMatrix[:,arrayCol] - normlzdSens1MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens2ParamValsRow[0,arrayCol] - normlzdSens1ParamValsRow[0,arrayCol] )

        elif normlzdDefaultParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdSens2ParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdDefaultParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdSens1ParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdSens2MetricValsMatrix[:,arrayCol] - normlzdDefaultMetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens2ParamValsRow[0,arrayCol] - normlzdDefaultParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdSens1MetricValsMatrix[:,arrayCol] - normlzdSens2MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens1ParamValsRow[0,arrayCol] - normlzdSens2ParamValsRow[0,arrayCol] )

        elif normlzdSens1ParamValsRow[0,arrayCol] < normlzdSens2ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdSens2ParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdSens1ParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdDefaultParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdSens2MetricValsMatrix[:,arrayCol] - normlzdSens1MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens2ParamValsRow[0,arrayCol] - normlzdSens1ParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdDefaultMetricValsMatrix[:,arrayCol] - normlzdSens2MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdDefaultParamValsRow[0,arrayCol] - normlzdSens2ParamValsRow[0,arrayCol] )

        elif normlzdSens2ParamValsRow[0,arrayCol] < normlzdSens1ParamValsRow[0,arrayCol] < normlzdDefaultParamValsRow[0,arrayCol]:

            normlzd_pMidRow[0,arrayCol] = normlzdSens1ParamValsRow[0,arrayCol]
            normlzd_pLeftRow[0, arrayCol] = normlzdSens2ParamValsRow[0, arrayCol]
            normlzd_pRightRow[0, arrayCol] = normlzdDefaultParamValsRow[0, arrayCol]
            normlzdLeftSensMatrix[:,arrayCol] = \
                ( normlzdSens1MetricValsMatrix[:,arrayCol] - normlzdSens2MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdSens1ParamValsRow[0,arrayCol] - normlzdSens2ParamValsRow[0,arrayCol] )
            normlzdRightSensMatrix[:,arrayCol] = \
                ( normlzdDefaultMetricValsMatrix[:,arrayCol] - normlzdSens1MetricValsMatrix[:,arrayCol] ) \
                / ( normlzdDefaultParamValsRow[0,arrayCol] - normlzdSens1ParamValsRow[0,arrayCol] )

        else:
            print("Error: Sensitivity parameter values are equal to each other or the default value in constructNormlzdSensCurvMatrices.")
            print( "normlzdSens1ParamValsRow=   ",
                  np.array2string(normlzdSens1ParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
            print( "normlzdSens2ParamValsRow=   ",
                  np.array2string(normlzdSens2ParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
            print( "normlzdDefaultParamValsRow= ",
                  np.array2string(normlzdDefaultParamValsRow, formatter={'float_kind': lambda x: f"{x:12.8f}"}) )
            quit()

    normlzd_dpMid = normlzd_pMidRow - normlzdDefaultParamValsRow

    # Output normlzd_dpMid as a column vector, like dnormlzdParamsSolnNonlin
    normlzd_dpMid = normlzd_dpMid.T

    return ( normlzdCurvMatrixPoly, normlzdSensMatrixPoly, normlzdConstMatrixPoly,
             normlzdOrdDparamsMin, normlzdOrdDparamsMax,
             normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,
             normlzd_pLeftRow, normlzd_pMidRow, normlzd_pRightRow)


def approxMatrixWithSvd( matrix , sValsRatio, sValsNumToKeep,
                         beVerbose):
    """
    Input: A matrix
    Output: A possibly lower-rank approximation of the matrix,
            with a max ratio of singular values specified by sValsRatio.
    """

    # vh = V^T = transpose of right-singular vector matrix, V.
    #  matrix = u @ np.diag(sVals) @ vh = (u * sVals) @ vh
    u, sVals, vh = np.linalg.svd( matrix, full_matrices=False )

    # Delete the small singular values in order to show just the most important patterns.
    # After this deletion, store inverse singular values in sValsTrunc
    sValsTrunc = np.copy(sVals)
    if (sValsNumToKeep == None):
        for idx, sVal in np.ndenumerate(sVals):
            # If a singular value is much smaller than largest singular value,
            #     then zero it out.
            if np.divide(sVals[0],np.maximum(sVal,np.finfo(float).eps)) > sValsRatio:
                sValsTrunc[idx] = 0.
    else:
        for idx, sVal in np.ndenumerate(sVals):
            if idx+1 > sValsNumToKeep: 
                sValsTrunc[idx] = 0.

    if beVerbose:
        print("\nOriginal singular values =")
        print(sVals)
        print("\nsValsTrunc =")
        print(sValsTrunc)

    matrixApprox = u @ np.diag(sValsTrunc) @ vh
    #matrixApprox = (u * sVals) @ vh

    if beVerbose:
        print("\nstd/mean of (matrixApprox-matrix) = ")
        print(np.std(np.subtract(matrixApprox, matrix))/np.mean(matrix))

    return matrixApprox

#def constructNormlzd2ndOrderTensor(numParams, numMetrics,
#                                   normlzdCurvMatrix, normlzdSensMatrixPoly, normlzdConstMatrix, corrParam):
#    '''Constructs a numParams x numMetrics x numParams tensor of second derivatives, d2metrics/dparams1dparams2.
#    The derivatives are normalized by observed metric values and max param values.'''
#
#    normlzdCurvMatrix, normlzdSensMatrixPoly, normlzdConstMatrix, \
#    normlzdOrdDparamsMin, normlzdOrdDparamsMax = \
#        constructNormlzdSensCurvMatrices(metricsNames, paramsNames, transformedParamsNames, \
#                                   metricsWeights, obsMetricValsCol, normMetricValsCol, magParamValsRow, \
#                                   sensNcFilenames, sensNcFilenamesExt, defaultNcFilename)
#
#    for i in np.arange(numMetrics):
#        # Construct matrix with M(j,k)=0 if curv opposite for i,j; otherwise, M(j,k)=1.
#        curvParamMatrix = np.outer( normlzdCurvMatrix[i,:], np.ones(numParams) )
#        corrParamMatrix = 0.5 * np.abs( np.sign( curvParamMatrix ) + np.sign( curvParamMatrix.T ) )
#        corrParamMatrix = np.fill_diagonal(corrParamMatrix, 1.0)
#        diagSqrtCurvMatrix = np.diag(np.sqrt(normlzdCurvMatrix[i,:]))
#
#    # d4 = np.einsum('j,jik,k->i', params, offdiagtensor, params)
#
#    return ( normlzd2ndOrderTensor  )

def calcNormlzdRadiusCurv(metricsNames, paramsNames, transformedParamsNames, paramsScales,
                          metricsWeights, obsMetricValsCol,
                          sensNcFilenames, sensNcFilenamesExt, defaultNcFilename):
    """
    Calculate radius of curvature of output from 2 sensitivity simulations plus the default
    simulation.
    """

    from set_up_inputs import setupDefaultParamVectors, \
                                           setupSensArrays
    from set_up_inputs import setUpDefaultMetricValsCol

    if ( len(paramsNames) != len(sensNcFilenames)   ):
        print("Number of parameters must equal number of netcdf files.")
        quit()

    # Number of tunable parameters
    numParams = len(paramsNames)

    # Number of metrics
    numMetrics = len(metricsNames)

    # Set up a column vector of metric values from the default simulation
    defaultMetricValsCol = \
        setUpDefaultMetricValsCol(metricsNames, defaultNcFilename)

    # Based on the default simulation,
    #    set up a column vector of metrics and a row vector of parameter values.
    defaultParamValsRow, defaultParamValsOrigRow = \
        setupDefaultParamVectors(paramsNames, transformedParamsNames,
                                 numParams,
                                 defaultNcFilename)

    defaultMetricValsMatrix = defaultMetricValsCol @ np.ones((1,numParams))

    # Based on the numParams sensitivity simulations,
    #    set up a row vector of modified parameter values.
    # Also set up numMetrics x numParams matrix,
    #    each column of which lists the metrics
    #    from one of the sensitivity simulations
    sensMetricValsMatrix, sensParamValsRow, sensParamValsOrigRow = \
        setupSensArrays(metricsNames, paramsNames, transformedParamsNames,
                        numMetrics, numParams,
                        sensNcFilenames,
                        beVerbose=False)

    # Set up sensitivity-simulation matrices from the extended sensitivity simulation
    sensMetricValsMatrixExt, sensParamValsRowExt, sensParamValsOrigRowExt = \
        setupSensArrays(metricsNames, paramsNames, transformedParamsNames,
                        numMetrics, numParams,
                        sensNcFilenamesExt,
                        beVerbose=False)

    normlzd_radius_of_curv = np.full_like(sensMetricValsMatrix, 0.0)

    # Calculate differences in parameter values between default, sensitivity,
    #    and extended sensitivity runs.
    delta_params_def_sens = sensParamValsRow - defaultParamValsRow
    delta_params_def_sensExt = sensParamValsRowExt - defaultParamValsRow
    delta_params_sens_sensExt = sensParamValsRowExt - sensParamValsRow

    # Calculate numMetrics x numParams matrix of metric values.
    delta_metrics_def_sens = sensMetricValsMatrix - defaultMetricValsMatrix
    delta_metrics_def_sensExt = sensMetricValsMatrixExt - defaultMetricValsMatrix
    delta_metrics_sens_sensExt = sensMetricValsMatrixExt - sensMetricValsMatrix
    for col in np.arange(numParams):
        for row in np.arange(numMetrics):
            # Distance between points in simulations = sqrt(dparam**2 + dmetric**2)
            length_def_sens = np.linalg.norm([delta_params_def_sens[0][col],
                                              delta_metrics_def_sens[row][col]])
            length_def_sensExt = np.linalg.norm([delta_params_def_sensExt[0][col],
                                                 delta_metrics_def_sensExt[row][col]])
            length_sens_sensExt = np.linalg.norm([delta_params_sens_sensExt[0][col],
                                                  delta_metrics_sens_sensExt[row][col]])
            semi_perim = 0.5 * ( length_def_sens + length_def_sensExt + length_sens_sensExt )
            # area of triangle formed by points.  Use Heron's formula.
            area = np.sqrt( semi_perim *
                           (semi_perim-length_def_sens) *
                           (semi_perim-length_def_sensExt) *
                           (semi_perim-length_sens_sensExt)
                          )
            if (area == 0.0):
                print( '\nIn calcNormlzdRadiusCurv, area == 0.0 for param ', paramsNames[col],
                        'and metric ', metricsNames[row] )

            # Greatest distance between parameter values in the 3 simulations:
            max_params_width = \
            np.max(np.abs([delta_params_def_sens[0][col],
                        delta_params_def_sensExt[0][col],
                        delta_params_sens_sensExt[0][col]]))
            if (max_params_width == 0.0):
                print( '\nIn calcNormlzdRadiusCurv, max_params_width == 0.0 for param ', paramsNames[col],
                        'and metric ', metricsNames[row] )

            # Calculate Menger curvature from triangle area and distance between points:
            normlzd_radius_of_curv[row][col] = 0.25 * length_def_sens*length_def_sensExt*length_sens_sensExt \
                                                / area / max_params_width

    #pdb.set_trace()
    fig, axs = plt.subplots(numMetrics, numParams, figsize=(24,36))
    for col in np.arange(numParams):
        for row in np.arange(numMetrics):

            paramVals = [defaultParamValsRow[0][col], sensParamValsRow[0][col], sensParamValsRowExt[0][col]]
            metricVals = [defaultMetricValsMatrix[row][col], sensMetricValsMatrix[row][col],
                  sensMetricValsMatrixExt[row][col]]

            axs[row, col].plot( paramVals, metricVals, marker=".", ls="" )
            axs[row, col].plot( paramVals, obsMetricValsCol[row][0] * np.ones((3,1)), color="r" )
            axs[row, col].set_xlabel(paramsNames[col])
            axs[row, col].set_ylabel(metricsNames[row])
            #fig.show()

    plt.show()
    plt.savefig('param_metric_scatter.png')
    #pdb.set_trace()

    return

def findOutliers(normlzdSensMatrix, normlzdWeightedSensMatrix,
                 defaultBiasesCol, normMetricValsCol, magParamValsRow, defaultParamValsOrigRow):
    """Find outliers in bias-senstivity scatterplot based on the RANSAC method."""




    #    ransac = linear_model.RANSACRegressor(max_trials=1000,random_state=0,
#                                          base_estimator=linear_model.LinearRegression(), residual_threshold=None)
#    ransac.fit(normlzdSensMatrix, -defaultBiasesCol / np.abs(normMetricValsCol) )
#    defaultBiasesApproxRansac = ransac.predict(normlzdSensMatrix) * np.abs(normMetricValsCol)
#    dnormlzdParamsSolnRansac = np.transpose( ransac.estimator_.coef_ )
#    ransac = linear_model.RANSACRegressor(max_trials=10000,random_state=0,
#             base_estimator=linear_model.ElasticNet(fit_intercept=False, random_state=0, tol=1e-3,
#                                                    l1_ratio=0.0, alpha=5),
#                                          residual_threshold=None)
#    ransac.fit(normlzdSensMatrix, -defaultBiasesCol / np.abs(normMetricValsCol) )
#    defaultBiasesApproxRansac = np.transpose( np.atleast_2d( ransac.predict(normlzdSensMatrix) ) ) \
#                                * np.abs(normMetricValsCol)
#    dnormlzdParamsSolnRansac = np.transpose( np.atleast_2d( ransac.estimator_.coef_ ) )
#    inlier_mask = ransac.inlier_mask_
#    outlier_mask = np.logical_not(inlier_mask)


    ransac = linear_model.HuberRegressor(fit_intercept=False)
    ransac.fit(normlzdSensMatrix, -defaultBiasesCol / np.abs(normMetricValsCol) )
    defaultBiasesApproxRansac = np.transpose( np.atleast_2d( ransac.predict(normlzdSensMatrix) ) ) \
                                * np.abs(normMetricValsCol)
    dnormlzdParamsSolnRansac = np.transpose( np.atleast_2d( ransac.coef_ ) )
#    defaultBiasesApproxRansac = ransac.predict(normlzdSensMatrix) * np.abs(normMetricValsCol)
#    dnormlzdParamsSolnRansac = np.transpose( ransac.coef_ )

    dparamsSolnRansac = dnormlzdParamsSolnRansac * np.transpose(magParamValsRow)
    paramsSolnRansac = np.transpose(defaultParamValsOrigRow) + dparamsSolnRansac

    outlier_mask = ransac.outliers_
    inlier_mask = np.logical_not(outlier_mask)

    print( "paramsSolnRansac = ", paramsSolnRansac )
    print( "dparamsSolnRansac = ", dparamsSolnRansac )

    # If the solution were perfect, this variable would equal
    #     the normalized, weighted right-hand side.
    normlzdWeightedDefaultBiasesApproxRansac = \
            normlzdWeightedSensMatrix @ dnormlzdParamsSolnRansac

    #pdb.set_trace()

    return (outlier_mask, defaultBiasesApproxRansac, normlzdWeightedDefaultBiasesApproxRansac,
            dnormlzdParamsSolnRansac, paramsSolnRansac)



def findParamsUsingElastic(normlzdSensMatrix, normlzdWeightedSensMatrix,
                 defaultBiasesCol, normMetricValsCol, metricsWeights,
                 magParamValsRow, defaultParamValsOrigRow,
                 normlzdCurvMatrix,
                 beVerbose):
    """Do linear regression with L1 (lasso or elastic net) regularization"""



    #regr = ElasticNet(fit_intercept=True, random_state=0, tol=1e-10, l1_ratio=0.5, alpha=0.01)
    #regr =linear_model.Lasso(fit_intercept=True, random_state=0, tol=1e-10, alpha=0.01) # don't fit intercept!;use line below
    regr = linear_model.Lasso(fit_intercept=False, random_state=0, tol=1e-10, alpha=0.01)
    #regr = linear_model.LassoCV(fit_intercept=True, random_state=0, eps=1e-5, tol=1e-10, cv=metricsWeights.size)
    #print( "alpha_ = ", regr.alpha_ )
    regr.fit(normlzdWeightedSensMatrix, -metricsWeights * defaultBiasesCol / np.abs(normMetricValsCol) )
    #regr.fit(normlzdSensMatrix, -defaultBiasesCol / np.abs(normMetricValsCol) )

    defaultBiasesApproxElastic = np.transpose( np.atleast_2d(regr.predict(normlzdWeightedSensMatrix)) ) \
                          * np.reciprocal(metricsWeights) * np.abs(normMetricValsCol)
    dnormlzdParamsSolnElastic = np.transpose( np.atleast_2d(regr.coef_) )
    dparamsSolnElastic = dnormlzdParamsSolnElastic * np.transpose(magParamValsRow)
    paramsSolnElastic = np.transpose(defaultParamValsOrigRow) + dparamsSolnElastic

    if beVerbose:
        print( "paramsSolnElastic = ", paramsSolnElastic )
        print( "dparamsSolnElastic = ", dparamsSolnElastic )

    #pdb.set_trace()

    # If the solution were perfect, this variable would equal
    #     the normalized, weighted right-hand side.
    defaultBiasesApproxElasticNonlin = \
            normlzdWeightedSensMatrix @ dnormlzdParamsSolnElastic \
                        * np.reciprocal(metricsWeights) * np.abs(normMetricValsCol) \
            + 0.5 * normlzdCurvMatrix @ (dnormlzdParamsSolnElastic**2) * np.abs(normMetricValsCol)

    #pdb.set_trace()

    return (defaultBiasesApproxElastic, defaultBiasesApproxElasticNonlin,
            dnormlzdParamsSolnElastic, paramsSolnElastic)

def calc_dimensional_param_vals(dnormlzdparams,magParamValsRow,defaultParamValsOrigRow):
    """Compute the real parameter values from the normalized parameter biases"""
    return (dnormlzdparams.reshape((-1,1))* np.transpose(magParamValsRow) + np.transpose(defaultParamValsOrigRow))

def check_recovery_of_param_vals(debug_level: int, recovery_test_dparam: np.ndarray, normlzdCurvMatrix, 
                            normlzdSensMatrixPoly, doPiecewise, normlzd_dpMid,
                            normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                            normlzdLeftSensMatrix, normlzdRightSensMatrix,
                            numMetrics, normlzdInteractDerivs, interactIdxs,
                            metricsNames, metricsWeights, normMetricsValsCol,
                            magparamValsRow, defaultParamValsOrigRow, reglrCoef, penaltyCoef, beVerbose):
    """
    Check if quadtune can recover fixed parameters.
    Calls fwdFnc using recovery_test_dparam and tries to recover them using solveUsingNonLin
    """

    normlzdDefaultBiasesApproxNonlin = fwdFnc(recovery_test_dparam, normlzdSensMatrixPoly, normlzdCurvMatrix, \
                          doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                          numMetrics, normlzdInteractDerivs, interactIdxs)
    
    defaultBiasesApproxNonlin, \
    nonlin_recovered_delta_param, paramsSolnNonlin, \
    lin_recovered_delta_param, paramsSolnLin, \
    defaultBiasesApproxNonlin2x, \
    defaultBiasesApproxNonlinNoCurv, defaultBiasesApproxNonlin2xCurv = \
        solveUsingNonlin(metricsNames, metricsWeights, normMetricsValsCol, magparamValsRow, \
                        defaultParamValsOrigRow, normlzdOrdDparamsMin, normlzdOrdDparamsMax, normlzdSensMatrixPoly, -normlzdDefaultBiasesApproxNonlin,\
                            normlzdCurvMatrix, doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix,\
                                normlzdRightSensMatrix, normlzdInteractDerivs, interactIdxs, reglrCoef, penaltyCoef)
    if beVerbose:
        print(f"\nStart test to check if quadtune can recover prescribed parameter deltas . . .\n")
        print(f"Prescribed parameter deltas: {recovery_test_dparam.flatten()}")
        print(f"Recovered parameter deltas: {nonlin_recovered_delta_param.flatten()}")
        print(f"Norm(recovery_test_dparam - nonlin_recovered_delta_param) = {np.linalg.norm(recovery_test_dparam - nonlin_recovered_delta_param)}")            
    
    
    assert np.allclose(recovery_test_dparam, nonlin_recovered_delta_param), "Recovered parameter delta is not close to chosen dparam"

def reweight_regions_by_weightedRegionsDict(boxSize,numMetricsToTune,weightedRegionsDict,
                                            metricsWeights):
    """
    Recalculates and re-normalizes metricsWeights if we are using custom 
    weighting on certain regions (i.e. doWeightRegions = True)
    """

    # will need total number of regions in map
    numBoxesInMap = int( (360/boxSize) * (180/boxSize) )
    # numTuningMetrics is the number of metrics that are actively being tuned for, 
    # because varPrefixes may contain fields that are not being used for tuning
    numTuningMetrics = int( numMetricsToTune / numBoxesInMap )
    for key in weightedRegionsDict:
        latidx_str, lonidx_str = key.split('_')
        latidx = int(latidx_str)
        lonidx = int(lonidx_str)
        weight = weightedRegionsDict[key]
        # the number of longitude indices is 360/boxSize 
        numLonIdxs = int(360/boxSize)
        # find the index of the custom-weighted region
        metricsWeightsIdx = (latidx - 1) * numLonIdxs + lonidx - 1
        # loop over varPrefix range to apply the weight to all metrics
        for metric_idx in range(numTuningMetrics):
            metricsWeights[metricsWeightsIdx + metric_idx * numBoxesInMap,:] = weight \
                                   * metricsWeights[metricsWeightsIdx + metric_idx * numBoxesInMap,:]

    # renormalize the weights by dividing by the sum divided by the number of metrics
    metricsWeights = metricsWeights / (np.sum(metricsWeights)/numTuningMetrics)
    return metricsWeights


if __name__ == '__main__':
    main(sys.argv[1:])
#        sensMatrixDashboard.run_server(debug=True)
