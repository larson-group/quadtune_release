# -*- coding: utf-8 -*-

# Run this app with `python3 quadtune_driver.py` and
# view the plots at http://127.0.0.1:8050/ in your web browser.
# (To open a web browser on a larson-group computer,
# login to malan with `ssh -X` and then type `firefox &`.)

"""
This file contains a set of utility functions that read
the netcdf files containing regional metric values, observations,
and parameter values, and set up various python arrays,
including the default bias column vector and the parameter
value row vector.
"""

import numpy as np
from re import search, match
import netCDF4
import sys



def setUpColAndRowVectors(metricsNames, metricsNorms,
                          obsMetricValsDict,
                          obsOffsetCol, obsGlobalAvgCol, doObsOffset,
                          paramsNames, transformedParamsNames,
                          prescribedParamsNames, prescribedParamValsRow,
                          prescribedTransformedParamsNames,
                          sensNcFilenames, sensNcFilenamesExt,
                          defaultNcFilename
                          ):
    """
    Given netcdf files that contain output from the default and sensitivity global simulations,
    set up column vectors of observations and default-simulation biases.  Also set up
    a row vector of default-simulation parameter values.
    """

    # Set up a column vector of observed metrics
    obsMetricValsCol = setUpObsCol(obsMetricValsDict, metricsNames,
                                   obsOffsetCol, obsGlobalAvgCol, doObsOffset)

    # Set up a normalization vector for metrics, normMetricValsCol.
    # It equals the observed value when metricsNorms has the special value of -999, 
    #     but otherwise it is set manually in metricsNorms itself.
    normMetricValsCol = np.copy(metricsNorms)
    for idx in np.arange(len(metricsNorms)):
        if np.isclose(metricsNorms[idx],-999.0): 
            normMetricValsCol[idx] = obsMetricValsCol[idx]

    # Based on the default simulation,
    #    set up a row vector of parameter values.
    numParams = len(paramsNames)

    defaultParamValsRow, defaultParamValsOrigRow = \
            setupDefaultParamVectors(paramsNames, transformedParamsNames,
                                numParams,
                                defaultNcFilename)


    sensParamValsRow, sensParamValsOrigRow, magParamValsRow = \
        setupSensParamVectors(paramsNames, transformedParamsNames,
                              numParams,
                              defaultParamValsRow,
                              sensNcFilenames)

    sensParamValsRowExt, sensParamValsOrigRowExt, magParamValsRowExt = \
        setupSensParamVectors(paramsNames, transformedParamsNames,
                              numParams,
                              defaultParamValsRow,
                              sensNcFilenamesExt)

# TODO: SHOULD I GENERALIZE magParamVals FOR EXT? OTHERWISE THE RESULT DEPENDS ON WHICH
# PARAMETERS ARE STORED IN WHICH sens FILE.

    # dnormlzdSensParams = Normalized dp values from one of the sensitivity files
    dnormlzdSensParams = ( sensParamValsRow - defaultParamValsRow ) \
                                / magParamValsRow

    # Set up a column vector of metric values from the default simulation
    defaultMetricValsCol = \
        setUpDefaultMetricValsCol(metricsNames, defaultNcFilename)

    #print("defaultMetricValsCol=", defaultMetricValsCol)

    # Store biases in default simulation
    # defaultBiasesCol = + delta_b
    #                  =  default simulation - observations
    defaultBiasesCol = np.subtract(defaultMetricValsCol, obsMetricValsCol)

    # Based on the default simulation,
    #    set up a row vector of prescribed parameter values.
    numPrescribedParams = len(prescribedParamsNames)
    defaultPrescribedParamValsRow, defaultPrescribedParamValsOrigRow = \
            setupDefaultParamVectors(prescribedParamsNames, prescribedTransformedParamsNames,
                                     numPrescribedParams,
                                     defaultNcFilename)

    # Calculate the magnitude of the maximum value of parameters
    #    from the default run (and sensitivity runs as a backup), for later use
    #    in scaling the normalized sensitivity matrix.
    # Initially, set values to the default-simulation values
    magPrescribedParamValsRow = np.abs(defaultPrescribedParamValsRow)
    # Now replace any zero default values with the value from the sensitivity run
    for idx, elem in np.ndenumerate(defaultPrescribedParamValsRow):
        if (np.abs(elem) <= np.finfo(elem.dtype).eps): # if default value is zero
            magPrescribedParamValsRow[0,idx[1]] = np.abs(prescribedParamValsRow[0,idx[1]]) # set to prescribed value
    if np.any( np.isclose(magPrescribedParamValsRow, np.zeros((1,numPrescribedParams))) ):
        print("\nprescribedParamValsRow =")
        print(prescribedParamValsRow)
        print("\nmagPrescribedParamValsRow =")
        print(magPrescribedParamValsRow)
        sys.exit("Error: A prescribed parameter value is zero and so is the prescribed default value.")

    dnormlzdPrescribedParams = ( prescribedParamValsRow - defaultPrescribedParamValsRow ) \
                                / magPrescribedParamValsRow

    #print("prescribedParamValsRow=", prescribedParamValsRow)
    #print("defaultPrescribedParamValsRow=", defaultPrescribedParamValsRow)
    #print("magPrescribedParamValsRow=", magPrescribedParamValsRow)

    dnormlzdPrescribedParams = dnormlzdPrescribedParams.T


    return ( obsMetricValsCol, normMetricValsCol,
             defaultBiasesCol,
             defaultParamValsOrigRow,
             sensParamValsRow, sensParamValsRowExt,
             dnormlzdSensParams,
             magParamValsRow,
             dnormlzdPrescribedParams,
             magPrescribedParamValsRow
           )


def setUp_x_ObsMetricValsDict(varPrefixes, suffix="", obsPathAndFilename=""):
    """
    This is intended for the case in which the metrics are tiles that cover the globe, not custom regions.
    Input: Filename containing observed values of metrics.
    Output: Dictionary of observations.
    """

    # Read netcdf file with metrics and parameters from default simulation
    f_obs = netCDF4.Dataset(obsPathAndFilename, 'r')

    obsMetricValsDict = {}
    obsWeightsDict = {}

    #varPrefixes = ["SWCF"]
    for varName in f_obs.variables:
        #print(varName)
        #         or re.search("^LWCF_[0-9]+_",varName):
        for varPrefix in varPrefixes:
            #if search(f"^{varPrefix}_[0-9]+_", varName):
            if search(f"^{varPrefix}{suffix}",varName):
                #and not "MSWCF" in varName
                varEntry = f_obs[varName]
                varVal = varEntry[:].data[:][0]
                #(See Issue #27) The line above might not work for CAM data and it might need to be replaced with: 
                # varVal =np.mean(varEntry[:].data[:])
                if varPrefix == 'O500':
                    obsMetricValsDict[varName] = varVal[16]
                else:
                    obsMetricValsDict[varName] = varVal
                #print((varName, varVal))
            # Extract observational weights,
            #     which are effectively numpy scalars (0d arrays)
            if search(f"^weights_[0-9]+_[0-9]+_{varPrefix}",varName):
                weightsEntry = f_obs[varName]
                weightsVal = weightsEntry[:].data
                obsWeightsDict[varName] = weightsVal

    f_obs.close()

    #print(obsMetricValsDict)
    #print(len(obsMetricValsDict))

    return (obsMetricValsDict, obsWeightsDict)

def calcObsGlobalAvgCol(varPrefixes,
                        obsMetricValsDict, obsWeightsDict):

    # Set metricsNorms to be a global average
    obsGlobalAvgObsWeights = np.zeros(len(varPrefixes))
    obsGlobalStdObsWeights = np.zeros(len(varPrefixes))
    obsGlobalAvgCol = np.empty(shape=[0, 1])
    obsGlobalStdCol = np.empty(shape=[0, 1])
    obsWeightsCol = np.empty(shape=[0, 1])
    for idx, varPrefix in np.ndenumerate(varPrefixes):
        keysVarPrefix = [key for key in obsWeightsDict.keys() if varPrefix in key]
        # obsWeightsNames = np.array(list(obsWeightsDict.keys()), dtype=str)
        obsWeightsNames = np.array(keysVarPrefix, dtype=str)
        obsWeightsUnnormlzd = setUpObsCol(obsWeightsDict, obsWeightsNames, 0, 0, doObsOffset=False)
        obsWeights = obsWeightsUnnormlzd / np.sum(obsWeightsUnnormlzd)
        # metricsWeights = obsWeights
        # obsWeights = np.vstack([obsWeights] * len(varPrefixes))
        # For the current element of varPrefix, e.g. 'SWCF', select the corresponding regions, e.g. 'SWCF_1_1' . . . 'SWCF_9_18':
        metricsNamesVarPrefix = [key for key in obsMetricValsDict.keys() if varPrefix in key]
        obsMetricValsColVarPrefix = setUpObsCol(obsMetricValsDict, metricsNamesVarPrefix, 0, 0, doObsOffset=False)
        #print("obsMetricValsColVarPrefix = ", obsMetricValsColVarPrefix)
        obsGlobalStdObsWeights[idx] = np.std(obsMetricValsColVarPrefix)
        obsGlobalAvgObsWeights[idx] = np.dot(obsWeights.T, obsMetricValsColVarPrefix)
        # For sea-level pressure, the global avg is too large to serve as a representative normalization
        if varPrefix == 'PSL':
            obsGlobalAvgObsWeights[idx] = 1e-3 * obsGlobalAvgObsWeights[idx]
        #if varPrefix == 'RESTOM':
        #    obsGlobalAvgObsWeights[idx] = 50.0
        print(f"obsGlobalAvgObsWeights for {varPrefix} =", obsGlobalAvgObsWeights[idx])
        obsGlobalAvgCol = np.vstack((obsGlobalAvgCol,
                                     obsGlobalAvgObsWeights[idx] * np.ones((len(obsWeights), 1))
                                     ))
        obsGlobalStdCol = np.vstack((obsGlobalStdCol,
                                     obsGlobalStdObsWeights[idx] * np.ones((len(obsWeights), 1))
                                     ))
        obsWeightsCol = np.vstack((obsWeightsCol,
                                     obsWeights
                                     ))

    return obsGlobalAvgCol, obsGlobalStdCol, obsWeightsCol

def setUp_x_MetricsList(varPrefixes, defPathAndFilename):
    """
    This is intended for the case in which the metrics are tiles that cover the globe, not custom regions.
    Input: Filename of default simulation.
    Output: List of 20x20reg metric values.
    """

    # Read netcdf file with metrics and parameters from default simulation
    f_def = netCDF4.Dataset(defPathAndFilename, 'r')

    metricsNamesWeightsAndNorms = []
    for varPrefix in varPrefixes:
        for varName in f_def.variables:
            #print(varName)
            if match("^numb_[0-9]+_[0-9]+",varName):
                areaWeightEntry = f_def[varName]
                areaWeightVal = areaWeightEntry[:].data[:][0]
                varFullString = varName.replace("numb", varPrefix)
                metricsNamesWeightsAndNorms.append([varFullString,  areaWeightVal, -999])
                #print((SWCF_string, areaWeightVal))

    metricGlobalValsFromFile = np.zeros(len(varPrefixes))
    for index, varPrefix in np.ndenumerate(varPrefixes):
        metricGlobalName = varPrefix + "_GLB"
        metricGlobalValsFromFile[index] = f_def.variables[metricGlobalName][0]

    f_def.close()

    #print(obsMetricValsDict)
    #print(metricsNamesWeightsAndNorms)

    return (metricsNamesWeightsAndNorms, metricGlobalValsFromFile)


def setUpObsCol(obsMetricValsDict, metricsNames,
                obsOffsetCol, obsGlobalAvgCol, doObsOffset):
    """ 
    Input: A python dictionary of observed metrics.
    Output: A column vector of observed metrics
    Input: doObsOffset = True if we want to tune to obs + (offset value)
    Input: obsOffsetCol = user-prescribed value of global mean that we want to match
    Input: obsGlobalAvgCol = global-average value of observed field
    """

    # Number of metrics
    numMetrics = len(metricsNames)

    # Set up column vector of numMetrics elements containing
    # "true" metric values from observations
    obsMetricValsCol = np.zeros((numMetrics,1))
    for idx in np.arange(numMetrics):
        metricName = metricsNames[idx]
        obsMetricValsCol[idx] = obsMetricValsDict[metricName]

    if doObsOffset:
        #obsMetricValsCol = obsMetricValsCol - obsGlobalAvgCol + obsOffsetCol
        obsMetricValsCol = obsMetricValsCol + obsOffsetCol

    return obsMetricValsCol


def setupDefaultParamVectors(paramsNames, transformedParamsNames,
                        numParams,
                        defaultNcFilename):
    """
    Input: Filename containing default-simulation metrics and parameters.
    Output: Row vector of default-simulation parameter values.
    """

    # Read netcdf file with metrics and parameters from default simulation
    f_defaultMetricsParams = netCDF4.Dataset(defaultNcFilename, 'r')

    # Create row vector size numParams containing
    # parameter values from default simulation
    defaultParamValsOrigRow = np.zeros((1, numParams))
    defaultParamValsRow = np.zeros((1, numParams))
    for idx in np.arange(numParams):
        paramName = paramsNames[idx]
        # Assume each metric is stored as length-1 array, rather than scalar.
        #   Hence the "[0]" at the end is needed.
        defaultParamValsOrigRow[0,idx] = f_defaultMetricsParams.variables[paramName][0]
        # Transform [0,1] variable to extend over range [0,infinity]
        if paramName in transformedParamsNames:
            #defaultParamValsRow[0,idx] = -np.log(1-defaultParamValsOrigRow[0,idx])
            defaultParamValsRow[0,idx] = np.log(defaultParamValsOrigRow[0,idx])
        else:
            defaultParamValsRow[0,idx] = defaultParamValsOrigRow[0,idx]

    f_defaultMetricsParams.close()

    return (defaultParamValsRow, defaultParamValsOrigRow)

def setupSensParamVectors(paramsNames, transformedParamsNames,
                          numParams,
                          defaultParamValsRow,
                          sensNcFilenames):

    """
    Input: Filenames of a set of sensitivity simulations for metrics and parameters.
    Output: Row vector of perturbed parameter values from that set of sensitivity simulations.
            Each parameter in row vector is the perturbed one from the appropriate sensitivity simulation.
    """

    # Create row vector size numParams containing
    # parameter values from sensitivity simulations
    sensParamValsOrigRow = np.zeros((1, numParams))
    # This variable contains transformed parameter values,
    #    if transformedParamsNames is non-empty:
    sensParamValsRow = np.zeros((1, numParams))
    for idx in np.arange(numParams):
        paramName = paramsNames[idx]
        # Read netcdf file with changed parameter values from all sensitivity simulations.
        f_sensParams = netCDF4.Dataset(sensNcFilenames[idx], 'r')
        # Assume each metric is stored as length-1 array, rather than scalar.
        #   Hence the "[0]" at the end is needed.
        sensParamValsOrigRow[0,idx] = f_sensParams.variables[paramName][0]
        # Transform [0,1] variable to extend over range [0,infinity]
        if paramName in transformedParamsNames:
            #sensParamValsRow[0,idx] = -np.log(1-sensParamValsRow[0,idx])
            sensParamValsRow[0,idx] = np.log(sensParamValsOrigRow[0,idx])
        else:
            sensParamValsRow[0,idx] = sensParamValsOrigRow[0,idx]
        f_sensParams.close()


    # Calculate the magnitude of the maximum value of parameters
    #    from the default run (and sensitivity runs as a backup), for later use
    #    in scaling the normalized sensitivity matrix.
    # Initially, set values to the default-simulation values
    magParamValsRow = np.abs(defaultParamValsRow)
    # Now replace any zero default values with the value from the sensitivity run
    for idx, elem in np.ndenumerate(defaultParamValsRow):
        if (np.abs(elem) <= np.finfo(elem.dtype).eps): # if default value is zero
            magParamValsRow[0,idx[1]] = np.abs(sensParamValsRow[0,idx[1]]) # set to sensitivity value
    if np.any( np.isclose(magParamValsRow, np.zeros((1,numParams))) ):
        print("\nsensParamValsRow =")
        print(sensParamValsRow)
        print("\nmagParamValsRow =")
        print(magParamValsRow)
        sys.exit("Error: A parameter value from both default and sensitivity simulation is zero.")

    return ( sensParamValsRow, sensParamValsOrigRow, magParamValsRow )

def setupSensArrays(metricsNames, paramsNames, transformedParamsNames,
                    numMetrics, numParams,
                    sensNcFilenames,
                    beVerbose):
    """
    Input: List of filenames, one per each sensitivity simulation.
    Output: Row vector of modified parameter values from sensitivity simulations.
            Sensitivity matrix of regional metrics to parameter values, where each column corresponds to
                a single sensitivity simulation.
    """

    # Create row vector size numParams containing
    # parameter values from sensitivity simulations
    sensParamValsOrigRow = np.zeros((1, numParams))
    # This variable contains transformed parameter values,
    #    if transformedParamsNames is non-empty:
    sensParamValsRow = np.zeros((1, numParams))
    for idx in np.arange(numParams):
        paramName = paramsNames[idx]
        # Read netcdf file with changed parameter values from all sensitivity simulations.
        f_sensParams = netCDF4.Dataset(sensNcFilenames[idx], 'r')
        # Assume each metric is stored as length-1 array, rather than scalar.
        #   Hence the "[0]" at the end is needed.
        sensParamValsOrigRow[0, idx] = f_sensParams.variables[paramName][0]
        # Transform [0,1] variable to extend over range [0,infinity]
        if paramName in transformedParamsNames:
            # sensParamValsRow[0,idx] = -np.log(1-sensParamValsRow[0,idx])
            sensParamValsRow[0, idx] = np.log(sensParamValsOrigRow[0, idx])
        else:
            sensParamValsRow[0, idx] = sensParamValsOrigRow[0, idx]
        f_sensParams.close()

    # sensParamValsRow = np.array([[2., 4.]])

    if beVerbose:
        print("\nsensParamValsOrigRow =")
        print(sensParamValsOrigRow)
        print("\nsensParamValsRow =")
        print(sensParamValsRow)

    # numMetrics x numParams matrix of metric values
    # from sensitivity simulations
    sensMetricValsMatrix = np.zeros((numMetrics, numParams))
    for col in np.arange(numParams):
        f_sens = netCDF4.Dataset(sensNcFilenames[col], 'r')
        for row in np.arange(numMetrics):
            metricName = metricsNames[row]
            sensMetricValsMatrix[row, col] = f_sens.variables[metricName][0]
            # if (metricName[0:3] == 'PSL'):  # subtract 9e4 from sea-level pressure for better scaling
            #    sensMetricValsMatrix[row,col] = f_sens.variables[metricName][0] - 9.e4
            #    print("metricName[0:3]=", metricName[0:3])
            #    print("sensMetricValsMatrix=", sensMetricValsMatrix[row][col])
            # else:
            #    sensMetricValsMatrix[row,col] = f_sens.variables[metricName][0]
        f_sens.close()

    # sensMetricValsMatrix = np.array([[1., 2.], [3., 4.]])

    if beVerbose:
        print("\nsensMetricValsMatrix =")
        print(sensMetricValsMatrix)

    return (sensMetricValsMatrix, sensParamValsRow, sensParamValsOrigRow)


def setUpDefaultMetricValsCol(metricsNames, defaultNcFilename):
    """
    Input: Filename containing default-simulation metrics.
    Output: Column vector of default-sim metrics.
    """

    # Number of metrics
    numMetrics = len(metricsNames)

    # Read netcdf file with metrics and parameters from default simulation
    f_defaultMetricsParams = netCDF4.Dataset(defaultNcFilename, 'r')

    # Set up column vector of numMetrics elements containing
    # metric values from default simulation
    defaultMetricValsCol = np.zeros((numMetrics,1))
    for idx in np.arange(numMetrics):
        metricName = metricsNames[idx]
        # Assume each metric is stored as length-1 array, rather than scalar.
        #   Hence the "[0]" at the end is needed.
        defaultMetricValsCol[idx] = f_defaultMetricsParams.variables[metricName][0]

    f_defaultMetricsParams.close()

    return defaultMetricValsCol

def calcInteractDerivs(interactIdxs,
                       dnormlzdParamsInteract,
                       normlzdInteractBiasesCols,
                       normlzdCurvMatrix, normlzdSensMatrix,
                       doPiecewise, normlzd_dpMid,
                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
                       numMetrics):

    from quadtune_driver import fwdFncNoInteract

    normlzdInteractDerivs = np.zeros(( numMetrics, len(interactIdxs) ))
    for idxTerm, idxTuple in np.ndenumerate(interactIdxs):

        # Set up a column vector of metric values from the default simulation
        normlzdInteractDerivs[:,idxTerm] = \
            (
              normlzdInteractBiasesCols[:,idxTerm]
            - fwdFncNoInteract(np.atleast_2d( dnormlzdParamsInteract[idxTerm][0] ),
                              normlzdSensMatrix[:,idxTuple[0]].reshape(-1, 1),
                              normlzdCurvMatrix[:,idxTuple[0]].reshape(-1, 1),
                              doPiecewise, np.atleast_2d(normlzd_dpMid[idxTuple[0],:]),
                              normlzdLeftSensMatrix[:,idxTuple[0]].reshape(-1, 1),
                              normlzdRightSensMatrix[:,idxTuple[0]].reshape(-1, 1),
                              numMetrics).reshape(-1,1)
            - fwdFncNoInteract(np.atleast_2d( dnormlzdParamsInteract[idxTerm][1] ),
                              normlzdSensMatrix[:,idxTuple[1]].reshape(-1, 1),
                              normlzdCurvMatrix[:,idxTuple[1]].reshape(-1, 1),
                              doPiecewise, np.atleast_2d(normlzd_dpMid[idxTuple[1],:]),
                              normlzdLeftSensMatrix[:,idxTuple[1]].reshape(-1, 1),
                              normlzdRightSensMatrix[:,idxTuple[1]].reshape(-1, 1),
                              numMetrics).reshape(-1,1)
            ) / ( dnormlzdParamsInteract[idxTerm][0] * dnormlzdParamsInteract[idxTerm][1] )

    return normlzdInteractDerivs

def createInteractIdxs(interactParamsNamesAndFilenames, paramsNames):

    # Define a numpy structured dtype to hold jk indices of each interaction term
    interactIdxType = np.dtype([('jIdx', np.intp), ('kIdx', np.intp)])

    interactIdxs = np.zeros(len(interactParamsNamesAndFilenames), dtype=interactIdxType)
    for idx, nameTuple in np.ndenumerate(interactParamsNamesAndFilenames):
        interactIdxs[idx] = ( np.where(paramsNames == nameTuple[0])[0],
                              np.where(paramsNames == nameTuple[1])[0]
                              )

    return interactIdxs

def readDnormlzdParamsInteract(interactParamsNamesAndFilenames, interactIdxs,
                               defaultParamValsRow, magParamValsRow,
                               paramsNames, transformedParamsNames, numParams):
    '''
    :return: dnormlzdParamsInteract = [ (dp_j, dp_k) , ( , ), . . . ] values from interact runs.
    '''

    # Define a numpy structured dtype to hold jk indices of each interaction term
    paramValsInteractType = np.dtype([('jParam', np.float64), ('kParam', np.float64)])

    dnormlzdParamsInteract = np.zeros( len(interactParamsNamesAndFilenames),
                                      dtype=paramValsInteractType )

    # Set up (numInteractRuns x numParams) matrix of param values in interaction runs
    for idx, nameTuple in np.ndenumerate(interactParamsNamesAndFilenames):
        interactParamValsRow, interactParamValsOrigRow = \
            setupDefaultParamVectors(paramsNames, transformedParamsNames,
                                     numParams,
                                     nameTuple[2])

        normlzdParamsInteract = \
            (
              interactParamValsRow[ 0, interactIdxs[idx][0] ]
                 / magParamValsRow[ 0, interactIdxs[idx][0] ],
              interactParamValsRow[ 0, interactIdxs[idx][1] ]
                 / magParamValsRow[ 0, interactIdxs[idx][1] ]
              )

        normlzdDefaultParamsInteract = \
            (
              defaultParamValsRow[ 0, interactIdxs[idx][0] ]
                / magParamValsRow[ 0, interactIdxs[idx][0] ],
              defaultParamValsRow[ 0, interactIdxs[idx][1] ]
                / magParamValsRow[ 0, interactIdxs[idx][1] ]
              )

        if np.any(np.isclose( normlzdParamsInteract , normlzdDefaultParamsInteract )):
            print("\nnormlzdParamsInteract =")
            print(normlzdParamsInteract)
            print("\nnormlzdDefaultParamsInteract =")
            print(normlzdDefaultParamsInteract)
            sys.exit("Error: An interacting parameter equals the default value.")

        dnormlzdParamsInteract[idx] = \
            tuple(
                np.subtract(normlzdParamsInteract , normlzdDefaultParamsInteract)
            )

    return dnormlzdParamsInteract

def calcNormlzdInteractBiasesCols(defaultMetricValsCol, normMetricValsCol,
                                  metricsNames,
                                  interactParamsNamesAndFilenames):
    '''
    :return = normlzdInteractBiasesCols = interact - default
    '''

    normlzdInteractBiasesCols = np.zeros((defaultMetricValsCol.shape[0],
                                             len(interactParamsNamesAndFilenames)))
    for idx, nameTuple in np.ndenumerate(interactParamsNamesAndFilenames):

        # Set up a column vector of metric values from the default simulation
        interactMetricValsCol = \
            setUpDefaultMetricValsCol(metricsNames, nameTuple[2])

        # Store biases in default simulation
        # defaultBiasesCol = + delta_b
        #                  =  default simulation - observations
        # shape = numMetrics x (number of interaction simulations)
        normlzdInteractBiasesCols[:,idx] = np.subtract(interactMetricValsCol, defaultMetricValsCol) \
                                     / np.abs(normMetricValsCol)

    return normlzdInteractBiasesCols

def checkInteractParamVals(
        interactIdxs, interactParamsNamesAndFilenames,
        sensParamValsRow, sensParamValsRowExt,
        defaultParamValsOrigRow,
        paramsNames, transformedParamsNames,
        numParams
):

    for interactIdx, nameTuple in np.ndenumerate(interactParamsNamesAndFilenames):

        interactParamValsRow, interactParamValsOrigRow = \
            setupDefaultParamVectors(paramsNames, transformedParamsNames,
                                     numParams,
                                     nameTuple[2])

        #print(f"\ninteractParamValsRow[0,:] = {interactParamValsRow[0, :]}")
        #print(f"\nsensParamValsRow[0,:] = {sensParamValsRow[0, :]}")
        #print(f"\nsensParamValsRowExt[0,:] = {sensParamValsRowExt[0, :]}")
        #print(f"\ndefaultParamValsOrigRow[0,:] = {defaultParamValsOrigRow[0, :]}")

        for paramIdx, paramName in np.ndenumerate(paramsNames):

            if paramIdx in interactIdxs[interactIdx]:

                if ( not np.isclose (interactParamValsRow[0,paramIdx], sensParamValsRow[0,paramIdx] ) and
                     not np.isclose( interactParamValsRow[0,paramIdx], sensParamValsRowExt[0,paramIdx] ) ):

                    print(f"\ninteractParamValsRow[0,:] = {interactParamValsRow[0,:]}")
                    print(f"\nsensParamValsRow[0,:] = {sensParamValsRow[0,:]}")
                    print(f"\nsensParamValsRowExt[0,:] = {sensParamValsRowExt[0, :]}")

                    sys.exit("Error: Interaction parameter value does not equal either of the sensitivity values.")

            else:

                if ( not np.isclose (interactParamValsRow[0,paramIdx], defaultParamValsOrigRow[0,paramIdx] ) ):

                    print(f"\ninteractParamValsRow[0,:] = {interactParamValsRow[0,:]}")
                    print(f"\ndefaultParamValsOrigRow[0,:] = {defaultParamValsOrigRow[0,:]}")

                    sys.exit("Error: Interaction parameter value does not equal default value.")


    return

def checkInteractDerivs(normlzdInteractBiasesCols,
                        dnormlzdParamsInteract,
                        numParams,
                        normlzdSensMatrix, normlzdCurvMatrix,
                        doPiecewise, normlzd_dpMid,
                        normlzdLeftSensMatrix, normlzdRightSensMatrix,
                        numMetrics,
                        normlzdInteractDerivs, interactIdxs):
    '''
    This function checks whether normlzdInteractDerivs has been calculated
    correctly by doing a forward calculation with the parameter values in the
    sensitivity files.
    '''

    from quadtune_driver import fwdFnc


    for idxTerm, idxTuple in np.ndenumerate(interactIdxs):
        dnormlzdTwoParams = np.zeros((numParams, 1))

        dnormlzdTwoParams[ idxTuple[0], 0 ] = dnormlzdParamsInteract[idxTerm][0]
        dnormlzdTwoParams[ idxTuple[1], 0 ] = dnormlzdParamsInteract[idxTerm][1]

        fwdFncInteractCol = \
            fwdFnc(dnormlzdTwoParams, normlzdSensMatrix, normlzdCurvMatrix,
                   doPiecewise, normlzd_dpMid,
                   normlzdLeftSensMatrix, normlzdRightSensMatrix,
                   numMetrics,
                   normlzdInteractDerivs, interactIdxs)

        if not np.allclose(normlzdInteractBiasesCols[:,idxTerm], fwdFncInteractCol):
            print(f"\nnormlzdInteractBiasesCols[:,{idxTerm}] =")
            print(normlzdInteractBiasesCols[:,idxTerm].T)
            print("\nfwdFncInteractCol =")
            print(fwdFncInteractCol.T)
            print(f"\nidxTerm = {idxTerm}")
            sys.exit("Error: normlzdInteractDerivs is not consistent with interactBiases.")

    return

def printInteractDiagnostics(interactIdxs,
                             normlzdInteractDerivs,
                             dnormlzdParamsSolnNonlin,
                             normlzdCurvMatrix, normlzdSensMatrix,
                             paramsNames, numMetrics):

    from quadtune_driver import calc_dnormlzd_dpj_dpk

    dnormlzd_dpj_dpk = calc_dnormlzd_dpj_dpk(dnormlzdParamsSolnNonlin, interactIdxs)

    #print( "dnormlzd_dpj_dpk = ", dnormlzd_dpj_dpk )

    interactTermsMatrix = normlzdInteractDerivs * (np.ones((numMetrics, 1)) * dnormlzd_dpj_dpk.T)

    #print( "paramsNames = ", paramsNames )
    #print( "interactIdxs = ", interactIdxs )

    interactTable = np.zeros(( numMetrics, 6, len(interactIdxs) ))
    for idxTerm, jkTuple in np.ndenumerate(interactIdxs):

        normlzdInteractDerivTerm = normlzdInteractDerivs[:, idxTerm] * dnormlzd_dpj_dpk[idxTerm, 0]

        normlzdCurvJTerm = \
            0.5 * normlzdCurvMatrix[ : , [jkTuple[0]] ] * np.square( dnormlzdParamsSolnNonlin[jkTuple[0]] )
        normlzdCurvKTerm = \
            0.5 * normlzdCurvMatrix[ : , [jkTuple[1]] ] * np.square( dnormlzdParamsSolnNonlin[jkTuple[1]] )

        normlzdSensJTerm = \
            normlzdSensMatrix[ : , [jkTuple[0]] ] * dnormlzdParamsSolnNonlin[ [jkTuple[0]] ]
        normlzdSensKTerm = \
            normlzdSensMatrix[ : , [jkTuple[1]] ] * dnormlzdParamsSolnNonlin[ [jkTuple[1]] ]

        sumNoInteractTerm = normlzdCurvJTerm + normlzdCurvKTerm + normlzdSensJTerm + normlzdSensKTerm
        normlzdInteractDerivRatio = normlzdInteractDerivTerm / sumNoInteractTerm

        intermediateResult = np.hstack((normlzdInteractDerivRatio, normlzdInteractDerivTerm,
                   normlzdSensJTerm, normlzdSensKTerm, normlzdCurvJTerm, normlzdCurvKTerm))

        interactTable[:, :, idxTerm] = intermediateResult[:, :, np.newaxis]


        #interactDerivRatiosCols[:,idxTerm] = \
        #    normlzdInteractDerivs[:, idxTerm] / np.sqrt(np.abs (normlzdCurvJ * normlzdCurvK))


    #interactDerivRatios = np.std( interactDerivRatiosCols, axis=0 )

    #print( "interactDerivRatios = ", interactDerivRatios )

    #normlzdCurvTermsMatrix = np.zeros((numMetrics, len(paramsNames)))
    #normlzdSensTermsMatrix = np.zeros((numMetrics, len(paramsNames)))
    #for paramIdx in np.arange(len(paramsNames)):

    #    normlzdCurvTermsMatrix[:,paramIdx] = \
    #        0.5 * normlzdCurvMatrix[:,paramIdx] * np.square( dnormlzdParamsSolnNonlin[paramIdx] )

    #    normlzdSensTermsMatrix[:,paramIdx] = \
    #        normlzdSensMatrix[:,paramIdx] * dnormlzdParamsSolnNonlin[paramIdx]

    return

def checkPiecewiseLeftRightPoints( dLeftRightParams,
                        normlzdSensMatrixPolySvd, normlzdCurvMatrix,
                        normlzd_dpMid,
                        normlzdLeftSensMatrix, normlzdRightSensMatrix,
                        numMetrics):
    '''
    This function checks whether normlzdPiecewiseLinMatrixFnc and normlzdSemiLinMatrixFnc
    yield the same answer at the lo and hi perturbations, normlzd_pLeftRow and normlzd_pRightRow.
    '''

    from quadtune_driver import normlzdPiecewiseLinMatrixFnc, normlzdSemiLinMatrixFnc


    piecewiseMatrix = \
        normlzdPiecewiseLinMatrixFnc(dLeftRightParams, normlzd_dpMid,
                                     normlzdLeftSensMatrix, normlzdRightSensMatrix)

    semiMatrix = \
        normlzdSemiLinMatrixFnc(dLeftRightParams, normlzdSensMatrixPolySvd, normlzdCurvMatrix,
                                numMetrics)

    diffMatrix = piecewiseMatrix - semiMatrix

    piecewiseVector = piecewiseMatrix @ dLeftRightParams

    semiVector = semiMatrix @ dLeftRightParams

    diffVector = piecewiseVector - semiVector

    if not np.allclose(piecewiseVector, semiVector):
        print("\npiecewiseVector =")
        print(piecewiseVector.T)
        print("\nsemiVector =")
        print(semiVector.T)
        sys.exit("Error: Piecewise emulator is inconsistent with quadratic emulator at lo/hi param values.")

    return