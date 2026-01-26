from typing import Any
import xarray as xr
import numpy as np


def process_PPE_params_file(PPE_params_filename: str, paramsNamesAndScales: np.ndarray, allparamsNamesInFile: list[str]):
    """
    Process the PPE parameter file and extract:
        - default parameters
        - all PPE parameters
        - parameter Names and scales for plotting

    :param PPE_params_filename: Path to a file containing parameters for a PPE
    :param paramsNamesAndScales: 2D-Array containing the parameter names with their scales
    :param allparamsNamesInFile: List containing all parameter names in the file
    """

    params_dataset = xr.open_dataset(PPE_params_filename,engine="netcdf4")

    paramsNames, paramsScales = get_PPE_paramNames(paramsNamesAndScales)

    paramsIndices = np.where(np.isin(allparamsNamesInFile, paramsNames))[0]



    defaultParamValsOrigRow = params_dataset.where(params_dataset.ens_idx.str.match('ctrl'), drop= True).params.values[0,paramsIndices]




    #PPE parameters without ctrl, validate, L00X and R00X ensemble members stored in numb_ensemble_members x num_params array
    PPE_parameters = remove_ctrl_and_validate_runs_from_PPE(params_dataset).params.values[:,paramsIndices]



    magParamValsRow = np.abs(defaultParamValsOrigRow)

    minParams = np.min(PPE_parameters,axis=0)

    maxParams = np.max(PPE_parameters,axis=0)

    for idx, val in enumerate(magParamValsRow):
        if val <= np.finfo(val.dtype).eps:
            magParamValsRow[idx] = maxParams[idx]

    normlzdOrdDparamsMinFlat = minParams/magParamValsRow - np.ones_like(defaultParamValsOrigRow)

    normlzdOrdDparamsMaxFlat = maxParams/magParamValsRow - np.ones_like(defaultParamValsOrigRow)


    magParamValsRow = magParamValsRow.reshape((1,-1))

    defaultParamValsOrigRow = defaultParamValsOrigRow.reshape((1,-1))

    return defaultParamValsOrigRow, PPE_parameters, paramsNames, paramsIndices, paramsScales, magParamValsRow, normlzdOrdDparamsMinFlat, normlzdOrdDparamsMaxFlat

    


def process_ppe_metrics_file(PPE_metrics_filename: str, varPrefixes:list[str], boxSize:int):
    """
    Process the PPE metrics file and extract:
        - default metrics values
        - all PPE metrics values
        - metrics Names
        - normalized metrics weights for the default case
        - obs metrics values
        - normalized obs metrics weights
    
    :param PPE_metrics_filename: Path to a file containing metrics for a PPE
    :param varPrefixes: List containing the prefixes of the names of the used metrics
    :param boxSize: Size of the boxes on the map

    """

    metrics_dataset = xr.open_dataset(PPE_metrics_filename,engine="netcdf4")

    metricsNames = get_metrics_names(varPrefixes, boxSize)

    weightsNames = get_weights_names(boxSize)

    default_metrics_dataset = metrics_dataset.sel(ens_idx="ctrl")

    PPE_metrics = remove_ctrl_and_validate_runs_from_PPE(metrics_dataset).isel(time=0,product=0)[metricsNames].to_array().values

    metricsWeights = default_metrics_dataset.isel(time=0,product=0)[weightsNames].to_array(dim="metricsNames").to_numpy()

    default_metrics = default_metrics_dataset.isel(time=0,product=0)[metricsNames]

    defaultMetricValsCol = default_metrics[metricsNames].to_array().values.reshape((-1,1)) 

    normlzdMetricsWeights = (metricsWeights/np.sum(metricsWeights)).reshape((-1,1))
    

    obsMetricValsCol = metrics_dataset[metricsNames].to_array(dim="metricsName").isel(time=0,product=1,ens_idx=0).to_numpy().reshape((-1,1))

    obsWeightsCol = metrics_dataset["weights"].values

    # check if only a single obsWeight was given. If so, use metricWeights instead
    if len([obsWeightsCol])  == 1:
        normlzdObsWeights = normlzdMetricsWeights
    else:
        normlzdObsWeights = obsWeightsCol/np.sum(obsWeightsCol)

    
    return defaultMetricValsCol, PPE_metrics, metricsNames, normlzdMetricsWeights, obsMetricValsCol, normlzdObsWeights




    



def construct_sensitivity_curvature_matrices_from_PPE_data(PPE_metrics:np.ndarray, default_metrics:np.ndarray, PPE_params:np.ndarray, default_params:np.ndarray, normMetricValsCol:np.ndarray, magParamValsRow:np.ndarray):
    """
    Create normalized Sensitivity and Curvature matrices from PPE data by repeatingly solving the system

    | dp_1(PPE1)  0.5dp_1^2(PPE1) dp_2(PPE1) ... |    |  ∂m_i/∂p_1  |    |dm_i(PPE1)|
    | dp_1(PPE2)  0.5dp_1^2(PPE2) dp_2(PPE2) ... |    |∂m_i^2/∂p_1^2|    |dm_i(PPE2)|
    | dp_1(PPE3)  0.5dp_1^2(PPE3) dp_2(PPE3) ... |    |  ∂m_i/∂p_2  | =  |dm_i(PPE3)|
    |                ...                         |    |    ...      |    |   ...    |

    for different metrics m_i.

    Here dm_i(PPEj) = m_i(PPEj) - m_i(default).
    
    Parameters and metrics are getting normalized for the solving process.

    :param PPE_metrics: 2D-Array containing the metrics values for each PPE run (num_metrics x num_PPEs)
    :param default_metrics: 2D-Array containing the default metrics values (num_metrics x 1)
    :param PPE_params: 2D-Array containing the parameters values for each PPE run (num_PPEs x num_params)
    :param default_params: 2D-Array containing the default parameters values (1 x num_params)
    :param normMetricValsCol: 2D-Array containing the normalized metric values (num_metrics x 1)
    :param magParamValsRow: 2D-Array containing the magnitude of parameter values (1 x num_params)



    """

    '''Create the static leftside matrix of the System'''
    lin_system = np.zeros((PPE_params.shape[0],len(default_params.flatten())*2))

    for i in range(lin_system.shape[0]):

        lin_row_params = PPE_params[i,:]


        dnormlzdlin_row_params = (lin_row_params - default_params) / np.abs(magParamValsRow)

        dnormlzdquad_row_params = 0.5*dnormlzdlin_row_params**2


        full_row = np.dstack((dnormlzdlin_row_params,dnormlzdquad_row_params)).flatten()
        lin_system[i,:] = full_row

    '''Solve the linear system for each metric, to create the sensitivity and curvature matrix'''
    right_sides= np.zeros((PPE_params.shape[0],len(normMetricValsCol)))

    for metric_Idx in range(len(normMetricValsCol)):

        metrics_values = PPE_metrics[metric_Idx]
        metric_default = default_metrics[metric_Idx]

        right_sides[:,metric_Idx] = (metrics_values - metric_default) / np.abs((normMetricValsCol[0]))


    derivatives = np.linalg.lstsq(lin_system,right_sides)[0] 

    
    normlzdSensMatrix = derivatives[::2].T
    normlzdCurvMatrix = derivatives[1::2].T

    return normlzdSensMatrix, normlzdCurvMatrix



def get_PPE_paramNames(paramsNamesAndScales:np.ndarray):
    """
    Split paramsNamesAndScales into paramsNames and paramsScales
    
    :param paramsNamesAndScales: 2D-Array containing the parameter names with their scales
    """

    paramsNames = paramsNamesAndScales[:,0]

    paramsScales = paramsNamesAndScales[:,1]

    return paramsNames, paramsScales

def get_metrics_names(varPrefixes:list[str], boxSize:int):
    """
    Create a array containing the names of the metrics
    
    :param varPrefixes: List containing the prefixes of the used names of the used metrics
    :param boxSize: Size of the boxes over which the global data is averaged
    """
    metricsNames = []

    for varPrefix in varPrefixes:
        for i in range(1,(180//boxSize)+1):
            for j in range(1,(360//boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")

    metricsNames = np.array(metricsNames)

    return metricsNames


def get_weights_names(boxSize:int):
    """
    Create a array containing the names of the metrics weights
    
    :param boxSize: Size of the boxes over which the global data is averaged
    """

    weightsNames = []

    for i in range(1,(180//boxSize)+1):
        for j in range(1,(360//boxSize)+1):
            weightsNames.append(f"numb_{i}_{j}")

    weightsNames = np.array(weightsNames)

    return weightsNames


def setUp_x_ObsMetricValsDictPPE(obsMetrics: np.ndarray, obsWeights:np.ndarray, metricsNames:np.ndarray):
    """
    Create a dict in the same style it has for regular data
    
    :param obsMetrics: Metrics for the obeservational data
    :param obsWeights: Weights for the observational data
    :param metricsNames: Names of the Metrics e.g. "SWCF_1_1"
    """
    obsWeightsDict = {}

    obsMetricsDict = {}


    for idx, metricsName in enumerate(metricsNames):
        obsWeightsDict[metricsName] = obsWeights[idx]
        obsMetricsDict[metricsName] = obsMetrics[idx]

    return obsMetricsDict, obsWeightsDict


def remove_ctrl_and_validate_runs_from_PPE(ppe_dataset:xr.Dataset):
    """
    Removes ctrl, validate and Hxxx, Lxx runs from the PPE
    
    :param ppe_dataset: File containing the either PPE metrics or parameters

    """

    mask = ~ppe_dataset["ens_idx"].str.match(r"^(ctrl|validate|H|L)")
     
    sanitized_ppe_data = ppe_dataset.sel(ens_idx=mask)

    return sanitized_ppe_data