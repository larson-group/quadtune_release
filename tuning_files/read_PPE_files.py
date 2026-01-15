from typing import Any
import xarray as xr
import numpy as np




def construct_sensitivity_curvature_matrices_from_PPE_data(PPE_metrics_filename: str, PPE_params_filename:str, metricsNames: np.ndarray, paramsIndices: np.ndarray, normMetricValsCol):
    """
    Create Sensitivity and Curvature matrices from PPE data by repeatingly solving the system

    | dp_1(PPE1)  0.5dp_1^2(PPE1) dp_2(PPE1) ... |    |  ∂m_i/∂p_1  |    |dm_i(PPE1)|
    | dp_1(PPE2)  0.5dp_1^2(PPE2) dp_2(PPE2) ... |    |∂m_i^2/∂p_1^2|    |dm_i(PPE2)|
    | dp_1(PPE3)  0.5dp_1^2(PPE3) dp_2(PPE3) ... |    |  ∂m_i/∂p_2  | =  |dm_i(PPE3)|
    |                ...                         |    |    ...      |    |   ...    |

    for different metrics m_i.
 
    
    :param PPE_metrics_filename: Name of the file containing the regional metrics

    :param PPE_parameter_filename: Name of the file containing the parameters of each PPE member

    :param metricsNames: List containing the names of all used metrics

    """


    '''Read in and preprocess data'''

    metrics_dataset = xr.open_dataset(PPE_metrics_filename,engine="netcdf4")

    params_dataset = xr.open_dataset(PPE_params_filename,engine="netcdf4")
    
    

    default_metrics = metrics_dataset.sel(ens_idx="ctrl").isel(time=0,product=0)
    
    default_params = params_dataset.where(params_dataset.ens_idx.str.match('ctrl'), drop= True).params.values[0,paramsIndices]


    metrics_dataset_no_ctrl, params_dataset_no_ctrl = preprocess_ppe_data(PPE_metrics_filename, PPE_params_filename)
    metrics_dataset_no_ctrl = metrics_dataset_no_ctrl

    '''Create the static leftside matrix of the System'''
    lin_system = np.empty((params_dataset_no_ctrl.shape[0],len(paramsIndices)*2))
    for i in range(lin_system.shape[0]):

        lin_row_params = params_dataset_no_ctrl[i,:].values[paramsIndices]
        

        dnormlzdlin_row_params = (lin_row_params - default_params) / np.abs(default_params)

        dnormlzdquad_row_params = 0.5*dnormlzdlin_row_params**2


        full_row = np.dstack((dnormlzdlin_row_params,dnormlzdquad_row_params)).flatten()
        lin_system[i,:] = full_row

    normlzdOrdDparamsMin = np.tile(np.min(params_dataset_no_ctrl.values[:,paramsIndices],axis=0)/np.abs(default_params) - np.ones_like(default_params),(len(metricsNames),1))

    normlzdOrdDparamsMax = np.tile(np.max(params_dataset_no_ctrl.values[:,paramsIndices],axis=0)/np.abs(default_params) - np.ones_like(default_params),(len(metricsNames),1))


    '''Solve the linear system multiple times with for different metrics, to create the sensitivity and curvature matrix'''
    right_sides= np.zeros((params_dataset_no_ctrl.shape[0],len(metricsNames)))
    


    for metric_Idx in range(len(metricsNames)):

        metrics_values = metrics_dataset_no_ctrl[metricsNames[metric_Idx]] 
        metric_default = default_metrics[metricsNames[metric_Idx]]

        right_sides[:,metric_Idx] = (metrics_values - metric_default) / -(normMetricValsCol[0])



    derivatives = np.linalg.lstsq(lin_system,right_sides)[0] 

    
    normlzdSensMatrix = derivatives[::2].T
    normlzdCurvMatrix = derivatives[1::2].T

    return normlzdSensMatrix, normlzdCurvMatrix, normlzdOrdDparamsMin, normlzdOrdDparamsMax




def get_PPE_paramNames(paramsNamesAndScales:np.ndarray):
    """
    Split paramsNamesAndScales into paramsNames and paramsScales
    
    :param paramsNamesAndScales: 2D-Array containing the parameter names with their scales
    """

    paramsNames = paramsNamesAndScales[:,0]

    paramsScales = paramsNamesAndScales[:,1]

    return paramsNames, paramsScales



def get_PPE_default_metrics_names_and_metrics_weights(PPE_metrics_filename:str, varPrefixes:list[str],boxSize):
    """
    Extract the names of the used metrics, and the corresponding weights for the default case.
    
    :param PPE_metrics_filename: Path to the file containing the metrics data

    :param varPrefixes: List containing the prefixes of the used names of the used metrics

    :param boxSize: Size of the boxes on the map

    """

    metrics_dataset = xr.open_dataset(PPE_metrics_filename,engine="netcdf4")

    default_metrics = metrics_dataset.sel(ens_idx="ctrl").isel(time=0,product=0)



    '''Create list containing all names of the used metrics'''
    metricsNames = []
    weightsNames = []

    for varPrefix in varPrefixes:
        for i in range(1,(180//boxSize)+1):
            for j in range(1,(360//boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")
                weightsNames.append(f"numb_{i}_{j}")

    metricsWeights = default_metrics[weightsNames].to_array(dim="metricsNames").to_numpy()


    return  np.array(metricsNames), default_metrics[metricsNames].to_array().values.reshape((-1,1)), metricsWeights.reshape((-1,1))



def get_PPE_default_params(PPE_params_filename:str, paramsIndices: np.ndarray):
    """
    Extract the default parameters from the PPE file
    
    :param PPE_params_filename: The file containing the PPE parameters
    ;param paramsIndices
    """
    params_dataset = xr.open_dataset(PPE_params_filename,engine="netcdf4")
    default_params = params_dataset.where(params_dataset.ens_idx.str.match('ctrl'), drop= True).params.values

    return default_params[0, paramsIndices].reshape((1,-1))


def get_PPE_param_idxs(allparamsNamesInFile: np.ndarray, paramsNames: np.ndarray):
    """
    Computes the indices of the used parameters in the list of all parameters
    
    :param allparamsNamesInFile: Numpy array containing all parameters in the file
    :param paramsNames: Numpy array containing the parameters which should be used by QuadTune
    """

    return np.where(np.isin(allparamsNamesInFile, paramsNames))[0]

    

def get_PPE_obs_metrics_weights(PPE_metrics_filename:str, metricsNames:list[str]):
    """
    Extract the obs metricsWeights from the PPE file
    
    :param PPE_metrics_filename: Path to the file containing the metrics data
    :param metricsNames: List containing all names of the used metrics
    """

    metrics_dataset = xr.open_dataset(PPE_metrics_filename,engine="netcdf4")

    obsMetrics = metrics_dataset[metricsNames].to_array(dim="metricsName").isel(time=0,product=1,ens_idx=0)


    #This expects the length of all prefixes to be 4
    # weightsNames =["numb" + name[4:] for name in metricsNames]


    obsWeights = metrics_dataset["weights"].values

    # check if only a single obsWeight was given. If so, extend it to all metrics
    if len([obsWeights])  == 1:
        helper_array = np.ones(len(metricsNames)) * obsWeights
        obsWeights = helper_array


    normlzdObsWeights = obsWeights/np.sum(obsWeights)

    return obsMetrics.to_numpy().reshape((-1,1)), normlzdObsWeights.reshape((-1,1))


def setUp_x_ObsMetricValsDictPPE(obsMetrics,obsWeights,metricsNames):

    obsWeightsDict = {}

    obsMetricsDict = {}


    for idx, metricsName in enumerate(metricsNames):
        obsWeightsDict[metricsName] = obsWeights[idx]
        obsMetricsDict[metricsName] = obsMetrics[idx]

    return obsMetricsDict, obsWeightsDict

def preprocess_ppe_data(PPE_metrics_filename,PPE_parameter_filename):
    """
    Removes ctrl, validate and Hxxx, Lxx runs from the PPE
    
    :param PPE_metrics_filename: File containing the regional PPE metrics
    :param PPE_parameter_filename: File containing the PPE parameters
    """

    metrics_dataset = xr.open_dataset(PPE_metrics_filename,engine="netcdf4")
    params_dataset = xr.open_dataset(PPE_parameter_filename,engine="netcdf4")

    mask = metrics_dataset["ens_idx"].str.match(r"^(ens|hm)")

    if not mask.any():
        mask = ~params_dataset.ens_idx.str.match('ctrl')

    
    preprocessed_metrics = metrics_dataset.isel(time=0,product=0).sel(ens_idx=mask)
    preprocessed_parameters = params_dataset.params.sel(ens_idx=mask)


    return preprocessed_metrics, preprocessed_parameters

