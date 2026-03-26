from email.policy import default
from typing import Any
import xarray as xr
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


def restrict_dataset_by_metric_loss(params_dataset,metrics_dataset,varPrefixes,boxSize, threshold):



    metric_names = []
    for varPrefix in varPrefixes:
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metric_names.append(f"{varPrefix}_{i}_{j}")

    ctrl = metrics_dataset.where(metrics_dataset.ens_idx.str.match('ctrl'), drop= True)[metric_names].isel(time=0,product=0).to_array().values
    

    clean_params = remove_ctrl_and_validate_runs_from_PPE(params_dataset)
    clean_metrics = remove_ctrl_and_validate_runs_from_PPE(metrics_dataset)

    metric_data = clean_metrics[metric_names].isel(time=0,product=0).to_array().values

    losses = np.sum((ctrl - metric_data)**2, axis=0)


    quantile = np.percentile(losses, 100*threshold)

    indices = np.where(losses <= quantile)[0]


    print(f"Number of PPE runs before restriction by metric loss: {len(losses.flatten())}")
    print(f"Number of PPE runs after restriction by metric loss: {indices.size}")

    valid_ens_idx = clean_metrics.ens_idx.values[indices]

    valid_ens_idx = np.append(valid_ens_idx, 'ctrl')


    restricted_paramsDataset = params_dataset.sel(ens_idx=valid_ens_idx)
    restricted_metricsDataset = metrics_dataset.sel(ens_idx=valid_ens_idx)
        
    return restricted_paramsDataset, restricted_metricsDataset


def restrict_dataset_by_param(params_dataset, metrics_dataset, metric_names, param_index, lower_bound = None, upper_bound = None, min_loss=None):


    if lower_bound is None:
        lower_bound = -np.inf
    if upper_bound is None:
        upper_bound = np.inf

    ctrl_params =  params_dataset.where(params_dataset.ens_idx.str.match('ctrl'), drop= True).params.values.flatten()
    ctrl_metrics = metrics_dataset.where(metrics_dataset.ens_idx.str.match('ctrl'), drop= True)[metric_names].isel(time=0,product=0).to_array().values


    indices = np.where((params_dataset.params[:,param_index] > lower_bound) & (params_dataset.params[:,param_index] < upper_bound))[0]

    print(f"Number of PPE runs before restriction by parameter value: {params_dataset.ens_idx.size}")
    print(f"Number of PPE runs after restriction by parameter value: {indices.size}")
        

    restricted_paramsDataset = params_dataset.isel(ens_idx=indices)
    restricted_metricsDataset = metrics_dataset.isel(ens_idx=indices)


    #if ctrl is outside of the bounds, set the PPE member with the lowest metric loss to ctrl
    if ctrl_params[param_index] <= lower_bound or ctrl_params[param_index] >= upper_bound:
        if not min_loss:
            print("Ctrl is outside of the bounds, setting the PPE member with the lowest metric loss to ctrl")
            metric_data = restricted_metricsDataset[metric_names].isel(time=0,product=0).to_array().values
            losses = np.sum((ctrl_metrics - metric_data)**2, axis=0)

            min_loss = np.argmin(losses)

        temp_ens_idx = restricted_metricsDataset['ens_idx'].values.copy()

        temp_ens_idx[min_loss] = 'ctrl'

        restricted_metricsDataset = restricted_metricsDataset.assign_coords(ens_idx=temp_ens_idx)
        restricted_paramsDataset = restricted_paramsDataset.assign_coords(ens_idx=temp_ens_idx)

    else:
        min_loss = None

    return restricted_paramsDataset, restricted_metricsDataset, min_loss

def process_PPE_params_file(params_dataset, paramsNamesAndScales: np.ndarray, allparamsNamesInFile: list[str]):
    """
    Process the PPE parameter file and extract:
        - default parameters
        - all PPE parameters
        - parameter Names and scales for plotting

    :param PPE_params_filename: Path to a file containing parameters for a PPE
    :param paramsNamesAndScales: 2D-Array containing the parameter names with their scales
    :param allparamsNamesInFile: List containing all parameter names in the file

    
    """

    paramsNames, paramsScales = get_PPE_paramNames(paramsNamesAndScales)

    paramsIndices = np.where(np.isin(allparamsNamesInFile, paramsNames))[0]

    # paramsIndices = np.array([np.where(paramsNames == name)[0][0] for name in allparamsNamesInFile])


    defaultParamValsOrigRow = params_dataset.where(params_dataset.ens_idx.str.match('ctrl'), drop= True).params.values[0,paramsIndices]




    #PPE parameters without ctrl, validate, L00X and R00X ensemble members stored in numb_ensemble_members x num_params array
    PPE_parameters = remove_ctrl_and_validate_runs_from_PPE(params_dataset).params.values[:,paramsIndices]



    magParamValsRow = np.abs(defaultParamValsOrigRow)

    minParams = np.min(PPE_parameters,axis=0)

    maxParams = np.max(PPE_parameters,axis=0)

    for idx, val in enumerate(magParamValsRow):
        if val <= np.finfo(val.dtype).eps:
            magParamValsRow[idx] = maxParams[idx]


    normlzdOrdDparamsMinFlat = (minParams-defaultParamValsOrigRow)/np.abs(magParamValsRow)

    normlzdOrdDparamsMaxFlat = (maxParams-defaultParamValsOrigRow)/np.abs(magParamValsRow)


    magParamValsRow = magParamValsRow.reshape((1,-1))

    defaultParamValsOrigRow = defaultParamValsOrigRow.reshape((1,-1))

    return defaultParamValsOrigRow, PPE_parameters, paramsNames, paramsIndices, paramsScales, magParamValsRow, normlzdOrdDparamsMinFlat, normlzdOrdDparamsMaxFlat

    


def process_ppe_metrics_file(metrics_dataset, varPrefixes:list[str], boxSize:int):
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


    metricsNames = get_metrics_names(varPrefixes, boxSize)

    weightsNames = get_weights_names(boxSize)

    default_metrics_dataset = metrics_dataset.sel(ens_idx="ctrl")

    PPE_metrics = remove_ctrl_and_validate_runs_from_PPE(metrics_dataset).isel(time=0,product=0)[metricsNames.flatten()].to_array().values

    metricsWeights = default_metrics_dataset.isel(time=0,product=0)[weightsNames].to_array(dim="metricsNames").to_numpy()

    


    default_metrics = default_metrics_dataset.isel(time=0,product=0)[metricsNames.flatten()]

    defaultMetricValsCol = default_metrics[metricsNames.flatten()].to_array().values.reshape(-1,1)


    
    normlzdMetricsWeights = (metricsWeights/np.sum(metricsWeights))

    #Repeat the weights to be used for multiple fields
    normlzdMetricsWeights = np.tile(normlzdMetricsWeights,len(varPrefixes)).reshape(-1,1)
    metricsWeights = np.tile(metricsWeights,len(varPrefixes)).reshape(-1,1)

    

    obsMetricValsCol = metrics_dataset[metricsNames.flatten()].to_array(dim="metricsName").isel(time=0,product=1,ens_idx=0).to_numpy().reshape(-1,1)


    # check if only a single obsWeight was given. If so, use metricWeights instead
    if "obs_weights" not in metrics_dataset.keys():  # This is the case for the file created by Zhun from the 2025 E3SM Zenodo archive
        ObsWeights = normlzdMetricsWeights
    else:
        ObsWeights = metrics_dataset["obs_weights"].values # This is the case for the file created from Regional files

    
    return defaultMetricValsCol, PPE_metrics, metricsNames, normlzdMetricsWeights, obsMetricValsCol, ObsWeights




    



def construct_sensitivity_curvature_matrices_from_PPE_data(PPE_metrics:np.ndarray, default_metrics:np.ndarray, PPE_params:np.ndarray, default_params:np.ndarray, normMetricValsCol:np.ndarray, magParamValsRow:np.ndarray,doRegularizeByRegError):
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

    dnormlzdlin = (PPE_params - default_params) / np.abs(magParamValsRow)
    dnormlzdquad = 0.5*dnormlzdlin**2

    lin_system = np.empty((PPE_params.shape[0],PPE_params.shape[1]*2))
    lin_system[:,0::2] = dnormlzdlin 
    lin_system[:,1::2] = dnormlzdquad
    


    # default_metrics = default_metrics.reshape((default_metrics.shape[0]*default_metrics.shape[1],-1),order='F')
    '''Solve the linear system for each metric, to create the sensitivity and curvature matrix'''
    right_sides= np.zeros((PPE_params.shape[0],len(normMetricValsCol)))

    for metric_Idx in range(len(normMetricValsCol)):

        metrics_values = PPE_metrics[metric_Idx]
        metric_default = default_metrics[metric_Idx]

        right_sides[:,metric_Idx] = (metrics_values - metric_default) / np.abs((normMetricValsCol[metric_Idx]))


    doRegularize = True
    doNormalize = True

    if doNormalize:
        sigma = np.std(lin_system,axis=0)
        sigma[sigma==0] = 1.0
        lin_system = lin_system/sigma


    if doRegularize:
        alpha = 1e-4


    else:
        alpha=0
        

    ridge_solver = Ridge(alpha=alpha, fit_intercept=False)



    # scaler = StandardScaler(with_mean=False)
    # lin_system = scaler.fit_transform(lin_system)

    

    # 

    print(f"CONDITION: {np.linalg.cond(lin_system.T@lin_system)}")
    print(f"CONDITION(Ridge): {np.linalg.cond(lin_system.T@lin_system + 1e-4*np.eye((lin_system.T@lin_system).shape[0]))}")
    ridge_solver.fit(lin_system, right_sides)

    # print(ridge_solver.alpha_)

    if doNormalize:
        derivatives = (ridge_solver.coef_/sigma).T
    else:
         derivatives = ridge_solver.coef_.T
    # derivatives = ridge_solver.coef_.T / scaler.scale_[:, np.newaxis]

    if doRegularizeByRegError:
        doRegularizeByRegError = False
        full_residuals = right_sides-lin_system @ derivatives
        mean_squared_residuals = np.mean(full_residuals**2,axis=1)

        q25, q75, q50 = np.percentile(mean_squared_residuals, [25, 75,50])
        iqr = q75 - q25
        upper_bound = 3 * iqr

        indices_to_remove = np.where(mean_squared_residuals > upper_bound)[0]
        PPE_metrics = np.delete(PPE_metrics, indices_to_remove, axis=1)
        PPE_params = np.delete(PPE_params, indices_to_remove, axis=0)
        print(f"Removed {len(indices_to_remove)} PPE runs with high residuals. Remaining runs: {PPE_params.shape[0]}")

        return construct_sensitivity_curvature_matrices_from_PPE_data(PPE_metrics, default_metrics, PPE_params, default_params, normMetricValsCol, magParamValsRow, doRegularizeByRegError)

    
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
        for i in range(1,int(180/boxSize)+1):
            for j in range(1,int(360/boxSize)+1):
                metricsNames.append(f"{varPrefix}_{i}_{j}")

    metricsNames = np.array(metricsNames)

    return metricsNames


def get_weights_names(boxSize:int):
    """
    Create a array containing the names of the metrics weights
    
    :param boxSize: Size of the boxes over which the global data is averaged
    """

    weightsNames = []

    for i in range(1,(int(180/boxSize)+1)):
        for j in range(1,int(360/boxSize)+1):
            weightsNames.append(f"numb_{i}_{j}")

    weightsNames = np.array(weightsNames)

    return weightsNames


def setUp_x_ObsMetricValsDictPPE(obsMetrics: np.ndarray, obsWeights:np.ndarray, metricsNames:np.ndarray, varPrefixes:list[str]):
    """
    Create a dict in the same style it has for regular data
    
    :param obsMetrics: Metrics for the obeservational data
    :param obsWeights: Weights for the observational data
    :param metricsNames: Names of the Metrics e.g. "SWCF_1_1"
    :param varPrefixes: List containing the prefixes of the used names of the used metrics
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
    # mask = ppe_dataset["ens_idx"].str.match(r"^(validate)")
    sanitized_ppe_data = ppe_dataset.sel(ens_idx=mask)
    

    return sanitized_ppe_data


def restrict_by_common_members(PPE_params, PPE_metrics, PPE_params_sst4k):
    """
    Restrict the PPE datasets to the common members between the two datasets,
    ensuring a 1-to-1 alignment.
    """
    default_params = PPE_params["params"].values
    sst4k_params = PPE_params_sst4k["params"].values

    matching_indices_default = []
    matching_indices_sst4k = []


    used_sst4k_indices = set()

    for idx, default_param_set in enumerate(default_params):
        matches = np.where(np.all(np.isclose(sst4k_params, default_param_set, atol=1e-6), axis=1))[0]

        for match_idx in matches:
            if match_idx not in used_sst4k_indices:
                matching_indices_default.append(idx)
                matching_indices_sst4k.append(match_idx)
                used_sst4k_indices.add(match_idx)
                break 

    print(f"Number of common, unique member pairs: {len(matching_indices_default)}")

    PPE_params = PPE_params.isel(ens_idx=matching_indices_default)
    PPE_metrics = PPE_metrics.isel(ens_idx=matching_indices_default)
    


    return PPE_params, PPE_metrics

