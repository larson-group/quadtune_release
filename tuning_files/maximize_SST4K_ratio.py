


import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, basinhopping

def maximizeSST4KRatio(normlzdSensMatrixPoly, normlzdCurvMatrix,
                       normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K,
                       normlzdWeightedSensMatrixPoly,
                       normlzdOrdDparamsMin, normlzdOrdDparamsMax,
                       doPiecewise, normlzd_dpMid,
                       normlzdLeftSensMatrix, normlzdRightSensMatrix,
                       normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,
                       metricsWeights, numMetrics,
                       magParamValsRow, defaultParamValsOrigRow,
                       paramsNames,
                       normlzdInteractDerivs=np.empty(0), interactIdxs=np.empty(0)):
    """
    Maximize the ratio of SST4K sensitivity to PD sensitivity using generalized eigenvalue problems
    and global optimization techniques.
    
    This function finds parameter perturbations that maximize the ratio of metric changes in SST4K
    simulations relative to metric changes in standard PD simulations. It uses both analytical
    solutions (generalized eigenvalue decomposition) and numerical optimization (COBYLA and basin hopping).
    
    
    :param normlzdSensMatrixPoly: numMetrics x numParams matrix of first-order derivatives
            (dmetrics/dparams).
    :param normlzdCurvMatrix: numMetrics x numParams matrix of second-order derivatives
            (d2metrics/dparams2).
    :param normlzdSensMatrixPolySST4K: numMetrics x numParams matrix of first-order derivatives
            for SST4K simulations.
    :param normlzdCurvMatrixSST4K: numMetrics x numParams matrix of second-order derivatives
            for SST4K simulations.
    :param normlzdWeightedSensMatrixPoly: Weighted version of normlzdSensMatrixPoly,
            with rows scaled by metricsWeights.
    :param normlzdOrdDparamsMin: Minimum normalized parameter perturbations
    :param normlzdOrdDparamsMax: Maximum normalized parameter perturbations
    :param doPiecewise: If True, use piecewise-linear sensitivity matrices; otherwise use
            quadratic Taylor expansion.
    :param normlzd_dpMid: Column vector of normalized parameter perturbations at the
            midpoint for piecewise-linear approximation.
    :param normlzdLeftSensMatrix: Sensitivity matrix for parameter values left of midpoint
            (standard PD).
    :param normlzdRightSensMatrix: Sensitivity matrix for parameter values right of midpoint
            (standard PD).
    :param normlzd_dpMidSST4K: Column vector of normalized parameter perturbations at the
            midpoint for piecewise-linear approximation (SST4K).
    :param normlzdLeftSensMatrixSST4K: Sensitivity matrix for parameter values left of midpoint
            (SST4K).
    :param normlzdRightSensMatrixSST4K: Sensitivity matrix for parameter values right of midpoint
            (SST4K).
    :param metricsWeights: Column vector of weights for each metric.
    :param numMetrics: Total number of metrics.
    :param magParamValsRow: Row vector of maximum parameter values used for normalization.
    :param defaultParamValsOrigRow: Row vector of default parameter values.
    :param paramsNames: List of parameter names.
    :param normlzdInteractDerivs: Array of interaction term derivatives, if any.
    "param interactIdxs: Array of (j,k) tuples of parameter indices for
            interaction terms. Defaults to empty array.
    
    Notes:
        - The function performs four different optimizations: linear and nonlinear using COBYLA,
          and linear and nonlinear using basinhopping global optimization.
        - Results are validated against the generalized eigenvalue problem solution.
        - Sanity checks verify that individual parameter contributions sum to the total.
    """
    print("-----------------Generalized Eigenvalue Problem and Ratio maximizing--------------------------\n")



    from quadtune_driver import calc_dimensional_param_vals, fwdFnc

    # Solve the generalized eigenvalue problem 
    dnormlzdParamsMaxSST4K, dnormlzdMetricsGenEig, dnormlzdMetricsGenEigSST4K =\
          solve_generalized_eigenvalue_problem(normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K, normlzdWeightedSensMatrixPoly, 
                 normlzdCurvMatrix, metricsWeights, paramsNames, magParamValsRow, defaultParamValsOrigRow,
                 doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,
                 normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,
                 numMetrics, normlzdInteractDerivs, interactIdxs)
    

    # Define the ratio function to be maximized

    
    def calc_SST4K_ratio(dnormlzdParams: np.ndarray, doNonLin: bool):

        normal= fwdFnc(dnormlzdParams.reshape((-1,1)),normlzdSensMatrixPoly, normlzdCurvMatrix * doNonLin, \
                        doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,\
                        numMetrics, normlzdInteractDerivs, interactIdxs)*metricsWeights 
        sst4k   = fwdFnc(dnormlzdParams.reshape((-1,1)), normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K * doNonLin, \
                        doPiecewise, normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,\
                        numMetrics, normlzdInteractDerivs, interactIdxs)*metricsWeights 
        return -1. * (sst4k.T@sst4k)/(normal.T@normal)
    

    # ================================================================================
    # LINEAR AND NONLINEAR COBYLA OPTIMIZATION
    # ================================================================================
    

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
    


    # ================================================================================
    # LINEAR AND NONLINEAR BASINHOPPINCOBYLA OPTIMIZATION
    # ================================================================================


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
        reference_ratio = -1 * calc_SST4K_ratio(dParams,doNonLin)
        for paramIdx in range(len(dParams)):
            for percentage in percentages:
                current_params = np.copy(dParams)
                current_params[paramIdx] *= percentage

                assert (newMaximum := -1*calc_SST4K_ratio(current_params,doNonLin)) <= reference_ratio, \
                f"Found new maximum with Parameter {paramIdx+1} multiplied with {percentage}. New maximum is: {newMaximum} \n"


    check_for_minimum_across_one_axis(res_lin.x, doNonLin=False)
    check_for_minimum_across_one_axis(res_nonlin.x, doNonLin=True)

            

    # Create values for plotting
    """
    These arrays contain the data for all plots.
        - First index: 0 -> non-linear, 1 -> linear
        - Second Index: 0 -> all parameters,  n -> Only using the parameter at position n-1
        - Third Index: Contains the actual data

        Example: [0,2,:] contains the data for the linear problem using only the second parameter

        The second index has an offset of 1 for the parameter indices, because the first index is reserved for the case of using all parameters.
        This is used in creating plots that show effects of single perturbed parameters compared to all parameters perturbed.
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

    return (dnormlzdMetricsGenEig, dnormlzdMetricsGenEigSST4K, 
            MetricsMaxRatioParams, MetricsSST4KMaxRatioParams)

    



def solve_generalized_eigenvalue_problem(normlzdSensMatrixPolySST4K, normlzdCurvMatrixSST4K, normlzdWeightedSensMatrixPoly, 
                 normlzdCurvMatrix, metricsWeights, paramsNames, magParamValsRow, defaultParamValsOrigRow,
                 doPiecewise, normlzd_dpMid, normlzdLeftSensMatrix, normlzdRightSensMatrix,
                 normlzd_dpMidSST4K, normlzdLeftSensMatrixSST4K, normlzdRightSensMatrixSST4K,
                 numMetrics, normlzdInteractDerivs, interactIdxs):
    from quadtune_driver import calc_dimensional_param_vals, fwdFnc
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
    
    return dnormlzdParamsMaxSST4K, dnormlzdMetricsGenEig, dnormlzdMetricsGenEigSST4K