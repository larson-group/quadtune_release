import numpy as np 
import xarray as xr
from scipy.optimize import NonlinearConstraint, minimize


def calc_constrained_max(field_data, var_to_optimize, constraints, varPrefixes, num_metrics):

    
    var_top_optimize_idx = np.where(var_to_optimize == varPrefixes)[0]

    optimizing_field_data = field_data[var_top_optimize_idx*num_metrics:(var_top_optimize_idx+1)*num_metrics]



def minimize_constrained(SensMatrix, CurvMatrix, SensMatrix_F, CurvMatrix_F, initial_guess, bounds, varPrefixes_to_optimize, varPrefixes_to_constrain, thresholds, num_metrics):

    def fwd_model_PD(dp, SensMatrix, CurvMatrix):
        varPrefix_idx = np.where(varPrefixes_to_optimize == np.array(varPrefix))[0][0]
        data = (SensMatrix + 0.5*CurvMatrix * dp)@dp
        return np.sum(data[varPrefix_idx*int(num_metrics):(varPrefix_idx+1)*int(num_metrics)]**2)
    
    def fwd_model_F(dp):
        varPrefix_idx = np.where(varPrefixes_to_optimize == np.array(varPrefix))[0][0]
        data = (SensMatrix_F + 0.5*CurvMatrix_F * dp)@dp
        return np.sum(data[varPrefix_idx*int(num_metrics):(varPrefix_idx+1)*int(num_metrics)]**2)
    

    def objective_function(dp):
        return -fwd_model_F(dp)
    
    def get_constraint(varPrefix,threshold):
        varPrefix_idx = np.where(varPrefixes_to_constrain == np.array(varPrefix))[0][0]

        def constraint_function(dp):
            restrictedSensMatrix_PD = SensMatrix[varPrefix_idx*int(num_metrics):(varPrefix_idx+1)*int(num_metrics),:]
            restrictedCurvMatrix_PD = CurvMatrix[varPrefix_idx*int(num_metrics):(varPrefix_idx+1)*int(num_metrics),:]
            data = fwd_model_PD(dp,restrictedSensMatrix_PD, restrictedCurvMatrix_PD)
            return data
        return NonlinearConstraint(constraint_function, 0, threshold)
    
    constraints = []
    for varPrefix, threshold in zip(varPrefixes_to_constrain, thresholds):
        constraint = get_constraint(varPrefix,threshold)
        constraints.append(constraint)


    result = minimize(objective_function, initial_guess, constraints=constraints,method='COBYLA', bounds=bounds )
    print("Optimal parameters:", result.x)
    print("Maximum value of the objective function:", -result.fun)
    print("Success:", result.success)
    return result
