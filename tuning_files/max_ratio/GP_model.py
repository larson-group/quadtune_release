import numpy as np

import torch
import gpytorch


import torch.nn as nn



from data_io import get_params_and_metrics_from_files
from optimization import normalize_metrics_data

class BatchedCompositeGPModel(gpytorch.models.ExactGP):
    """
    Batched Exact Gaussian Process model using a composite covariance kernel 
    combining an additive 1D RBF skeleton with a bounded isotropic RBF patch.

    Parameters
    ----------
    train_x : torch.Tensor
        Batched training input tensor of shape (num_tasks, num_samples, num_params).
    train_y : torch.Tensor
        Batched training target tensor of shape (num_tasks, num_samples).
    likelihood : gpytorch.likelihoods.Likelihood
        Gaussian likelihood module configured for the batch shape.
    k_dimensions : int
        Number of input parameter dimensions.
    min_dist : float
        Minimum Euclidean distance between training points in parameter space, 
        used to bound the lengthscale of the local isotropic RBF kernel.
    batch_shape : torch.Size
        Batch dimension corresponding to the number of regional output tasks.
    """
    def __init__(self, train_x, train_y, likelihood, k_dimensions, min_dist, batch_shape):
        super().__init__(train_x, train_y, likelihood)
        

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        

        additive_kernels = [
            gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(active_dims=(i,), batch_shape=batch_shape),
                batch_shape=batch_shape
            )
            for i in range(k_dimensions)
        ]
        k_additive = additive_kernels[0]
        for k in additive_kernels[1:]:
            k_additive = k_additive + k
            

        upper_bound = max(float(min_dist * 0.75), 1e-4)
        patch_base = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=gpytorch.constraints.Interval(1e-5, upper_bound),
            batch_shape=batch_shape
        )
        patch_base.lengthscale = float(min_dist / 2.0)
        
        k_patch = gpytorch.kernels.ScaleKernel(patch_base, batch_shape=batch_shape)
        

        self.covar_module = k_additive + k_patch

    def forward(self, x):
        """
        Computes the prior or posterior MultivariateNormal distribution at `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input parameter tensor.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Evaluated multivariate normal distribution across all batched tasks.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def construct_gp_model(X_train, y_train):
    """
    Constructs and trains a batched composite GPyTorch emulator model via 
    Exact Marginal Log-Likelihood optimization.

    Parameters
    ----------
    X_train : numpy.ndarray
        2D array of normalized training parameter sets of shape (num_samples, num_params).
    y_train : numpy.ndarray
        2D array of normalized regional target metrics of shape (num_samples, num_tasks).

    Returns
    -------
    BatchedCompositeGPModel
        Trained GPyTorch model in evaluation mode with its likelihood attached.
    """
    num_tasks = y_train.shape[1] 
    batch_shape = torch.Size([num_tasks])
    

    X_base = torch.as_tensor(X_train, dtype=torch.float64)
    X_tensor = X_base.unsqueeze(0).expand(num_tasks, *X_base.shape)
    

    y_tensor = torch.as_tensor(y_train, dtype=torch.float64).T
    
    k_dimensions = X_base.shape[1]
    

    if len(X_base) <= 1:
        min_dist = 1.0
    else:
        distances = torch.cdist(X_base, X_base)
        distances.fill_diagonal_(float('inf'))
        min_dist = torch.min(distances).item()


    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=batch_shape, 
        dtype=torch.float64,
        noise_constraint=gpytorch.constraints.Interval(1e-6, 1e-5)
    )

    likelihood.noise = 1e-6
    
    model = BatchedCompositeGPModel(X_tensor, y_tensor, likelihood, k_dimensions, min_dist, batch_shape)
    
    model = model.double()
    likelihood = likelihood.double()
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    n_iters = 150
    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(X_tensor)
        
        loss = -mll(output, y_tensor).sum() 
        loss.backward()
        optimizer.step()
        
    model.eval()
    likelihood.eval()
    model.likelihood = likelihood
    
    return model


def predict(model, X_test, include_observation_noise=False):
    """
    Evaluates the batched GPyTorch model at test parameter locations and returns 
    predictive means, standard deviations, and 95% confidence bounds.

    Parameters
    ----------
    model : BatchedCompositeGPModel
        Trained GPyTorch model containing a `.likelihood` attribute.
    X_test : numpy.ndarray or torch.Tensor
        1D or 2D array/tensor of normalized parameter vectors to evaluate.
    include_observation_noise : bool, optional
        If True, passes the latent distribution through the Gaussian likelihood 
        to include observation noise variance (default is False).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        - mean : Predictive mean of shape (num_samples, num_tasks).
        - std : Predictive standard deviation of shape (num_samples, num_tasks).
        - lower_95 : Lower 95% confidence bound of shape (num_samples, num_tasks).
        - upper_95 : Upper 95% confidence bound of shape (num_samples, num_tasks).
    """
    if isinstance(X_test, np.ndarray):
        X_test_tensor = torch.as_tensor(X_test, dtype=torch.float64)
    else:
        X_test_tensor = X_test.to(dtype=torch.float64)
        
    if X_test_tensor.ndim == 1:
        X_test_tensor = X_test_tensor.unsqueeze(0) # (1, 14)
        
    num_tasks = model.mean_module.batch_shape[0]
    

    X_test_batched = X_test_tensor.unsqueeze(0).expand(num_tasks, *X_test_tensor.shape)
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        latent_pred = model(X_test_batched)
        
        if include_observation_noise:
            pred_dist = model.likelihood(latent_pred)
        else:
            pred_dist = latent_pred

        mean = pred_dist.mean.cpu().numpy()
        std = pred_dist.stddev.cpu().numpy()
        
        lower, upper = pred_dist.confidence_region()
        lower_95 = lower.cpu().numpy()
        upper_95 = upper.cpu().numpy()

    return mean.T, std.T, lower_95.T, upper_95.T



def normalize_data(X, X_default, y, y_default, global_average_obs):
    """
    Normalizes training parameter sets and regional target metrics relative to 
    the default simulation and global reference averages.

    Parameters
    ----------
    X : numpy.ndarray
        2D array of physical parameter values.
    X_default : numpy.ndarray
        1D array of default physical parameter values.
    y : numpy.ndarray
        2D array of regional metric outputs in physical units.
    y_default : numpy.ndarray
        1D array of default regional metric outputs in physical units.
    global_average_obs : numpy.ndarray
        1D array of global reference averages for each metric.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        Normalized parameter array `X_normalized` and normalized metric array `y_normalized`.
    """
    X_normalized = (X - X_default) / X_default
    y_normalized = normalize_metrics_data(y, y_default, global_average_obs)

    return X_normalized, y_normalized


def construct_train_data(files, params_names, varNames, boxSize):
    """
    Wrapper function that extracts training parameters and regional metrics 
    from a list of NetCDF files.

    Parameters
    ----------
    files : list of str
        Paths to the regional NetCDF files.
    params_names : list of str
        Names of the tunable parameters to extract.
    varNames : list of str
        Names of the climate variables to extract.
    boxSize : int
        Grid resolution in degrees (assumes a global 360x180 grid).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        2D array of training parameters `X_train` and 2D array of regional 
        metrics `y_train`.
    """
    X_train, y_train = get_params_and_metrics_from_files(files, params_names, varNames, boxSize)
    return X_train, y_train






class CombinedGP(nn.Module):
    """
    PyTorch module wrapper that concatenates the predictive means of two 
    batched GPyTorch models along the spatial task dimension.

    Parameters
    ----------
    model1 : BatchedCompositeGPModel
        First GPyTorch emulator model (e.g., primary constraint field).
    model2 : BatchedCompositeGPModel
        Second GPyTorch emulator model (e.g., secondary constraint field).
    """
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        """
        Evaluates both GP models at `x` and concatenates their mean predictions.

        Parameters
        ----------
        x : torch.Tensor
            Input parameter perturbation tensor.

        Returns
        -------
        DummyOutput
            Object with a `.mean` attribute containing the concatenated 
            prediction tensor.
        """
        out1 = self.model1(x).mean
        out2 = self.model2(x).mean
        
        stacked_mean = torch.cat([out1, out2], dim=-1)
        
        class DummyOutput:
            def __init__(self, mean):
                self.mean = mean
                
        return DummyOutput(stacked_mean)


def create_torch_evaluator(gp_model):
    """
    Wraps a trained GPyTorch model into a NumPy-compatible callable function.

    Parameters
    ----------
    gp_model : gpytorch.models.ExactGP
        Trained GPyTorch model to wrap.

    Returns
    -------
    callable
        Function that takes a 1D NumPy parameter array `dp` of shape (P_n,) 
        and returns a 1D NumPy prediction array of shape (N,).
    """
    gp_model.eval()  
    
    def evaluator(dp):

        dp_tensor = torch.tensor(dp, dtype=torch.float64).unsqueeze(0)
        
        
        with torch.no_grad():
            pred = gp_model(dp_tensor).mean.squeeze(0).numpy()
            
        return pred
        
    return evaluator