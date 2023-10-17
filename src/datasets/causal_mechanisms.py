"""Utilities for causal mechanisms generation.
"""

# import GPy
import numpy as np
from typing import Tuple, Any
from numpy.typing import NDArray
from torch import nn, from_numpy
from abc import ABCMeta, abstractmethod
from sklearn.gaussian_process.kernels import PairwiseKernel
from sklearn.linear_model import LinearRegression



# Base class for causal mechanism generation
class PredictionModel(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        raise NotImplementedError



# * Linear mechanisms *
class LinearMechanism(PredictionModel):
    """Linear causal mechanism by linear regression.

    Parameters
    ----------
    min_weight: float, default is -1
        Minimum value of causal mechanisms weights
    max_weight: float, default is 1
        Maximum value of causal mechanisms weights
    min_abs_weight: float, default is 0.05
        Minimum value of the absolute value of any causal mechanism weight.
        Low value of min_abs_weight potentially lead to lambda-unfaithful distributions.
    """

    def __init__(
        self,
        min_weight: float=-1.,
        max_weight: float=1.,
        min_abs_weight: float=0.05
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_abs_weight = min_abs_weight
        
        # One of max_weight or min_weight must be larger (abs value) than min_abs_weight
        if not(abs(max_weight) > min_abs_weight or abs(min_weight) > min_abs_weight):
            raise ValueError("The range of admitted weights is empty. Please set"\
                             " one between `min_weight` and `max_weight` with absolute"\
                             "  value larger than `min_abs_weight`.")
        
        self.linear_reg = LinearRegression(fit_intercept=False)
        
    
    def predict(self, X: NDArray) -> NDArray:
        """Transform X via a linear causal mechanism.

        Parameters
        ----------
        X: NDArray of shape (num_samples, num_parents)
            Samples of the parents to be transformed by the causal mechanism.

        Returns
        -------
        y: NDArray
            The output of the causal mechanism
        """
        if X.ndim != 2:
            raise ValueError(f"Number of dimensions {X.ndim} different from 2."\
             " If input has 1 dimension, consider reshaping it with reshape(-1, 1)")
        n_covariates = X.shape[1]      

        # Random initialization of the causal mechanism
        self.linear_reg.coef_ = np.random.uniform(self.min_weight, self.max_weight, n_covariates) 

        # Reject ~0 coefficients
        for i in range(n_covariates):
            while (abs(self.linear_reg.coef_[i]) < self.min_abs_weight):
                self.linear_reg.coef_[i] = np.random.uniform(self.min_weight, self.max_weight, 1)
            
        self.linear_reg.intercept_ = 0
        effect = self.linear_reg.predict(X)
        return effect    


# * Nonlinear mechanisms *
class NeuralNetMechanism(PredictionModel):
    """Nonlinear causal mechanism parametrized by a neural network.
    """
    def __init__(
        self,
        weights_mean: float = 0.,
        weights_std: float = 1.,
        hidden_dim: int = 10,
        activation: nn.Module = nn.PReLU()
    ):
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.hidden_dim = hidden_dim
        self.activation = activation

        self._model = None


    def predict(
        self, 
        X: NDArray
    ) -> NDArray:
        """Generate the effect given the parents.

        The effect is generated as a nonlinear function parametrized by a neural network
        with a single hidden layer. 

        Parameters
        ----------
        X : NDArray of shape (num_samples, num_parents)
            Input of the neural network with the parent node instances.

        Returns
        -------
        effect: NDArray of shape (num_samples)
            The output of the neural network with X as input.
        """
        n_samples = X.shape[0]
        n_causes = X.shape[1]

        # Make architecture
        self._model = nn.Sequential(
            nn.Linear(n_causes, self.hidden_dim),
            self.activation,
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1)
        )
        self._model.apply(self._weight_init)

        # Convert to torch.Tensor for forward pass
        data = X.astype('float32')
        data = from_numpy(data)

        # Forward pass
        effect = np.reshape(self.model(data).data, (n_samples,))

        return effect.numpy()
    

    def _weight_init(self, module):
        """Random initialization of model weights.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight.data, mean=self.weights_mean, std=self.weights_std
            )

    @property
    def model(self):
        if self._model is None:
            raise ValueError("Torch model not initialized. Call `self.predict()` first")
        return self._model


class GaussianProcessMechanism(PredictionModel):
    def __init__(
        self,
        # lengthscale: float,
        # variance: float
        gamma: float = 1.
    ):
        """Nonlinear causal mechanism sampled from a gaussian process.

        Parameters
        ----------
        gamma : float
            The gamma parameters fixing the variance of the kernel. 
            Larger values of gamma determines bigger magnitude of the causal mechanisms.
        # lengthscale: float
        #     The lengthscale of the RBF kernel.
        # variance: float
        #     The variance of the rbf kernel. This determines the magnitude of the
        #     causal mechanism outputs (the larger the variacne the larger the output)
        """
        # self.lengthscale = lengthscale
        # self.variance = variance
        self.rbf = PairwiseKernel(gamma=gamma, metric="rbf")


    def predict(self, X: NDArray) -> NDArray:
        """Generate the effect given the parents.

        The effect is generated as a nonlinear function sampled from a 
        gaussian process.

        Parameters
        ----------
        X : NDArray of shape (num_samples, num_parents)
            Input of the RBF kernel.

        Returns
        -------
        effect: NDArray of shape (num_samples)
            Causal effect sampled from the gaussina process with
            covariance matrix given by the RBF kernel with X as input.
        """
        num_samples = X.shape[0]
        # rbf = GPy.kern.RBF(input_dim=X.shape[1],lengthscale=self.lengthscale,variance=self.f_magn)
        # covariance_matrix = rbf.K(X,X)
        covariance_matrix = self.rbf(X, X)

        # Sample the effect as a zero centered normal with covariance given by the RBF kernel
        effect = np.random.multivariate_normal(np.zeros(num_samples), covariance_matrix)
        return effect


# Base class for PostnonLinearModel invertible functions
class InvertibleFunction(metaclass=ABCMeta):
    """Invertible functions for the post-nonlinear model abstract class. 
    
    The class implementing `InvertibleFunction` must have a `forward` method
    applying an invertible transformation to the input. 
    If the transformation is not invertible, then the post-nonlinear model
    is not identifiable. 
    """
    
    @abstractmethod
    def forward(self, input: NDArray):
        raise NotImplementedError("InvertibleFunction is missing the required forward method.")
    
    def __call__(self, input: NDArray):
        return self.forward(input)


class Identity(InvertibleFunction):
    def forward(self, input: NDArray):
        return input
