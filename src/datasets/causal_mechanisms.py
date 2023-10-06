"""Utilities for causal mechanisms generation.
"""

# import GPy
import numpy as np
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
        min_weight: float,
        max_weight: float,
        min_abs_weight: float
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_abs_weight = min_abs_weight
    
    def predict(self, X: NDArray) -> NDArray:
        n_covariates = X.shape[1]        
        linear_reg = LinearRegression()
        linear_reg.coef_ = np.random.uniform(self.min_weight, self.max_weight, n_covariates) 

        # Reject ~0 coefficients
        for i in range(n_covariates):
            while (abs(linear_reg.coef_[i]) < self.min_abs_weight):
                linear_reg.coef_[i] = np.random.uniform(self.min_weight, self.max_weight, 1)
            
        linear_reg.intercept_ = 0
        y = linear_reg.predict(X)
        return y


# * Nonlinear mechanisms *
class NeuralNetMechanism(PredictionModel):
    """Nonlinear causal mechanism parametrized by a neural network.
    """
    def __init__(
        self,
        weights_mean: float = 0.,
        weights_std: float = 1.,
        hidden_dim: int = 10,
        activation: nn.Module = nn.Prelu()
    ):
        self.weights_mean = weights_mean
        self.weights_std = weights_std
        self.hidden_dim = hidden_dim
        self.activation = activation


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
        layers = []

        layers.append(nn.modules.Linear(n_causes, self.hidden_dim))
        layers.append(self.activation)
        layers.append(nn.LayerNorm(self.hidden_dim)) # Avoid magnitude increase in causal direction
        layers.append(nn.modules.Linear(self.hidden_dim, 1))

        layers = nn.Sequential(*layers)
        layers.apply(self._weight_init)

        # Convert to torch.Tensor for forward pass
        data = X.astype('float32')
        data = from_numpy(data)

        # Forward pass
        effect = np.reshape(layers(data).data, (n_samples,))

        return effect.numpy()
    

    def _weight_init(self, model):
        """Random initialization of model weights.
        """
        if isinstance(model, nn.modules.Linear):
            nn.init.normal_(
                model.weight.data, mean=self.weights_mean, std=self.weights_std
            )



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
        effect = np.random.multivariate_normal(np.zeros(num_samples),covariance_matrix)
        return effect


# Base class for PostnonLinearModel invertible functions
class InvertibleFunction(metaclass=ABCMeta):
    
    @abstractmethod
    def forward(self, input: NDArray):
        raise NotImplementedError("InvertibleFunction is missing the required forward method.")
    
    def __call__(self, input: NDArray):
        return self.forward(input)

class Identity(InvertibleFunction):
    def forward(self, input: NDArray):
        return input
