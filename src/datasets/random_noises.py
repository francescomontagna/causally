# import GPy # TODO fix gpy install
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import PairwiseKernel
from abc import ABCMeta, abstractmethod


class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def sample(self, shape : tuple[int]) -> NDArray:
        raise NotImplementedError


class RandomNoiseDistribution(Distribution, metaclass=ABCMeta):
    """Base class for custom generation of random distributions.

    Samples from the random distribution are generated as nonlinear transformations
    of samples form a standard normal.
    """
    def sample(self, shape : tuple[int]) -> NDArray:
        # Sample from standard normal
        noise = np.random.normal(0, 1, shape)

        # Pass through the nonlinear mechanism
        noise = self.model(noise)
        return noise

    @abstractmethod
    def model(self, X: NDArray) -> NDArray:
        raise NotImplementedError
    

class GPNoise(RandomNoiseDistribution):
    """Sample non linear function from gaussian process.
    """
    def sampleGP(self, X : NDArray) -> NDArray:
        assert X.dim() == 2, f"Number of dimensions {X.dim()} different from 2."\
             "If input has 1 dimension, consider reshaping it with reshape(-1, 1)"
        self.rbf = PairwiseKernel(gamma=1, metric="rbf")
        covariance_matrix = self.rbf(X, X)
        sample = np.random.multivariate_normal(np.zeros(len(X)), covariance_matrix)
        return sample
    
    def model(self, X: NDArray) -> NDArray:
        return self.sampleGP(X.reshape(-1, 1))
    
# TODO: need to allow fixing weights interval! See Elias code
class TanHNoise(RandomNoiseDistribution):
    """Simple 1 layer NN with tanh activation
    """
    def model(self, X: NDArray) -> NDArray:
        # hidden_dimension?
        d = X.shape[0]
        n_hidden = 1000
        activation = nn.Tanh() 
        layer1 = nn.Linear(d, n_hidden)
        layer2 = nn.Linear(n_hidden, d)
        layer1.weight = nn.init.xavier_normal(layer1.weight)
        layer2.weight = nn.init.xavier_normal(layer2.weight)
        model = nn.Sequential(
            layer1,
            activation,
            layer2
        )
        return model(torch.from_numpy(X)).numpy()
    

# TODO: need to allow fixing weights interval! See Elias code
class EluNoise(RandomNoiseDistribution):
    """Simple 1 layer NN with ELU activation
    """
    def model(self, X: NDArray) -> NDArray:
        # hidden_dimension?
        d = X.shape[0]
        n_hidden = 1000
        activation = nn.ELU() 
        layer1 = nn.Linear(d, n_hidden)
        layer2 = nn.Linear(n_hidden, d)
        layer1.weight = nn.init.normal(layer1.weight)
        layer2.weight = nn.init.normal(layer2.weight)
        model = nn.Sequential(
            layer1,
            activation,
            layer2,
            activation
        )
        return model(torch.from_numpy(X)).numpy()
    

# Wrappers
class Normal(Distribution):
    """Wrapper for np.random.Generator.normal() sampler.
    """
    def __init__(
            self,
            loc : float,
            std: float
    ):
        self.loc = loc
        self.std = std

    def sample(self, shape: tuple[int]) -> NDArray:
        return np.random.normal(self.loc, self.std, shape)