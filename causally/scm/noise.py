import numpy as np
import torch
import torch.nn as nn
from typing import Union
from abc import ABCMeta, abstractmethod


# *** Abstract base classes *** #
class Distribution(metaclass=ABCMeta):
    """Base class to represent noise distributions."""

    @abstractmethod
    def sample(self, size: tuple[int]) -> np.array:
        raise NotImplementedError


class RandomNoiseDistribution(Distribution, metaclass=ABCMeta):
    """Base class for custom generation of random distributions.

    Samples from the random distribution are generated as nonlinear transformations
    of samples form a standard normal.
    """

    def __init__(self, standardize: bool = False) -> None:
        super().__init__()
        self.standardize = standardize

    def sample(self, size: tuple[int]) -> np.array:
        """Sample random noise of the required input size.

        Parameters
        ----------
        size: tuple[int]
            Shape of the sampled noise.

        Returns
        -------
        noise : np.array of shape (num_samples, num_nodes)
            Sample a random vector of noise terms.
        standardize : bool, default False
            If True, remove empirical mean and normalize deviation to one.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )

        # Sample from standard normal
        noise = np.random.normal(0, 1, size)

        # Pass through the nonlinear mechanism
        noise = self._forward(noise)
        if self.standardize:
            noise = self._standardize(noise)
        return noise

    def _standardize(self, noise: np.array) -> np.array:
        """Remove empirical mean and set empirical variance to one."""
        return (noise - noise.mean()) / noise.std()

    @abstractmethod
    def _forward(self, X: np.array) -> np.array:
        """Nonlinear transform of the input noise X."""
        raise NotImplementedError


# *** Wrappers of numpy distributions *** #
class Normal(Distribution):
    """Wrapper for np.random.normal() sampler.

    Parameters
    ----------
    loc: Union[float, np.array of floats], default 0
        The mean of the sample.
    std: Union[float, np.array of floats], default 1
        The standard deviation of the sample.
    """

    def __init__(
        self, loc: Union[float, np.array] = 0.0, std: Union[float, np.array] = 1.0
    ):
        super().__init__()
        self.loc = loc
        self.std = std

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from a Gaussian distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.normal(self.loc, self.std, size)


class Exponential(Distribution):
    r"""Wrapper for np.random.exponential() sampler.

    Parameters
    ----------
    scale: Union[float, np.array of floats], default 1
        The scale parameter :math:`\beta = \frac{1}{\lambda}`, must be non-negative.
    """

    def __init__(self, scale: Union[float, np.array] = 1.0):
        super().__init__()
        self.scale = scale

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from an exponential distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.exponential(self.scale, size)


class Uniform(Distribution):
    r"""Wrapper for np.random.uniform() sampler.

    Parameters
    ----------
    low: Union[float, np.array of floats], default 0
        Lower boundary of the output interval. All values generated will be greater than
        or equal to low. The default value is 0.
    high: Union[float, np.array of floats], default 1
        Upper boundary of the output interval. All values generated will be less than or
        equal to high. The high limit may be included in the returned array of floats due
        to floating-point rounding in the equation ``low + (high-low) * random_sample()``.
        The default value is 1.0.
    """

    def __init__(
        self, low: Union[float, np.array] = 0.0, high: Union[float, np.array] = 1.0
    ):
        super().__init__()
        self.low = low
        self.high = high

    def sample(self, size: tuple[int]) -> np.array:
        """Draw random samples from a uniform distribution.

        Parameters
        ----------
        size: tuple[int]
            Required shape of the random sample.
        """
        if len(size) != 2:
            ValueError(
                f"Expected number of input dimensions is 2, but were given {len(size)}."
            )
        return np.random.uniform(self.low, self.high, size)


# *** MLP transformation of standard normal *** #
class MLPNoise(RandomNoiseDistribution):
    """Neural network to generate samples as transformation of a standard normal.

    Generate a random variable with unknown distribution as a nonlinear transformation of
    a standard Gaussian. The transformation is parametrized by a simple neural network
    with one hidden layer and nonlinear activations.

    Parameters
    ----------
    hidden_dim: int, default 100
        Number of neurons in the hidden layer.
    activation: nn.Module, default ``nn.Sigmoid``
        The nonlinear activation function.
    bias: bool, default True
        If True, include the bias term.
    a_weight: float, default -3
        Lower bound for the value of the model weights.
    b_weight: float, default 3
        Upper bound for the value of the model weights.
    a_bias: float, default -1
        Lower bound for the value of the model bias terms.
    b_bias: float, default 1
        Upper bound for the value of the model bias terms.
    standardize: bool, default False
        If True, remove the empirical mean and variance from the transformed
        random variable.
    """

    def __init__(
        self,
        hidden_dim: int = 100,
        activation: nn.Module = nn.Sigmoid(),
        bias: bool = False,
        a_weight: float = -3.0,
        b_weight: float = 3.0,
        a_bias: float = -1.0,
        b_bias: float = 1.0,
        standardize: bool = False,
    ) -> None:
        super().__init__(standardize)
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bias = bias
        self.a_weight = a_weight
        self.b_weight = b_weight
        self.a_bias = a_bias
        self.b_bias = b_bias

    @torch.no_grad()
    def _forward(self, X: np.array) -> np.array:
        if X.ndim != 2:
            raise ValueError(
                f"Number of dimensions {X.ndim} different from 2."
                " If input has 1 dimension, consider reshaping it with reshape(-1, 1)"
            )
        num_features = X.shape[1]
        torch_X = torch.from_numpy(X)
        layer1 = self._init_params(
            nn.Linear(
                num_features, self.hidden_dim, bias=self.bias, dtype=torch_X.dtype
            )
        )
        layer2 = self._init_params(
            nn.Linear(
                self.hidden_dim, self.hidden_dim, bias=self.bias, dtype=torch_X.dtype
            )
        )
        layer3 = self._init_params(
            nn.Linear(
                self.hidden_dim, num_features, bias=self.bias, dtype=torch_X.dtype
            )
        )
        model = nn.Sequential(layer1, self.activation, layer2, self.activation, layer3)
        return model(torch_X).numpy()

    def _init_params(self, layer: nn.Module) -> None:
        nn.init.uniform_(layer.weight, a=self.a_weight, b=self.b_weight)
        if layer.bias is not None:
            nn.init.uniform_(layer.bias, a=self.a_bias, b=self.b_bias)
        return layer
