import torch
import random
import numpy as np

from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import LinearRegression
from torch.distributions.distribution import Distribution
from typing import Union, Callable, Tuple

from datasets.causal_mechanisms import PredictionModel, Identity
from datasets.random_graphs import GraphGenerator
from datasets.random_noises import RandomNoiseDistribution


class BaseStructuralCausalModel(metaclass=ABCMeta):
    """Base class for synthetic data generation.

    DataGenrator supports generation of linear, nonlinear ANM and PNL
    structural causal model with several distribution of the noise terms.
    The adjacency matrix are sampled according to one model between
    Erdos-Rényi, Scale-Free, Gaussian Random Process.

    Parameters
    ----------
    num_samples : int:
        Number of samples in the dataset
    graph_generator : GraphGenerator
        Random graph generator implementing the 'get_random_graph' method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    seed : int
        Seed for reproducibility

    Attributes
    ----------
    num_nodes : int
        Number of nodes in the causal graph.
    adjacency : NDArray of shape (num_nodes, num_nodes) 
        Matrix rerpesentation of the causal graph. A[i, j] = 1 encodes a directed
        edge from node i to node j, whereas zero entris correspond to no edges.
    noise : NDArray of shape (num_samples, num_nodes)
        Samples of the noise terms.
    """
    def __init__(
        self, 
        num_samples : int, 
        graph_generator : GraphGenerator,
        noise_generator :  Union[RandomNoiseDistribution, Distribution],
        seed : int
    ):
        self._set_random_seed(seed)

        self.num_samples = num_samples
        self.num_nodes = graph_generator.num_nodes
        self.adjacency = graph_generator.get_random_graph()
        self.noise = noise_generator.sample((self.num_samples, self.num_nodes))
        assert not np.isnan(self.noise.sum()), "Nan value detected in the noise matrix"


    def _set_random_seed(self, seed: int):
        """Manually set the random seed.

        # NOTE: I am not sure whether this is a good way.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)


    def sample(self) -> Tuple[NDArray, NDArray]:
        """
        Sample a dataset of observations.

        Returns:
        X : NDArray
            Numpy array with the generated dataset.
        """
        X = self.noise.copy()

        for i in range(self.num_nodes):
            parents = np.nonzero(self.adjacency[:,i])[0]
            if len(np.nonzero(self.adjacency[:,i])[0]) > 0:                
                X[:, i] = self._sample_mechanism(X[:,parents], self.noise[:, i])
                # TODO: how to handle mixed mechanisms?

        return X, self.adjacency
    

    @abstractmethod
    def _sample_mechanism(self, X: NDArray, noise: NDArray) -> NDArray:
        raise NotImplementedError()



class LinearModel(BaseStructuralCausalModel):
    """Class for data generation from a linear structural causal model.
    
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
        num_samples : int, 
        graph_generator : GraphGenerator,
        noise_generator :  Union[RandomNoiseDistribution, Distribution],
        min_weight: float = -1,
        max_weight: float = 1,
        min_abs_weight = 0.05,
        seed = 42,
    ):
        super().__init__(num_samples, graph_generator, noise_generator, seed)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_abs_weight = min_abs_weight

        
    def _sample_mechanism(self, parents: NDArray, noise: NDArray) -> NDArray:
        """Define a linear combination of the columns of 'parents'.

        Use LinearRegression from sklearn, setting the regression coefficient 
        with uniform proability distirbution between self.min_weight and self.max_weight.
        To avoid quasi-unfaithful mechanism, 

        Parameters
        ----------
        parents: NDArray of shape (num_samples, num_parents)
            Tensor with the observations of the parents of the variable to be generated.
        noise: NDArray of shape (num_samples, )
            Vector of the additive noise term.

        Returns
        -------
        X : NDArray
            The variable generation as linear combination of the parents plus the noise column.
        """
        _, n_covariates = parents.shape
        linear_reg = LinearRegression()
        linear_reg.coef_ = np.random.uniform(self.min_weight, self.max_weight, n_covariates) 
        
        # Reject ~0 coefficients
        for i in range(n_covariates):
            while (abs(linear_reg.coef_[i]) < self.min_abs_weight):
                linear_reg.coef_[i] = np.random.uniform(self.min_weight, self.max_weight, 1)
            
        linear_reg.intercept_ = 0
        X = linear_reg.predict(parents) + noise
        return X
    


class PostNonlinearModel(BaseStructuralCausalModel):
    """Class for data generation from a postnonlinear model.
    
    Parameters
    ----------
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a `predict` method.
    invertible_function: Callable
        Invertible post-nonlinearity. Invertibility required for identifiability.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: RandomNoiseDistribution | Distribution,
        causal_mechanism : PredictionModel,
        invertible_function : Callable,
        seed=42
    ):
        super().__init__(num_samples, graph_generator, noise_generator, seed)
        
        self.causal_mechanism = causal_mechanism
        self.invertible_function = invertible_function

    def _sample_mechanism(self, X: NDArray, noise: NDArray) -> NDArray:
        """Generate effect given the direct parents `X` and the vector of noise terms.

        Parameters
        ----------
        X: NDArray of shape (n_samples, n_parents)
            Multidimensional array of the parents observataions.
        noise: NDArray of shape(n_samples,)
            Vector of random noise observations.

        Returns
        -------
        effect: NDArray of shape (n_samples,)
            Vector of the effect observations generated given the parents and the noise.
        """
        anm_effect = self.causal_mechanism.predict(X) + noise
        assert not np.isnan(anm_effect.sum()), "Nan value in ANM mechanism output"

        # TODO: add all violations handling here!

        # Apply the invertible postnonlinearity
        effect = self.invertible_function(anm_effect)
        return effect


class AdditiveNoiseModel(PostNonlinearModel):
    """Class for data generation from a nonlinear additive noise model.

    The additive noise model is generated as a postnonlinear model with
    the invertible post-nonlinear function being the identity.
    
    Parameters
    ----------
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a `predict` method.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: RandomNoiseDistribution | Distribution,
        causal_mechanism : PredictionModel,
        seed=42
    ):
        invertible_function = Identity()
        super().__init__(num_samples, graph_generator, noise_generator, causal_mechanism, invertible_function, seed)
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, X: NDArray, noise: NDArray) -> NDArray:
        return super()._sample_mechanism(X, noise)