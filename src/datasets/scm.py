import torch
import random
import numpy as np

from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from torch.distributions.distribution import Distribution
from typing import Union, Tuple

from datasets.causal_mechanisms import PredictionModel, LinearMechanism, InvertibleFunction
from datasets.random_graphs import GraphGenerator
from datasets.random_noises import RandomNoiseDistribution

# * Base SCM abstract class *
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
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.

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
        seed: int=None
    ):
        self._set_random_seed(seed)

        self.num_samples = num_samples
        self.num_nodes = graph_generator.num_nodes
        self.adjacency = graph_generator.get_random_graph()
        self.noise = noise_generator.sample((self.num_samples, self.num_nodes))


    def _set_random_seed(self, seed: int):
        """Manually set the random seed. If the seed is None, then do nothing.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)


    def sample(self) -> Tuple[NDArray, NDArray]:
        """
        Sample a dataset of observations.

        Returns:
        X : NDArray
            Numpy array with the generated dataset.
        A: NDArray
            Numpy adjacency matrix representation of the causal graph.
        """
        X = self.noise.copy()

        for i in range(self.num_nodes):
            parents = np.nonzero(self.adjacency[:,i])[0]
            if len(np.nonzero(self.adjacency[:,i])[0]) > 0:                
                X[:, i] = self._sample_mechanism(X[:,parents], self.noise[:, i])

        return X, self.adjacency
    

    @abstractmethod
    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        raise NotImplementedError()



# * PNL *
class PostNonlinearModel(BaseStructuralCausalModel):
    """Class for data generation from a postnonlinear model.
    
    Parameters
    ----------
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a `predict` method.
    invertible_function: InvertibleFunction
        Invertible post-nonlinearity. Invertibility required for identifiability.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: RandomNoiseDistribution | Distribution,
        causal_mechanism : PredictionModel,
        invertible_function : InvertibleFunction,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, seed)
        
        self.causal_mechanism = causal_mechanism
        self.invertible_function = invertible_function

    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        """Generate effect given the direct parents `X` and the vector of noise terms.

        Parameters
        ----------
        parents: NDArray of shape (n_samples, n_parents)
            Multidimensional array of the parents observataions.
        child_noise: NDArray of shape(n_samples,)
            Vector of random noise observations of the generated effect.

        Returns
        -------
        effect: NDArray of shape (n_samples,)
            Vector of the effect observations generated given the parents and the noise.
        """
        anm_effect = self.causal_mechanism.predict(parents) + child_noise

        # TODO: add all violations handling here!

        # Apply the invertible postnonlinearity
        effect = self.invertible_function(anm_effect)
        return effect



# * ANM *
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
        seed: int=None
    ):
        invertible_function = lambda x: x # identity
        super().__init__(num_samples, graph_generator, noise_generator, causal_mechanism, invertible_function, seed)
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        return super()._sample_mechanism(parents, child_noise)



# * Linear SCM *
class LinearModel(AdditiveNoiseModel):
    """Class for data generation from a linear structural causal model.

    The LinearModel is defined as an additive noise model with linear mechanisms.
    
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
        seed: int=None
    ):
        causal_mechanism = LinearMechanism(min_weight, max_weight, min_abs_weight)
        super().__init__(num_samples, graph_generator, noise_generator, causal_mechanism, seed)




# * SCM with mixed linear-nonlinear mechanisms *
class MixedLinearNonlinearModel(PostNonlinearModel):
    """Class for data generation with mixed linear and nonlinear mechanisms.

    Parameters
    ----------
    linear_mechanism: PredictionModel
        Object for the generation of the linear causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a `predict` method.
    nonlinear_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a `predict` method.
    invertible_function: InvertibleFunction
        Invertible post-nonlinearity (not applied to the linear mechanism).
        Invertibility required for identifiability.
    linear_fraction: float, default 0.5
        The fraction of linear mechanisms over the total number of causal relationships.
        E.g. for `linear_fraction = 0.5` data are generated from an SCM with half of the
        structural equations with linear causal mechanisms. Be aware that linear mechanisms
        are not identifiable in case of additive noise term.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: RandomNoiseDistribution | Distribution,
        linear_mechanism : PredictionModel,
        nonlinear_mechanism : PredictionModel,
        invertible_function : InvertibleFunction,
        linear_fraction = 0.5,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, nonlinear_mechanism, invertible_function, seed)
        self._linear_mechanism = linear_mechanism
        self.linear_fraction = linear_fraction

        
    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        # Randomly sample the type of mechanism: linear or nonlinear
        linear = np.random.binomial(n=1, p=self.linear_fraction) == 1

        if linear:
            return self._linear_mechanism.predict(parents) + child_noise
        else:
            return super()._sample_mechanism(parents, child_noise)
