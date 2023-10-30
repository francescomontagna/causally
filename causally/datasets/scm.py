import torch
import random
import numpy as np

from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod
from torch.distributions.distribution import Distribution
from typing import Union, Tuple, List

from datasets.causal_mechanisms import PredictionModel, LinearMechanism, InvertibleFunction
from datasets.random_graphs import GraphGenerator
from datasets.random_noises import RandomNoiseDistribution
from datasets.scm_properties import SCMProperty

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
    adjacency : NDArray of shape (num_nodes, num_nodes) 
        Matrix rerpesentation of the causal graph. A[i, j] = 1 encodes a directed
        edge from node i to node j, whereas zero entris correspond to no edges.
    noise : NDArray of shape (num_samples, num_nodes)
        Samples of the noise terms.
    self.misspecifications: Dict[str, SCMProperty]
        Dictionary of SCM properties violating common assumptions.
        The key is a string serving as identifier of the model violation (e.g. 'confounded')
        Valid misspecifications are measurement error, autoregressive effect,
        unfaithful path cancelling and presence of latent confounders.

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
        self.adjacency = graph_generator.get_random_graph()
        self.noise = noise_generator.sample((self.num_samples, graph_generator.num_nodes))
        self.misspecifications = dict()


    def _set_random_seed(self, seed: int):
        """Manually set the random seed. If the seed is None, then do nothing.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)


    def add_misspecificed_property(self, property: SCMProperty):
        """Specify misspecification to the SCM, e.g. presence of latent confounders.

        Parameters
        ----------
        property: SCMProperty
            Model misspecification
        """
        self.misspecifications[property.identifier] = property


    def sample(self) -> Tuple[NDArray, NDArray]:
        """
        Sample a dataset of observations.

        Parameters
        ----------
        violations: List[str]
            The list of violations to be modeled. Elements in the list are chosen from
            {'confounded', 'unfaithful', 'timino', 'measurement error'}, the order of
            the list doesn't matter.

        Returns
        -------
        X : NDArray
            Numpy array with the generated dataset.
        A: NDArray
            Numpy adjacency matrix representation of the causal graph.
        """
        # TODO: need to handle parameters of the violations!
        X = self.noise.copy()
        adjacency = self.adjacency.copy()

        # Misspecify the causal graphs
        # TODO: very bad, Python compiler does not know self.misspecifications["confounded"] is ConfoundedModel instance
        if "confounded" in list(self.misspecifications.keys()):
            adjacency = self.misspecifications["confounded"].confound_adjacency(adjacency)
        if "unfaithful" in list(self.misspecifications.keys()):
            adjacency, unfaithful_triplets_order = self.misspecifications["unfaithful"].unfaithful_adjacency(adjacency)

        # Generate the data
        num_nodes = len(adjacency)
        for i in range(num_nodes):
            parents = np.nonzero(adjacency[:,i])[0]
            if len(np.nonzero(adjacency[:,i])[0]) > 0:                
                X[:, i] = self._sample_mechanism(X[:,parents], self.noise[:, i])

                if "timino" in list(self.misspecifications.keys()):
                    X[:, i] = add_time_effect(X[:, i])


        # Misspecify the dataset
        if "measurement error" in  list(self.misspecifications.keys()):
            self.misspecifications["measurement error"].add_measure_error(X)
        if "confounded" in list(self.misspecifications.keys()):
            d, _ = self.adjacency.shape
            self.misspecifications["confounded"].confound_dataset(X, n_confounders=d)
        if "unfaithful" in list(self.misspecifications.keys()):
            self.misspecifications["unfaithful"].unfaithful_dataset(
                X, self.noise, unfaithful_triplets_order
            )

        return X, self.adjacency
    

    @abstractmethod
    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        raise NotImplementedError()



# * ANM *
class AdditiveNoiseModel(BaseStructuralCausalModel):
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
        super().__init__(num_samples, graph_generator, noise_generator, seed)
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        return super()._sample_mechanism(parents, child_noise)



class PostNonlinearModel(AdditiveNoiseModel):
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
        super().__init__(num_samples, graph_generator, noise_generator, causal_mechanism, seed)
        
        self.causal_mechanism = causal_mechanism
        self.invertible_function = invertible_function

    
    def add_misspecificed_property(self, property: SCMProperty):
        if property.identifier == "unfaithful":
            raise ValueError("The PostNonlinear model does not support faithfulness violation.")
        else:
            super().add_misspecificed_property(property)

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
        anm_effect = super()._sample_mechanism(parents, child_noise)

        # Apply the invertible postnonlinearity
        effect = self.invertible_function(anm_effect)
        return effect


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
class MixedLinearNonlinearModel(AdditiveNoiseModel):
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
        linear_fraction = 0.5,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, nonlinear_mechanism, seed)
        self._linear_mechanism = linear_mechanism
        self.linear_fraction = linear_fraction

        
    def _sample_mechanism(self, parents: NDArray, child_noise: NDArray) -> NDArray:
        # Randomly sample the type of mechanism: linear or nonlinear
        linear = np.random.binomial(n=1, p=self.linear_fraction) == 1

        if linear:
            return self._linear_mechanism.predict(parents) + child_noise
        else:
            return super()._sample_mechanism(parents, child_noise)
