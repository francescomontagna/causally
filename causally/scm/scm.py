import torch
import random
import numpy as np

from abc import ABCMeta, abstractmethod
from torch.distributions.distribution import Distribution
from typing import Union, Tuple, Dict

from causally.scm.causal_mechanism import PredictionModel, LinearMechanism, InvertibleFunction
from causally.graph.random_graph import GraphGenerator
from causally.scm.noise import RandomNoiseDistribution
from causally.scm.scm_property import (
    _ConfoundedMixin, _MeasurementErrorMixin, _UnfaithfulMixin, _AutoregressiveMixin
)
from causally.scm.context import Context
from causally.utils.graph import topological_order

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
        Random graph generator implementing the ``get_random_graph`` method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.
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
        self.noise_generator = noise_generator
        self.assumptions: Dict[str, Context] = dict()


    def _set_random_seed(self, seed: int):
        """Manually set the random seed. If the seed is None, then do nothing.
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)


    def make_assumption(self, assumption: Context):
        """Make an assumption on SCM, e.g. presence of latent confounders.

        Assumptions are defined as instance of Context, and define a
        modelling assumption on the SCM. 

        Parameters
        ----------
        assumption: Context
            The class containig information on the SCM assumption to add.
            E.g. ``assumption = ConfoundedModel(p_confounder=0.2)`` defines 
            a structural causal model where each pair has a latent common
            cause with probability 0.2.
        """
        self.assumptions[assumption.identifier] = assumption


    def sample(self) -> Tuple[np.array, np.array]:
        """
        Sample a dataset of observations.

        Returns
        -------
        X : np.array
            Numpy array with the generated dataset.
        A: np.array
            Numpy adjacency matrix representation of the causal graph.
        """
        adjacency = self.adjacency.copy()

        # Pre-process: graph misspecification

        # TODO: add constraints 
        # i.e. unfaithful must be before confounded to avoid cancelling occurring on confounders matrix
        if "unfaithful" in list(self.assumptions.keys()):
            p_unfaithful = self.assumptions["unfaithful"]
            adjacency, unfaithful_triplets_order = _UnfaithfulMixin.unfaithful_adjacency(
                adjacency, p_unfaithful
            )
        if "confounded" in list(self.assumptions.keys()):
            p_confounded = self.assumptions["unfaithful"]
            adjacency = _ConfoundedMixin.confound_adjacency(adjacency, p_confounded)

        # Sample the noise
        noise = self.noise_generator.sample((self.num_samples, len(adjacency)))
        X = noise.copy()

        # Generate the data starting from source nodes
        for i in topological_order(adjacency):
            parents = np.nonzero(adjacency[:,i])[0]
            if len(np.nonzero(adjacency[:,i])[0]) > 0:    
                X[:, i] = self._sample_mechanism(X[:,parents], noise[:, i])

                # Autoregressive effect          
                if "autoregressive" in list(self.assumptions.keys()):
                    order = self.assumptions["autoregressive"]
                    X[:, i] = _AutoregressiveMixin.add_time_lag(X[:, i], order)


        # Post-process: data misspecification
        if "measurement error" in  list(self.assumptions.keys()):
            gamma = self.assumptions["measurement error"]
            X = _MeasurementErrorMixin.add_measure_error(X, gamma)
        if "confounded" in list(self.assumptions.keys()):
            d, _ = self.adjacency.shape
            X = _ConfoundedMixin.confound_dataset(X, n_confounders=d)
        if "unfaithful" in list(self.assumptions.keys()):
            X = _UnfaithfulMixin.unfaithful_dataset(
                X, noise, unfaithful_triplets_order
            )

        return X, self.adjacency
    

    @abstractmethod
    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        raise NotImplementedError()



# # * ANM *
class AdditiveNoiseModel(BaseStructuralCausalModel):
    """Class for data generation from a nonlinear additive noise model.

    The additive noise model is generated as a postnonlinear model with
    the invertible post-nonlinear function being the identity.
    
    Parameters
    ----------
    num_samples : int:
        Number of samples in the dataset
    graph_generator : GraphGenerator
        Random graph generator implementing the 'get_random_graph' method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        causal_mechanism : PredictionModel,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, seed)
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        effect = self.causal_mechanism.predict(parents) + child_noise
        return effect



class PostNonlinearModel(AdditiveNoiseModel):
    """Class for data generation from a postnonlinear model.
    
    Parameters
    ----------
    num_samples : int:
        Number of samples in the dataset
    graph_generator : GraphGenerator
        Random graph generator implementing the 'get_random_graph' method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    invertible_function: InvertibleFunction
        Invertible post-nonlinearity. Invertibility required for identifiability.
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        causal_mechanism : PredictionModel,
        invertible_function : InvertibleFunction,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, causal_mechanism, seed)
        
        self.causal_mechanism = causal_mechanism
        self.invertible_function = invertible_function

    
    def make_assumption(self, assumption: Context):
        if assumption.identifier == "unfaithful":
            raise ValueError("The PostNonlinear model does not support faithfulness violation.")
        elif assumption.identifier == "autoregressive":
            raise ValueError("The PostNonlinear model does not support autoregressive effects violation.")
        else:
            super().make_assumption(assumption)

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        """Generate effect given the direct parents ``X`` and the vector of noise terms.

        Parameters
        ----------
        parents: np.array of shape (n_samples, n_parents)
            Multidimensional array of the parents observataions.
        child_noise: np.array of shape(n_samples,)
            Vector of random noise observations of the generated effect.

        Returns
        -------
        effect: np.array of shape (n_samples,)
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
    num_samples : int:
        Number of samples in the dataset
    graph_generator : GraphGenerator
        Random graph generator implementing the 'get_random_graph' method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    min_weight: float, default is -1
        Minimum value of causal mechanisms weights
    max_weight: float, default is 1
        Maximum value of causal mechanisms weights
    min_abs_weight: float, default is 0.05
        Minimum value of the absolute value of any causal mechanism weight.
        Low value of min_abs_weight potentially lead to lambda-unfaithful distributions.
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.
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
    num_samples : int:
        Number of samples in the dataset
    graph_generator : GraphGenerator
        Random graph generator implementing the 'get_random_graph' method. 
    noise_generator :  Union[RandomNoiseDistribution, Distribution]
        Sampler of the noise terms. It can be either a custom implementation of 
        the base class RandomNoiseDistribution, or a torch Distribution.
    linear_mechanism: PredictionModel
        Object for the generation of the linear causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    nonlinear_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    invertible_function: InvertibleFunction
        Invertible post-nonlinearity (not applied to the linear mechanism).
        Invertibility required for identifiability.
    linear_fraction: float, default 0.5
        The fraction of linear mechanisms over the total number of causal relationships.
        E.g. for ``linear_fraction = 0.5`` data are generated from an SCM with half of the
        structural equations with linear causal mechanisms. Be aware that linear mechanisms
        are not identifiable in case of additive noise term.
    seed : int, default None
        Seed for reproducibility. If None, then random seed not set.
    """
    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        linear_mechanism : PredictionModel,
        nonlinear_mechanism : PredictionModel,
        linear_fraction = 0.5,
        seed: int=None
    ):
        super().__init__(num_samples, graph_generator, noise_generator, nonlinear_mechanism, seed)
        self._linear_mechanism = linear_mechanism
        self.linear_fraction = linear_fraction

        
    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        # Randomly sample the type of mechanism: linear or nonlinear
        linear = np.random.binomial(n=1, p=self.linear_fraction) == 1

        if linear:
            return self._linear_mechanism.predict(parents) + child_noise
        else:
            return super()._sample_mechanism(parents, child_noise)
