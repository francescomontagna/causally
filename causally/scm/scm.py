import torch
import random
import numpy as np

from abc import ABCMeta, abstractmethod
from typing import Union, Tuple

from causally.scm.causal_mechanism import (
    PredictionModel,
    LinearMechanism,
    InvertibleFunction,
)
from causally.graph.random_graph import GraphGenerator
from causally.scm.noise import RandomNoiseDistribution, Distribution
from causally.scm.scm_property import (
    _ConfoundedMixin,
    _MeasurementErrorMixin,
    _UnfaithfulMixin,
    _AutoregressiveMixin,
)
from causally.scm.context import SCMContext
from causally.utils.graph import topological_order


# * Base SCM abstract class *
class BaseStructuralCausalModel(metaclass=ABCMeta):
    """Base abstract class for synthetic data generation.

    Classes inheriting from ``BaseStructuralCausalModel`` must implement the method
    ``_sample_mechanism``, specifying how to sample effects from parents and
    noise terms observtations.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    seed: int, default None
        Seed for reproducibility. If ``None``, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Distribution,
        scm_context: SCMContext = None,
        seed: int = None,
    ):
        self._set_random_seed(seed)

        self.num_samples = num_samples
        self.adjacency = graph_generator.get_random_graph()
        self.noise_generator = noise_generator

        if scm_context is not None:
            self.scm_context = scm_context
        else:
            self.scm_context = SCMContext()

    def _set_random_seed(self, seed: int):
        """Manually set the random seed. If the seed is None, then do nothing."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)

    def sample(self) -> Tuple[np.array, np.array]:
        """
        Sample a dataset of observations.

        Returns
        -------
        X: np.array, shape (num_samples, num_nodes)
            Numpy array with the generated dataset.
        A: np.array, shape (num_nodes, num_nodes)
            Numpy adjacency matrix representation of the causal graph.
            The presence of a directed edge from node ``i`` to node ``j``
            is denoted by ``A[i, j] = 1``. Absence of the edge is denote by
            ``A[i, j] = 0``.
        """
        adjacency = self.adjacency.copy()

        # Pre-process for graph misspecification
        # NOTE: Unfaithful before confounded to avoid cancelling on confounders matrix
        if "unfaithful" in list(self.scm_context.assumptions):
            child_p1_effects = (
                dict()
            )  # key: child, value: sum of p1 values additive on child
            p_unfaithful = self.scm_context.p_unfaithful
            (
                adjacency,
                unfaithful_triplets_order,
            ) = _UnfaithfulMixin.unfaithful_adjacency(adjacency, p_unfaithful)
            self.scm_context.unfaithful_adjacency = adjacency
        if "confounded" in list(self.scm_context.assumptions):
            p_confounder = self.scm_context.p_confounder
            adjacency = _ConfoundedMixin.confound_adjacency(adjacency, p_confounder)
            self.scm_context.confounded_adjacency = adjacency

        # Sample the noise
        noise = self.noise_generator.sample((self.num_samples, len(adjacency)))
        X = noise.copy()

        # Generate the data starting from source nodes
        graph_order = topological_order(self.adjacency)
        for node in graph_order:
            parents = np.nonzero(adjacency[:, node])[0]
            if len(parents) > 0:
                # Unfaithful path canceling
                if "unfaithful" in list(self.scm_context.assumptions):
                    X[:, node] = _UnfaithfulMixin.generate_variable(
                        X,
                        node,
                        noise[:, node],
                        parents,
                        unfaithful_triplets_order,
                        self._sample_mechanism,
                        child_p1_effects,
                    )

                else:
                    X[:, node] = self._sample_mechanism(X[:, parents], noise[:, node])

            # Autoregressive effect
            if "autoregressive" in list(self.scm_context.assumptions):
                order = self.scm_context.autoregressive_order
                X[:, node] = _AutoregressiveMixin.add_time_lag(X[:, node], order)

        # Post-process for data misspecification
        if "measurement error" in list(self.scm_context.assumptions):
            gamma = self.scm_context.measure_error_gamma
            X = _MeasurementErrorMixin.add_measure_error(X, gamma)
        if "confounded" in list(self.scm_context.assumptions):
            d, _ = self.adjacency.shape
            X = _ConfoundedMixin.confound_dataset(X, n_confounders=d)

        return X, self.adjacency

    @abstractmethod
    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        raise NotImplementedError()


# # * ANM *
class AdditiveNoiseModel(BaseStructuralCausalModel):
    """Class for data generation from a nonlinear additive noise model.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinear causal mechanism.
        It must be an instance of a class inheriting from
        ``causally.scm.causal_mechanism.PredictionModel``, implementing
        the ``predict`` method.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    seed: int, default None
        Seed for reproducibility. If ``None``, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        causal_mechanism: PredictionModel,
        scm_context: SCMContext = None,
        seed: int = None,
    ):
        super().__init__(
            num_samples, graph_generator, noise_generator, scm_context, seed
        )
        self.causal_mechanism = causal_mechanism

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        effect = self.causal_mechanism.predict(parents) + child_noise
        return effect


class PostNonlinearModel(AdditiveNoiseModel):
    """Class for data generation from a postnonlinear model.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    causal_mechanism: PredictionModel
        Object for the generation of the nonlinear causal mechanism.
        It must be an instance of a class inheriting from
        ``causally.scm.causal_mechanism.PredictionModel``, implementing
        the ``predict`` method.
    invertible_function: InvertibleFunction
        Invertible post-nonlinearity. Invertibility is required for identifiability.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    seed: int, default None
        Seed for reproducibility. If ``None``, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        causal_mechanism: PredictionModel,
        invertible_function: InvertibleFunction,
        scm_context: SCMContext = None,
        seed: int = None,
    ):
        super().__init__(
            num_samples,
            graph_generator,
            noise_generator,
            causal_mechanism,
            scm_context,
            seed,
        )
        self._check_assumptions()  # Unfaithfulness and autoregressive not in PNL assumptions.
        self.causal_mechanism = causal_mechanism
        self.invertible_function = invertible_function

    def _check_assumptions(self):
        if "unfaithful" in self.scm_context.assumptions:
            raise ValueError(
                "The PostNonlinear model does not support faithfulness violation."
                + " Provide an ``SCMContext`` without the assumption of unfaithfulness."
            )
        elif "autoregressive" in self.scm_context.assumptions:
            raise ValueError(
                "The PostNonlinear model does not support autoregressive effects violation."
                + " Provide an ``SCMContext`` without the assumption of autoregressive effects."
            )

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        """Generate effect given the direct parents ``X`` and the vector of noise terms.

        Parameters
        ----------
        parents: np.array, shape (n_samples, n_parents)
            Multidimensional array of the parents observataions.
        child_noise: np.array, shape(n_samples,)
            Vector of random noise observations of the generated effect.

        Returns
        -------
        effect: np.array, shape (n_samples,)
            Vector of the effect observations generated given the parents and the noise.
        """
        anm_effect = super()._sample_mechanism(parents, child_noise)

        # Apply the invertible postnonlinearity
        effect = self.invertible_function(anm_effect)
        return effect


# * Linear SCM *
class LinearModel(AdditiveNoiseModel):
    """Class for data generation from a linear structural causal model.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    min_weight: float, default is -1
        Minimum value of causal mechanisms weights
    max_weight: float, default is 1
        Maximum value of causal mechanisms weights
    min_abs_weight: float, default is 0.05
        Minimum value of the absolute value of any causal mechanism weight.
        Low value of min_abs_weight potentially lead to lambda-unfaithful distributions.
    seed: int, default None
        Seed for reproducibility. If None, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        scm_context: SCMContext = None,
        min_weight: float = -1,
        max_weight: float = 1,
        min_abs_weight=0.05,
        seed: int = None,
    ):
        causal_mechanism = LinearMechanism(min_weight, max_weight, min_abs_weight)
        super().__init__(
            num_samples,
            graph_generator,
            noise_generator,
            causal_mechanism,
            scm_context,
            seed,
        )


# * SCM with mixed linear-nonlinear mechanisms *
class MixedLinearNonlinearModel(AdditiveNoiseModel):
    """Class for data generation with mixed linear and nonlinear mechanisms.

    Parameters
    ----------
    num_samples: int
        Number of samples in the dataset.
    graph_generator: GraphGenerator
        Random graph generator implementing the ``get_random_graph`` method.
    noise_generator:  Distribution
        Sampler of the noise random variables. It must be an instance of
        a class inheriting from ``causally.scm.noise.Distribution``, implementing
        the ``sample`` method.
    linear_mechanism: LinearMechanism
        LinearMechanism instance for the generation of effects as a
        linear combination of the causes.
    nonlinear_mechanism: PredictionModel
        Object for the generation of the nonlinar causal mechanism.
        The object passed as argument must implement the PredictionModel abstract class,
        and have a ``predict`` method.
    scm_context: SCMContext, default None
        ``SCMContext`` object specifying the modeling assumptions of the SCM.
        If ``None`` this is equivalent to an ``SCMContext`` object with no
        assumption specified.
    linear_fraction: float, default 0.5
        The fraction of linear structural equations over the total number of variables.
        E.g. for ``linear_fraction = 0.5`` data are generated from an SCM with half of the
        structural equations with linear causal mechanisms. Be aware that causal relations
        may not be identifiable, e.g. in the case of additive Gaussian noise terms.
    seed: int, default None
        Seed for reproducibility. If None, then the random seed is not set.
    """

    def __init__(
        self,
        num_samples: int,
        graph_generator: GraphGenerator,
        noise_generator: Union[RandomNoiseDistribution, Distribution],
        linear_mechanism: PredictionModel,
        nonlinear_mechanism: PredictionModel,
        scm_context: SCMContext = None,
        linear_fraction=0.5,
        seed: int = None,
    ):
        super().__init__(
            num_samples,
            graph_generator,
            noise_generator,
            nonlinear_mechanism,
            scm_context,
            seed,
        )
        self._linear_mechanism = linear_mechanism
        self.linear_fraction = linear_fraction

    def _sample_mechanism(self, parents: np.array, child_noise: np.array) -> np.array:
        # Randomly sample the type of mechanism: linear or nonlinear
        linear = np.random.binomial(n=1, p=self.linear_fraction) == 1

        if linear:
            return self._linear_mechanism.predict(parents) + child_noise
        else:
            return super()._sample_mechanism(parents, child_noise)
