import random
import numpy as np
import warnings
from typing import Union, List, Dict, Callable
from causally.utils.graph import is_a_moral_collider, find_moral_colliders


# Collection of mixing classes for data generation under required assumptions


# * Confounded model utilities *
class _ConfoundedMixin:
    @staticmethod
    def confound_adjacency(adjacency: np.array, p_confounder=float):
        """Add latent common causes to the input adjacency matrix.

        Parameters
        ----------
        adjacency: np.array, shape (num_nodes x num_nodes)
            The adjacency matrix without latent confounders.
        p_confounder: float
            The probability of adding a latent common cause between a pair of nodes,
            parametrizing a Bernoulli random variable.

        Returns
        -------
        confounded_adj: np.array, shape (2*num_nodes, 2*num_nodes)
            The adjacency matrix with additional latent confounders.
        """
        num_nodes, _ = adjacency.shape

        # Generate the matrix of the latent confounders
        confounders_matrix = np.zeros((num_nodes, num_nodes))

        # Add confounding effects
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Add confounder with probability p_confounder
                confounded = np.random.binomial(n=1, p=p_confounder) == 1
                if confounded:
                    # Sample the confounder node from confounders_matrix nodes
                    confounder_node = random.choice(range(num_nodes))
                    confounders_matrix[confounder_node, i] = 1
                    confounders_matrix[confounder_node, j] = 1

        # Make confounders_matrix source nodes of the input adjacency.
        confounded_adj = np.vstack((confounders_matrix, adjacency))
        confounded_adj = np.hstack((np.zeros(confounded_adj.shape), confounded_adj))

        return confounded_adj

    @staticmethod
    def confound_dataset(X: np.array, n_confounders: int):
        """Remove latent confounders observations from the input dataset.

        Parameters
        ----------
        X: np.array, shape (num_samples, 2*num_nodes)
            The dataset with latent confoudners observations. The columns
            corresponding to confounders' observations are the first ``n_confounders``.
        n_confounders: int
            The number of latent confounders.

        Returns
        -------
        X: np.array, shape (num_samples, num_nodes)
            The dataset without latent confounders' observations' columns.
        """
        X = X[:, n_confounders:]

        return X


# * Measurement Error Utilities *
class _MeasurementErrorMixin:
    @staticmethod
    def add_measure_error(X: np.array, gamma: Union[float, List[float]]):
        r"""Add measurement error sampled from a Gaussian to the input dataset X.

        Parameters
        ----------
        X: np.array, shape (num_samples, num_nodes)
            The input dataset without measurement errors
        gamma: Union[float, List[float]]
            The inverse signal to noise ratio

        Returns
        -------
        X: np.array, shape (num_samples, num_nodes)
            The input dataset with measurement error sampled from a zero
            centered gaussian with variance

            .. math::
              \operatorname{Var}(\operatorname{error})} = \gamma*{\operatorname{Var}(\operatorname{signal})}
        """
        n_samples, n_nodes = X.shape
        if not isinstance(gamma, list):
            gamma = [gamma for _ in range(n_nodes)]

        X_std = np.std(X, axis=0)
        for node in range(n_nodes):
            error_std = np.sqrt(gamma[node]) * X_std[node]
            error_sample = error_std * np.random.standard_normal((n_samples,))
            X[:, node] += error_sample

        return X


# * Path Canceling Utilities *
class _UnfaithfulMixin:
    @staticmethod
    def generate_variable(
        X: np.array,
        node: int,
        node_noise: np.array,
        parents: np.array,
        unfaithful_triplets: List[tuple[int]],
        sample_mechanism: Callable,
        child_p1_effects: Dict[int, np.array],
    ):
        """Generate observations for the input node, accounting for path canceling effects.

        Parameters
        ----------
        X: np.array, shape (num_samples, num_nodes)
            The input dataset without measurement errors.
        node: int
            Index of the random variable to generate.
        node_noise: np.array, shape (num_samples, )
            Noise observations of the random variable to generate.
        parents: np.array
            List with the indices of the node's parents random variables,
            according to the unfaithful adjacency.
        unfaithful_triplets: List[tuple(int)]
            List of triplets involved in path canceling. Triplets are ordered
            according to their topological order, e.g. ``1->0<-2``, ``1->2``
            is uniquely represented by ``[1, 2, 0]`` toporder of the triplet
        sample_mechanism: Callable
            The function generating observations from the parents.
            It takes as argument the parents observations and the noise of the
            generated node.
        child_p1_effects: Dict[int, np.array]
            Dictionary with key: child index, value: sum of the contributions of
            p1 parents to the child observations.
            Given a triplet (p1, p2, child), path canceling is enforced
            by setting p1 effect on p2 and child equal in magnitude and opposite in sign.

        Returns
        -------
        total_effect: np.array, shape (num_samples)
            The observations of the node random variable.
        """
        p2_p1_parents = dict()  # key: p2, value: list of the p1 parents additive on p2
        child_linear_parents = (
            dict()
        )  # key: pchild, value: list of the p2 parents linear on child

        additive_effects = 0  # container of p1 additive effect on p2
        for triplet in unfaithful_triplets:
            p1, p2, child = triplet

            # Case 1: node is p2 in triplet
            # additive effect of f(p1) on p2 and child.
            if p2 == node:
                parents = np.delete(parents, np.where(parents == p1))
                p1_effect = sample_mechanism(
                    X[:, p1], child_noise=0
                )  # mechanism only, no noise
                child_p1_effects[child] = child_p1_effects.get(child, 0) + p1_effect
                l = p2_p1_parents.get(p2, list())
                if p1 not in l:  # p1 additive on p2 only once
                    l.append(p1)
                    p2_p1_parents[p2] = l
                    additive_effects += p1_effect

            # case 2: node is child in triplet
            elif child == node:
                parents = np.delete(parents, np.where(parents == p2))
                l = child_linear_parents.get(child, list())
                if p2 not in l:  # p2 linear effect only once
                    l.append(p2)
                    child_linear_parents[child] = l

        # Handle the case in which list of parents is empty due to cancelling
        total_effect = node_noise
        if len(parents) > 0:
            total_effect = sample_mechanism(
                parents=X[:, parents], child_noise=node_noise
            )

        # Additive p1 effect for each triplet where node == p2
        total_effect += additive_effects

        # -p2 effects for each triplet where node == child
        if child_linear_parents.get(node, None) is not None:
            linear_parents_effect = -sum([X[:, p] for p in child_linear_parents[node]])
            total_effect += linear_parents_effect

        # Additive p1 effect for each triplet where node == child: this cancels out with p2 effects
        if child_p1_effects.get(node, None) is not None:
            total_effect += child_p1_effects[node]
        return total_effect

    @staticmethod
    def unfaithful_adjacency(adjacency: np.array, p_unfaithful: float):
        """Make a copy of the input adjacency cancelling unfaithful edges.

        Parameters
        ----------
        adjacency: np.array, shape (num_nodes, num_nodes)
            The input adjacency matrix faithful to the data distribution.
        p_unfaithful: float
            Probability of  unfaitfhul conditional independence in the presence of
            a fully connected triplet.

        Return
        ------
        unfaithful_adj: np.array
            Transformed groundtruth adjacency matrix unfaithful to the data distribution.
            unfaithful to the graph
        unfaithful_triplets_toporder: List[tuple(int)]
            Represent moralized colliders by their topological order.
            E.g. ``1->0<-2``, ``1->2`` is uniquely represented by ``[1, 2, 0]`` toporder of the triplet
        """
        moral_colliders_toporder = find_moral_colliders(adjacency)
        unfaithful_adj = adjacency.copy()
        unfaithful_triplets_toporder = list()

        # For each child, if (p1, p2, c) lead to unfaithful cancelling of p1 -> c
        # then p2 -> c and p1 -> p2 can not be cancelled
        locked_edges = list()  # edges that can not be canceled

        for triplet in moral_colliders_toporder:
            p1, p2, child = triplet
            # Check that triplet is still a moral collider and p1 -> child can be canceled
            if is_a_moral_collider(unfaithful_adj, p1, p2, child) and not (
                (p1, child) in locked_edges
            ):
                if np.random.binomial(n=1, p=p_unfaithful) == 1:
                    unfaithful_adj[p1, child] = 0  # remove p1 -> c
                    unfaithful_triplets_toporder.append(
                        triplet
                    )  # store triplet involved in canceling
                    # Update fixed edges
                    if (p2, child) not in locked_edges:
                        locked_edges.append((p2, child))
                    if (p1, p2) not in locked_edges:
                        locked_edges.append((p1, p2))

        return unfaithful_adj, unfaithful_triplets_toporder


# * Time effects utilities *
class _AutoregressiveMixin:
    @staticmethod
    def add_time_lag(
        X: np.array, order: int, weight_a: float = -1.0, weight_b: float = 1.0
    ):
        """Add time effect to the input.

        The time effect is added as a liner combination of the previous ``order``
        observations of the variable ``X``.

        Parameters
        ----------
        X: np.array, shape (num_samples)
            Observations of a random node.
        order: int
            The number of time lags
        weight_a: float, default -1.
            Lower bound for the uniformly sampled coefficients of the linear time effect.
        weight_b: float, default 1.
            Upper bound for the uniformly sampled coefficients of the linear time effect.

        Returns
        -------
        X: np.array, shape (num_samples)
            Observations of a random node with addition of the time lagged effects.
        """
        if len(X) <= order:
            warnings.warn(
                "The autoregressive order is larger or equal than the number"
                " of samples of X. This would cause an IndexError. Reducing"
                f" order to len(X) - 1 = {len(X) - 1}"
            )
            order = len(X) - 1
        linear_coeffs = np.random.uniform(weight_a, weight_b, (order,))
        for t in range(order, len(X)):
            for k in range(order):
                X[t] += linear_coeffs[k] * X[t - k]

        return X
