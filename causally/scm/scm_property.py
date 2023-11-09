import random
import numpy as np
import warnings

from abc import ABCMeta
from typing import Union, List


# * Base property class. Just for type hinting *
class SCMProperty(metaclass=ABCMeta):
    def __init__(self, identifier: str) -> None:
        self.identifier = identifier


# * Latent Confouders Utilities *
class ConfoundedModel(SCMProperty):
    """Utility functions for SCM generation under confounding effects.

    Parameters
    ----------
    p_confounder: float, default 0.2
        The probability of adding a latent common cause between a pair of nodes,
        sampled as a Bernoulli random variable.
    """
    def __init__(self, p_confounder: float = 0.2):
        super().__init__(identifier="confounded")
        self.p_confounder = p_confounder


    def confound_adjacency(self, adjacency: np.array):
        """Add latent common causes to the input adjacency matrix.

        Parameters
        ----------
        adjacency: np.array of shape (num_nodes x num_nodes)
            The adjacency matrix without latent confounders.

        Returns
        -------
        confounded_adj: np.array of shape (2*num_nodes, 2*num_nodes)
            The adjacency matrix with additional latent confounders.
        """
        num_nodes, _ = adjacency.shape


        # Generate the matrix of the latent confounders
        confounders_matrix = np.zeros((num_nodes, num_nodes))
        
        # Add confounding effects
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                # Add confounder with probability self.p_confounder.
                confounded = np.random.binomial(n=1, p=self.p_confounder) == 1
                if confounded:
                    # Sample the confounder node from confounders_matrix nodes
                    confounder_node = random.choice(range(num_nodes))
                    confounders_matrix[confounder_node, i] = 1
                    confounders_matrix[confounder_node, j] = 1
        
        # Make confounders_matrix source nodes of the input adjacency.
        confounded_adj = np.vstack((confounders_matrix, adjacency))
        confounded_adj = np.hstack((np.zeros(confounded_adj.shape), confounded_adj))
        
        # TODO: test if confounded_adj is uppertriangular. 
        return confounded_adj
    

    def confound_dataset(self, X: np.array, n_confounders: int):
        """Remove latent confounders observations from the input dataset.

        Parameters
        ----------
        X: np.array of shape (num_samples, 2*num_nodes)
            The dataset with latent confoudners observations. The columns
            corresponding to confounders' observations are the first ``n_confounders``.
        n_confounders: int
            The number of latent confounders.

        Returns
        -------
        X: np.array of shape (num_samples, num_nodes)
            The dataset without latent confounders' observations' columns.
        """
        X = X[:, n_confounders:]

        # TODO: unit test output shape
        return X


# * Measurement Error Utilities *
class MeasurementErrorModel(SCMProperty):
    r"""Utility functions for SCM generation with measurement error.

    Parameters
    ----------
    gamma: Union[float, List[float]] 
        The inverse signal to noise ratio

        .. math::

        \frac{\operatorname{Var}(\operatorname{error})}{\operatorname{Var}(\operatorname{signal})}

        parametrizing the variance of the measurement error proportionally to the
        variance of the signal. If a single float is provided, then gamma is the
        same for each column of the data matrix. Else, gamma is a vector of shape
        (num_nodes, ).
    """
    def __init__(self, gamma:Union[float, List[float]]) -> None:
        super().__init__(identifier="measurement error")
        if not gamma > 0 and gamma <= 1:
            raise ValueError("Signal to noise ratio outside of  (0, 1] interval")
        self.gamma = gamma


    def add_measure_error(self, X: np.array):
        r"""Add measurement error to the input dataset X.

        X: np.array of shape (num_samples, num_nodes)
            The input dataset without measurement errors
        
        Returns
        -------
        X: np.array of shape (num_samples, num_nodes)
            The input dataset with measurement error sampled from a zero
            centered gaussian with variance
            
            .. math::
              \operatorname{Var}(\operatorname{error})} = \gamma*{\operatorname{Var}(\operatorname{signal})}
        """
        n_samples, n_nodes = X.shape
        if not isinstance(self.gamma, list):
            gamma = [self.gamma for _ in range(n_nodes)]
        else:
            gamma = self.gamma
            
        X_std = np.std(X, axis=0)
        for node in range(n_nodes):
            error_std = np.sqrt(gamma[node])*X_std[node]
            error_sample = error_std*np.random.standard_normal((n_samples, ))
            X[:, node] += error_sample
        
        # TODO: unit test error added on one sample dataset
        return X
    


# * Path Canceling Utilities *
class UnfaithfulModel(SCMProperty):
    """Utility functions for SCM generation with measurement error.

    Class modelling unfaithful data cancelling in fully connected triplets
    ``X -> Y <- Z -> X``. 

    Parameters
    ----------
    p_unfaithful: float
        Probability of  unfaitfhul conditional independence in the presence of
        a fully connected triplet. 
    """
    def __init__(self, p_unfaithful: float) -> None:
        super().__init__(identifier="unfaithful")
        self.p_unfaithful = p_unfaithful


    def unfaithful_dataset(
        self,
        X: np.array,
        noise: np.array,
        unfaithful_triplets_toporder: List[List[int]]
    ):
        """Find cancelled edges and modify X according to the unfathful SCM.

        Unfaithful edge cancellations are found by comparing faithful_adj and
        unfaithful_adj matrices. Then, X is post-processed to be distributed 
        faitfhully with respect to the unfaithful_adj adjacency matrix.
        
        Parameters
        ----------
        X : np.array of shape (num_samples, num_nodes)
            The input matrix of the data without path cancelling.
        noise: np.array: of shape (num_samples, num_nodes)
            Matrix of the SCM additive noise terms.
        unfaithful_triplets_toporder : List[List[int]]
            Represent moralized colliders by their topological order.
            E.g. ``1->0<-2``, ``1->2`` is uniquely represented by ``[1, 2, 0]`` toporder of the triplet.
            To model unfaithfulness, add ``X_noise[:, 2]`` to ``X[0:, ]``
        """
        # edges_removed = np.transpose(np.nonzero(unfaithful_adj - faithful_adj))
        added_noise = dict()
        for ordered_triplet in unfaithful_triplets_toporder:
            p1, p2, child = ordered_triplet
            child_added_noise = added_noise.get(child, list())
            if p2 not in child_added_noise:
                X[:, child] += noise[:, p2]
                child_added_noise.append(p2)
                added_noise[child] = child_added_noise
            
        # TODO: unit test
        return X


    def unfaithful_adjacency(self, adjacency):
        """Make a copy of the input adjacency cancelling unfaithful edges.

        Parameters
        ----------
        adjacency: np.array of shape (num_nodes, num_nodes)
            The input adjacency matrix faithful to the data distribution.

        Return
        ------
        unfaithful_adj : np.array
            Transformed groundtruth adjacency matrix unfaithful to the data distribution.
            unfaithful to the graph
        unfaithful_triplets_toporder : List[tuple(int)]
            Represent moralized colliders by their topological order.
            E.g. ``1->0<-2``, ``1->2`` is uniquely represented by ``[1, 2, 0]`` toporder of the triplet
        """

        moral_colliders_toporder = self._find_moral_colliders(adjacency)
        unfaithful_adj = adjacency.copy()
        unfaithful_triplets_toporder = list()

        # For each child, if (p1, p2, c) lead to unfaithful deletion of p1 -> c
        # then I can not reuse p2 in position 1 for future unfaithful deletions
        fixed_edges = list()

        for triplet in moral_colliders_toporder:
            p1, p2, child = triplet
            # Check if triplet still has collider in unfaithful_adj
            if self._is_a_collider(unfaithful_adj, p1, p2, child) and not((p1, child) in fixed_edges):
                if np.random.binomial(n=1, p=self.p_unfaithful):
                    unfaithful_adj[p1, child] = 0 # remove p1 -> c
                    # Remove all others directed paths from the groundtruth and adj graph
                    unfaithful_triplets_toporder.append(triplet)
                    if (p2, child) not in fixed_edges:
                        fixed_edges.append((p2, child))

        # TODO: test if unfaithful_adj is acyclic

        return unfaithful_adj, unfaithful_triplets_toporder


    def _is_a_collider(self, A: np.array, p1: int, p2: int, c: int):
        """
        Paramaters
        ----------
        A : np.array
            Adj. matrix with potential collider
        p1 : int
            First parent of the potential collider
        p2 : int
            Second parent of the potential collider
        c : int
            Head of the potential collider
        """
        # Check p1 -> c and p2 --> c
        collider_struct = A[p1, c] == 1 and A[p2, c] == 1
        return collider_struct
    

    def _find_moral_colliders(self, adjacency):
        """Find moral v-structures in the input adjacency matrix.

        Parameters
        ----------
        adjacency: np.array of shape (num_nodes, num_nodes)
            The input adjacency matrix faithful to the data distribution.

        Return
        ------
        moral_colliders_toporder : List[List[int]]
            Represent moralized colliders by their topological order.
            E.g. ``1->0<-2``,``1->2``  is uniquely represented by ``[1, 2, 0]`` toporder of the triplet.
        """
        # Represent moralized colliders by their topological order.
        moral_colliders_toporder = list()

        # Find moral v-structures
        num_nodes = len(adjacency)
        for child in range(num_nodes):
            parents = np.flatnonzero(adjacency[:, child])
            n_parents = len(parents)
            # Check if child is the tip of v-structures
            if n_parents > 1:
                for i in range(n_parents):
                    for j in range(i+1, n_parents):
                        p_i, p_j = parents[i], parents[j]
                        # Check if collider is moral, and store the triplet's topological order
                        is_moralized = adjacency[p_i, p_j] + adjacency[p_j, p_i] == 1
                        if is_moralized:
                            moral_collider = [p_i, p_j]
                            if adjacency[p_j, p_i] == 1:
                                moral_collider = [p_j, p_i]
                            moral_collider.append(child)
                            moral_colliders_toporder.append(moral_collider)
        return moral_colliders_toporder



# * Time effects utilities * 
class AutoregressiveModel(SCMProperty):
    r"""Utility functions for SCM generation with time lags effects.

    Structural equations take the autoregressive form

    .. math:

    X_i(t):= f_i(\operatorname{PA}_i(t)) + N_i + \sum_{k=t-\operatorname{order}} \alpha(k)*X_i(k)

    where :math:``f_i`` is the nonlinear causal mechanism,
    :math:``N_i`` is the noise term of the structural equation,
    ``\alpha(k)`` is a coefficient uniformly sampled between -1 and 1,
    :math:``t`` is the sample step index, interpreted as the time step.

    Parameters
    ----------
    order: int
        The number of time lags
    """
    def __init__(self, order: int) -> None:
        super().__init__(identifier="autoregressive")
        self.order = order


    def add_time_lag(self, X: np.array):
        """Add time effect to the input.

        The time effect is added as a liner combination of the prevous ``self.order``
        observations of the variable ``X``.

        Parameters
        ----------
        X: np.array of shape (num_samples)
            Observations of a random node.

        Returns
        -------
        X: np.array of shape (num_samples)
            Observations of a random node with addition of the time lagged effects.
        """
        if len(X) <= self.order:
            warnings.warn("The autoregressive order is larger or equal than the number"\
                            " of samples of X. This would cause an IndexError. Reducing"\
                            f" self.order to len(X) - 1 = {len(X) - 1}")
            self.order = len(X) - 1
        linear_coeffs = np.random.uniform(-1, 1, (self.order, ))
        for t in range(self.order, len(X)):
            for k in range(self.order):
                X[t] += linear_coeffs[k]*X[t-k]

        # TODO: unit test time lag added on a dataset with two samples and two nodes.
        return X