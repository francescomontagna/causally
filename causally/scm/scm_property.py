import random
import numpy as np
import warnings
from typing import Union, List
from causally.utils.graph import is_a_collider, find_moral_colliders


# Collection of mixing classes for data generation under required assumptions


# * Confounded model utilites *
class _ConfoundedMixin:

    @ staticmethod
    def confound_adjacency(adjacency: np.array, p_confounder=float):
        """Add latent common causes to the input adjacency matrix.

        Parameters
        ----------
        adjacency: np.array of shape (num_nodes x num_nodes)
            The adjacency matrix without latent confounders.
        p_confounder: float
            The probability of adding a latent common cause between a pair of nodes,
            sampled as a Bernoulli random variable.

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
        
        # TODO: test if confounded_adj is uppertriangular. 
        return confounded_adj
    

    @ staticmethod
    def confound_dataset(X: np.array, n_confounders: int):
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
class _MeasurementErrorMixin:
    @staticmethod
    def add_measure_error(X: np.array, gamma: Union[float, List[float]]):
        r"""Add measurement error to the input dataset X.

        Parameters
        ----------
        X: np.array of shape (num_samples, num_nodes)
            The input dataset without measurement errors
        gamma: Union[float, List[float]] 
            The inverse signal to noise ratio
        
        Returns
        -------
        X: np.array of shape (num_samples, num_nodes)
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
            error_std = np.sqrt(gamma[node])*X_std[node]
            error_sample = error_std*np.random.standard_normal((n_samples, ))
            X[:, node] += error_sample
        
        # TODO: unit test error added on one sample dataset
        return X
    

# * Path Canceling Utilities *
class _UnfaithfulMixin:

    @staticmethod
    def unfaithful_dataset(
        X: np.array,
        noise: np.array,
        unfaithful_triplets_toporder: List[List[int]]
    ):
        """Modify X according to the unfaithful SCM.
        
        Parameters
        ----------
        X : np.array of shape (num_samples, num_nodes)
            The input matrix of the data without path cancelling.
        noise: np.array: of shape (num_samples, num_nodes)
            Matrix of the SCM additive noise terms.
        unfaithful_triplets_toporder : List[List[int]]
            Represent moralized colliders with in unfaithful path cancelling by their causal order.
            E.g. ``1->0<-2<-1`` is uniquely represented by ``[1, 2, 0]`` topological order of the
            triplet. To model unfaithfulness, add ``X_noise[:, 2]`` to ``X[0:, ]``
        """
        # edges_removed = np.transpose(np.nonzero(unfaithful_adj - faithful_adj))
        added_noise = dict() 
        for ordered_triplet in unfaithful_triplets_toporder:
            p1, p2, child = ordered_triplet
            child_added_noise = added_noise.get(child, list())
            if p2 not in child_added_noise: # Avoid unfaithful effects of p2 on child more than once
                X[:, child] += noise[:, p2]
                child_added_noise.append(p2)
                added_noise[child] = child_added_noise
            
        # TODO: unit test
        return X


    @staticmethod
    def unfaithful_adjacency(adjacency: np.array, p_unfaithful: float):
        """Make a copy of the input adjacency cancelling unfaithful edges.

        Parameters
        ----------
        adjacency: np.array of shape (num_nodes, num_nodes)
            The input adjacency matrix faithful to the data distribution.
        p_unfaithful: float
            Probability of  unfaitfhul conditional independence in the presence of
            a fully connected triplet. 

        Return
        ------
        unfaithful_adj : np.array
            Transformed groundtruth adjacency matrix unfaithful to the data distribution.
            unfaithful to the graph
        unfaithful_triplets_toporder : List[tuple(int)]
            Represent moralized colliders by their topological order.
            E.g. ``1->0<-2``, ``1->2`` is uniquely represented by ``[1, 2, 0]`` toporder of the triplet
        """
        moral_colliders_toporder = find_moral_colliders(adjacency)
        unfaithful_adj = adjacency.copy()
        unfaithful_triplets_toporder = list()

        # For each child, if (p1, p2, c) lead to unfaithful deletion of p1 -> c
        # then I can not reuse p2 in position 1 for future unfaithful deletions
        fixed_edges = list()

        for triplet in moral_colliders_toporder:
            p1, p2, child = triplet
            # Check if triplet still has collider in unfaithful_adj
            if is_a_collider(unfaithful_adj, p1, p2, child) and not((p1, child) in fixed_edges):
                if np.random.binomial(n=1, p=p_unfaithful) == 1:
                    unfaithful_adj[p1, child] = 0 # remove p1 -> c
                    # Remove all others directed paths from the groundtruth and adj graph
                    unfaithful_triplets_toporder.append(triplet)
                    if (p2, child) not in fixed_edges:
                        fixed_edges.append((p2, child))

        # TODO: test if unfaithful_adj is acyclic

        return unfaithful_adj, unfaithful_triplets_toporder



# * Time effects utilities * 
class _AutoregressiveMixin:

    @staticmethod
    def add_time_lag(X: np.array, order: int):
        """Add time effect to the input.

        The time effect is added as a liner combination of the prevous ``order``
        observations of the variable ``X``.

        Parameters
        ----------
        X: np.array of shape (num_samples)
            Observations of a random node.
        order: int
            The number of time lags

        Returns
        -------
        X: np.array of shape (num_samples)
            Observations of a random node with addition of the time lagged effects.
        """
        if len(X) <= order:
            warnings.warn("The autoregressive order is larger or equal than the number"\
                            " of samples of X. This would cause an IndexError. Reducing"\
                            f" order to len(X) - 1 = {len(X) - 1}")
            order = len(X) - 1
        linear_coeffs = np.random.uniform(-1, 1, (order, ))
        for t in range(order, len(X)):
            for k in range(order):
                X[t] += linear_coeffs[k]*X[t-k]

        # TODO: unit test time lag added on a dataset with two samples and two nodes.
        return X