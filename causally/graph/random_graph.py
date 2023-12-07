import random
import igraph as ig
import numpy as np
import networkx as nx
from abc import ABCMeta, abstractmethod

from causally.utils.graph import max_edges_in_dag


# Base class
class GraphGenerator(metaclass=ABCMeta):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    @abstractmethod
    def get_random_graph(self, seed: int) -> np.array:
        """Sample the random directed acyclic graph (DAG).

        Parameters
        ----------
        seed: int, default None
            The random seed for the graph generation.

        Returns
        -------
        A: np.array
            Adjacency matrix representation of the output DAG.
            The presence of directed edge from node ``i`` to node ``j``
            is denoted by ``A[i, j] = 1``. Absence of edges is denote by 0.
        """
        raise NotImplementedError()

    def _manual_seed(self, seed: int) -> None:
        """Set manual seed for deterministic graph generation. If None, seed is not set."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _make_random_order(self, A: np.array) -> np.array:
        """Randomly permute nodes of A to avoid trivial ordering."""
        n_nodes = A.shape[0]
        permutation = np.random.permutation(range(n_nodes))
        A = A[permutation, :]
        A = A[:, permutation]
        return A


# ***************************************** #
# Gaussian Random Partition Graphs Generator
# ***************************************** #
class GaussianRandomPartition(GraphGenerator):
    """
    Generator of Gaussian Random Partition directed acyclic graphs.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    p_in : float
        Probability of edge connection with nodes in the cluster.
    p_out : float
        Probability of edge connection with nodes in different clusters.
    n_clusters : int
        Number of clusters in the graph.
    min_cluster_size: int, default 2
        Minimum number of elements for each cluster.
    """

    def __init__(
        self,
        num_nodes: float,
        p_in: float,
        p_out: float,
        n_clusters: int,
        min_cluster_size: int = 2,
    ):
        if num_nodes / n_clusters < min_cluster_size:
            raise ValueError(
                f"Expected ratio ``num_nodes/n_clusters`` must be at least {min_cluster_size}"
                f" Instead got {num_nodes/n_clusters}. Decrease ``n_clusters`` or ``min_cluster_size``."
            )

        super().__init__(num_nodes)
        self.p_in = p_in
        self.p_out = p_out
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.size_of_clusters = self._sample_cluster_sizes()

    def get_random_graph(self, seed: int = None) -> np.array:
        self._manual_seed(seed)

        # Initialize with the first cluster and remove it from the list
        A = self._sample_er_cluster(self.size_of_clusters[0])
        size_of_clusters = np.delete(self.size_of_clusters, [0])

        # Join all clusters together
        for c_size in size_of_clusters:
            A = self._disjoint_union(A, c_size)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A

    def _sample_cluster_sizes(self) -> np.array:
        """Sample the size of each cluset.

        The size of the clusters is sampled from a multinomial distribution,
        and post-processed to ensure at least 3 nodes per cluster
        """
        cluster_sizes = np.random.multinomial(
            self.num_nodes, pvals=[1 / self.n_clusters for _ in range(self.n_clusters)]
        )
        # At least 3 elements per cluster. Take elements from largest to smallest cluster
        while np.min(cluster_sizes) < self.min_cluster_size:
            argmax = np.argmax(cluster_sizes)
            argmin = np.argmin(cluster_sizes)
            cluster_sizes[argmax] -= 1
            cluster_sizes[argmin] += 1
        return cluster_sizes

    def _sample_er_cluster(self, cluster_size) -> np.array:
        """Sample each cluster of GRP graphs with Erdos-Renyi model"""
        A = ErdosRenyi(num_nodes=cluster_size, p_edge=self.p_in).get_random_graph()
        return A

    def _disjoint_union(self, A: np.array, c_size: int) -> np.array:
        """
        Merge adjacency A with cluster of size ``c_size`` nodes into a DAG.

        The cluster is sampled from the Erdos-Rényi model.
        Nodes are labeled with respect to the cluster they belong.

        Parameters
        ----------
        A : np.array
            Current adjacency matrix
        c_size : int
            Size of the cluster to generate
        """
        # Join the graphs by block matrices
        n = A.shape[0]
        er_cluster = self._sample_er_cluster(cluster_size=c_size)
        er_cluster = np.hstack([np.zeros((c_size, n)), er_cluster])
        A = np.hstack([A, np.zeros((n, c_size))])
        A = np.vstack([A, er_cluster])

        # Add connections among clusters from A to er_cluster
        for i in range(n):
            for j in range(n, i + c_size):
                if np.random.binomial(n=1, p=self.p_out) == 1:
                    # print(f"edge {(i, j)} between clusters!")
                    A[i, j] = 1

        return A


# ***************************** #
#  Erdos-Rényi Graphs Generator #
# ***************************** #
class ErdosRenyi(GraphGenerator):
    """
    Generator of Erdos-Renyi directed acyclic graphs.

    This class is a wrapper of ``igraph`` Erdos-Renyi graph sampler.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    expected_degree : int, default is None
        Expected degree of each node.
    p_edge : float, default is None
        Probability of edge between each pair of nodes.
    min_num_edges: int, default 2
        The minimum number of edges required in the graph.
        If 0, allows for empty graphs. If larger than the maximum number of edges for the DAG,
        it is set to ``num_nodes * (num_nodes - 1) / 2``.
    """

    def __init__(
        self,
        num_nodes: int,
        expected_degree: int = None,
        p_edge: float = None,
        min_num_edges: int = 2,
    ):
        if expected_degree is not None and p_edge is not None:
            raise ValueError(
                "Only one parameter between 'p_edge' and 'expected_degree' can be"
                f" provided. Got instead expected_degree={expected_degree}"
                f"and p_edge={p_edge}."
            )
        if expected_degree is None and p_edge is None:
            raise ValueError(
                "Please provide a value for one and only one argument between"
                " 'expected_degree' and 'p_edge'."
            )
        if expected_degree is not None and expected_degree == 0:
            raise ValueError(
                "expected value of 'expected_degree' is at least 1. Got 0 instead"
            )
        if p_edge is not None and p_edge < 0.1:
            raise ValueError(
                "expected value of 'p_edge' is at least 0.1." f" Got {p_edge} instead"
            )
        if min_num_edges < 0:
            raise ValueError(
                "Minimum number of edges must be larger or equals to 0."
                + f" Got instead {min_num_edges}."
            )

        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.p_edge = p_edge
        self.min_num_edges = min_num_edges

    def get_random_graph(self, seed: int = None) -> np.array:
        self._manual_seed(seed)
        A = -np.ones((self.num_nodes, self.num_nodes))

        # Ensure at least self.min_num_edges edges (one edge if the graph is bivariate)
        while np.sum(A) < min(self.min_num_edges, max_edges_in_dag(self.num_nodes)):
            if self.p_edge is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(n=self.num_nodes, p=self.p_edge)
            elif self.expected_degree is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(
                    n=self.num_nodes, m=self.expected_degree * self.num_nodes
                )
            undirected_adjacency = ig_to_adjmat(undirected_graph)
            A = acyclic_orientation(undirected_adjacency)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A


# ******************************** #
# Barabasi Albert Graphs Generator #
# ******************************** #
class BarabasiAlbert(GraphGenerator):
    """
    Generator of Scale Free directed acyclic graphs.

    This class is a wrapper of ``igraph`` Barabasi graph sampler.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    expected_degree : int
        Expected degree of each node.
    preferential_attachment_out: bool, default True
        Select the preferential attachment strategy. If True,
        new nodes tend to have incoming edge from existing nodes with high out-degree.
        Else, new nodes tend to have outcoming edge towards existing nodes with high in-degree.
    min_num_edges: int, default 2
        The minimum number of edges required in the graph.
        If 0, allows for empty graphs. If larger than the maximum number of edges for the DAG,
        it is set to ``num_nodes * (num_nodes - 1) / 2``.
    """

    def __init__(
        self,
        num_nodes: int,
        expected_degree: int,
        preferential_attachment_out: bool = True,
        min_num_edges: int = 2,
    ):
        if expected_degree == 0:
            raise ValueError(
                "expected value of 'expected_degree' is at least 1. Got 0 instead"
            )
        if min_num_edges < 0:
            raise ValueError(
                "Minimum number of edges must be larger or equals to 0."
                + f" Got instead {min_num_edges}."
            )
        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.preferential_attachment_out = preferential_attachment_out
        self.min_num_edges = min_num_edges

    def get_random_graph(self, seed: int = None) -> np.array:
        self._manual_seed(seed)
        A = -np.ones((self.num_nodes, self.num_nodes))

        # Ensure at least self.min_num_edges edges (one edge if the graph is bivariate)
        while np.sum(A) < min(self.min_num_edges, max_edges_in_dag(self.num_nodes)):
            G = ig.Graph.Barabasi(
                n=self.num_nodes, m=self.expected_degree, directed=True
            )
            A = ig_to_adjmat(G)
            if self.preferential_attachment_out:
                A = A.transpose(1, 0)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A


# ********************** #
#       Utilities        #
# ********************** #
def acyclic_orientation(A):
    return np.triu(A, k=1)


def ig_to_adjmat(G: ig.Graph):
    return np.array(G.get_adjacency().data)


def graph_viz(A: np.array):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nx.draw_networkx(G)
