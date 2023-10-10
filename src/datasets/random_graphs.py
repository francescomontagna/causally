import igraph as ig
import numpy as np
import networkx as nx
from numpy.typing import NDArray
from abc import ABCMeta, abstractmethod

from utils.data import max_edges_in_dag


# Base class
class GraphGenerator(metaclass=ABCMeta):
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    @abstractmethod
    def get_random_graph(self):
        raise NotImplementedError()

    def _manual_seed(self, seed: int) -> None:
        """Set manual seed for deterministic graph generation. If None, seed is not set."""
        if seed is not None:
            np.random.seed(seed)
    

    def _make_random_order(self, A: NDArray) -> NDArray:
        """Randomly permute nodes of A to avoid trivial ordering."""
        n_nodes = A.shape[0]
        order = np.random.permutation(range(n_nodes))
        A = A[order, :]
        A = A[:, order]
        return A


# ***************************************** #
# Gaussian Random Partition Graphs Generator 
# ***************************************** #
class GaussianRandomPartition(GraphGenerator):
    def __init__(
        self,
        num_nodes: float,
        p_in: float,
        p_out: float,
        n_clusters: int,
        min_cluster_size: int = 2
    ):
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
            Minimum number of elements for each cluser.

        Attributes
        ----------
        size_of_clusters : List[int]
            The size of the graph's clusters. This is randomly sampled from a multinomial
            distribution with parameters TODO: which paremeters?
        """
        if num_nodes/n_clusters < min_cluster_size:
            raise ValueError(f"Expected ratio `num_nodes/n_clusters' must be at least {min_cluster_size}"\
                             f" Instead got {num_nodes/n_clusters}. Decrease `n_clusters` or `min_cluster_size`.")
                             

        super().__init__(num_nodes)
        self.p_in = p_in
        self.p_out = p_out
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.size_of_clusters = self._sample_cluster_sizes()


    def get_random_graph(self, seed: int = None) -> NDArray:
        self._manual_seed(seed)

        # print(f"size of the clusters: {size_of_clusters}")

        # Initialize with the first cluster and remove it from the list
        A = self._sample_er_cluster(self.size_of_clusters[0])
        size_of_clusters = np.delete(self.size_of_clusters, [0])

        # Join all clusters together
        for c_size in size_of_clusters:
            A = self._disjoint_union(A, c_size)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A


    def _sample_cluster_sizes(self) -> NDArray:
        """Sample the size of each cluset.

        The size of the clusters is sampled from a multinomial distribution, 
        and post-processed to ensure at least 3 nodes per cluster
        """
        cluster_sizes = np.random.multinomial(
            self.num_nodes, pvals=[1/self.n_clusters for _ in range(self.n_clusters)]
        )
        # At least 3 elements per cluster. Take elements from largest to smallest cluster
        while np.min(cluster_sizes) < self.min_cluster_size:
            argmax = np.argmax(cluster_sizes)
            argmin = np.argmin(cluster_sizes)
            cluster_sizes[argmax] -= 1
            cluster_sizes[argmin] += 1
        return cluster_sizes


    def _sample_er_cluster(self, cluster_size) -> NDArray:
        """Sample each cluster of GRP graphs with Erdos-Renyi model
        """
        A = ErdosRenyi(num_nodes=cluster_size, p_edge=self.p_in).get_random_graph()
        return A


    def _disjoint_union(self, A: NDArray, c_size: int) -> NDArray:
        """
        Merge adjacency A with cluster of size `c_size` nodes into a DAG.
        
        The cluster is sampled from the Erdos-Rényi model. 
        Nodes are labeled with respect to the cluster they belong.

        Parameters
        ----------
        A : NDArray
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
            for j in range(n, i+c_size):
                if np.random.binomial(n=1, p=self.p_out) == 1:
                    # print(f"edge {(i, j)} between clusters!")
                    A[i, j] = 1

        return A

    
# ***************************** #
#  Erdos-Rényi Graphs Generator #
# ***************************** #
class ErdosRenyi(GraphGenerator):
    def __init__(
        self,
        num_nodes : int,
        expected_degree : int = None,
        p_edge : float = None
    ):
        """
        Generator of Erdos-Renyi directed acyclic graphs.

        This class is a wrapper of `igraph` Erdos-Renyi graph sampler.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes
        expected_degree : int, default is None
            Expected degree of each node.
        p_edge : float, default is None
            Probability of edge between each pair of nodes.
        """
        if expected_degree is not None and p_edge is not None:
            raise ValueError("Only one parameter between 'p' and 'm' can be assigned a value."\
                             f" Got instead m={expected_degree} and p={p_edge}.")
        if expected_degree is None and p_edge is None:
            raise ValueError("Please provide a value for one of argument between 'm' and 'p'.")

        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.p_edge = p_edge


    def get_random_graph(self, seed: int = None) -> NDArray:
        self._manual_seed(seed)
        A = np.zeros((self.num_nodes, self.num_nodes))

        # Ensure at least two edges (one edge if the graph is bivariate)
        while np.sum(A) < min(2, max_edges_in_dag(self.num_nodes)):
            if self.p_edge is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(n=self.num_nodes, p=self.p_edge)
            elif self.expected_degree is not None:
                undirected_graph = ig.Graph.Erdos_Renyi(n=self.num_nodes, m=self.expected_degree*self.num_nodes)
            undirected_adjacency = ig_to_adjmat(undirected_graph)
            A = acyclic_orientation(undirected_adjacency)

        # Permute to avoid trivial ordering
        A = self._make_random_order(A)
        return A
    

# ******************************** #
# Barabasi Albert Graphs Generator #
# ******************************** #
class BarabasiAlbert(GraphGenerator):
    def __init__(
        self,
        num_nodes : int,
        expected_degree : int,
        preferential_attachment_out: bool = True
    ):
        """
        Generator of Scale Free directed acyclic graphs.

        This class is a wrapper of `igraph` Barabasi graph sampler.
        
        Parameters
        ----------
        d : int
            Number of nodes
        expected_degree : int
            Expected degree of each node.
        preferential_attachment_out: bool, default True
            Select the preferential attachment strategy. If True,
            new nodes tend to have incoming edge from existing nodes with high out-degree.
            Else, new nodes tend to have outcoming edge towards existing nodes with high in-degree.
        """
        super().__init__(num_nodes)
        self.expected_degree = expected_degree
        self.preferential_attachment_out = preferential_attachment_out

    def get_random_graph(self, seed: int = None) -> NDArray:
        self._manual_seed(seed)
        A = np.zeros((self.num_nodes, self.num_nodes))
        
        # Ensure at least two edges (one edge if the graph is bivariate)
        while np.sum(A) < min(2, max_edges_in_dag(self.num_nodes)):
            G = ig.Graph.Barabasi(n=self.num_nodes, m=self.expected_degree, directed=True)
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

def ig_to_adjmat(G : ig.Graph):
    return np.array(G.get_adjacency().data)

def graph_viz(A : np.array):
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    nx.draw_networkx(G)
