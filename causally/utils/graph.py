"""Generation and storage of data for large scale experiments
"""
import networkx as nx
import numpy as np


# * Utility functions *
def generate_and_store_dataset(data_output_file: str, gt_output_file: str, model):
    """Generate a dataset with `model`, and store the resulting dataset and grountruth.

    Parameters
    ----------
    data_output_file: str
        Path for storage of the dataset as `.npy` file.
    gt_output_file: str
        Path for storage of the groundtruth adjacency matrix as `.npy` file.
    model: BaseStructuralCausalModel
        Instance of `BaseStructuralCausalModel` generating the data and the groundtruth.
    """
    dataset, groundtruth = model.sample()
    np.save(data_output_file, dataset)
    np.save(gt_output_file, groundtruth)


def max_edges_in_dag(num_nodes: int) -> int:
    """Compute the maximum number of edges allowed for a direcetd acyclic graph:

    The max number of edges is compute as `self.num_nodes*(self.num_nodes-1)/2`
    """
    return int(num_nodes * (num_nodes - 1) / 2)


def topological_order(adjacency: np.array):
    # DAG test
    if not nx.is_directed_acyclic_graph(
        nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    ):
        raise ValueError("The input adjacency matrix is not acyclic.")

    # Define toporder one leaf at the time
    order = list()
    num_nodes = len(adjacency)
    mask = np.zeros((num_nodes))
    for _ in range(num_nodes):
        children_per_node = (
            adjacency.sum(axis=1) + mask
        )  # adjacency[i, j] = 1 --> i parent of j
        leaf = np.argmin(children_per_node)  # find leaf as node with no children
        mask[leaf] += float("inf")  # select node only once
        order.append(leaf)  # update order

    order = order[::-1]  # source first
    return order


# * For unfaithful generation *
def is_a_collider(A: np.array, p1: int, p2: int, c: int):
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


def find_moral_colliders(adjacency: np.array):
    """Find moral v-structures in the input adjacency matrix.

    Parameters
    ----------
    adjacency: np.array of shape (num_nodes, num_nodes)
        The input adjacency matrix faithful to the data distribution.

    Returns
    -------
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
                for j in range(i + 1, n_parents):
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
