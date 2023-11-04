"""Generation and storage of data for large scale experiments
"""

import networkx as nx
import numpy as np
from numpy.typing import NDArray


# * Utility functions *
def generate_and_store_dataset(
    data_output_file : str,
    gt_output_file: str,
    model
):
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
    return int(num_nodes*(num_nodes-1)/2)


def topological_order(adjacency: NDArray):
    # DAG test
    if not nx.is_directed_acyclic_graph(nx.from_numpy_array(adjacency, create_using=nx.DiGraph)):
        raise ValueError("The input adjacency matrix is not acyclic.")
    

    # Define toporder one leaf at the time
    order = list()
    num_nodes = len(adjacency)
    mask = np.zeros((num_nodes))
    for _ in range(num_nodes):
        children_per_node = adjacency.sum(axis=1) + mask # adjacency[i, j] = 1 --> i parent of j
        leaf = np.argmin(children_per_node) # find leaf as node with no children
        mask[leaf] += float("inf") # select node only once
        order.append(leaf) # update order
    
    order = order[::-1] # source first
    return order