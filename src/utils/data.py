"""Generation and storage of data for large scale experiments
"""

import numpy as np


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
