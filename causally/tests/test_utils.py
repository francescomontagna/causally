import numpy as np
from numpy.typing import NDArray
from causally.utils.graph import topological_order, find_moral_colliders
from causally.graph.random_graph import ErdosRenyi


# * Utilities *
def top_order_errors(A: NDArray, order: NDArray):
    """Topological order divergence.

    Parameters
    ----------
    A : np.array
        Ground truth adjacency matrix
    order: np.array
        Inferred topological order (order[0] source node)

    Returns
    -------
    reversed_edges: int
        Number of errors in the topological order.
    """
    reversed_edges = 0
    for i in range(len(order)):
        reversed_edges += A[order[i + 1 :], order[i]].sum()
    return reversed_edges


# * Tests *
def test_given_dags_when_running_topological_order_then_order_is_correct():
    num_nodes = 10
    generator = ErdosRenyi(num_nodes=num_nodes, p_edge=1)  # fully connected
    for i in range(20):
        A = generator.get_random_graph(seed=i + 42)
        pred_order = topological_order(A)
        errors = top_order_errors(A, pred_order)
        assert (
            errors == 0
        ), "Predicted order not compatible with the groundtruth adjacency."


def test_given_dag_with_moral_collider_then_find_moral_collider_finds_it():
    dag = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert find_moral_colliders(dag)[0] == [0, 1, 2]
