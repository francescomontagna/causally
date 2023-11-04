from numpy.typing import NDArray
from utils.data import topological_order
from datasets.random_graphs import ErdosRenyi

# * Utilities *
def top_order_errors(A : NDArray, order : NDArray):
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
        reversed_edges += A[order[i+1:], order[i]].sum()
    return reversed_edges


# * Tests *
def test_given_dags_when_running_topological_order_then_order_is_correct():
    num_nodes = 10
    generator = ErdosRenyi(num_nodes=num_nodes, p_edge=1) # fully connected
    for i in range(20):
        A = generator.get_random_graph(seed=i+42)
        pred_order = topological_order(A)
        errors = top_order_errors(A, pred_order)
        assert  errors == 0, "Predicted order not compatible with the groundtruth adjacency."
