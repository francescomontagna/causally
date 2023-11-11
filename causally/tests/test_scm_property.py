import numpy as np
import random

from causally.graph.random_graph import ErdosRenyi
from causally.scm.scm_property import (
    _ConfoundedMixin,
    _MeasurementErrorMixin,
    _UnfaithfulMixin,
    _AutoregressiveMixin,
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


###################### Test unfaithful model ######################
def test_given_fully_connected_matrix_when_model_unfaithful_then_path_cancelling_in_moral_triplets():
    adjacency = np.triu(np.ones((5, 5)), k=1)
    unfaithful_adj, _ = _UnfaithfulMixin.unfaithful_adjacency(adjacency, p_unfaithful=1)

    target_adj = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    assert np.allclose(unfaithful_adj, target_adj)


def test_given_adj_when_p_unfaithful_0_then_adjacency_and_unfaithful_adjacency_are_equal():
    p_unfaithful = 0.0
    num_nodes = 20
    p_edge = 1
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    adjacency = graph_generator.get_random_graph()
    unfaithful_adj, _ = _UnfaithfulMixin.unfaithful_adjacency(adjacency, p_unfaithful)
    assert np.allclose(adjacency, unfaithful_adj, 0.00001)


###################### Test confounded model ######################
def test_given_p_confounded_when_generating_graphs_then_rate_of_confounded_pairs_is_p_confounded():
    p_confounder = 0.2
    num_nodes = 20
    p_edge = 0.4
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)

    # sum_confounded_pairs = 0
    for _ in range(10):
        adjacency = graph_generator.get_random_graph()
        confounded_adj = _ConfoundedMixin.confound_adjacency(adjacency, p_confounder)
        confounders_matrix = confounded_adj[num_nodes:, num_nodes:]
        n_direct_confounders = 0
        for confounder in range(num_nodes):
            n_direct_confounders += int(confounders_matrix[confounder, :].sum())

        number_of_pairs = num_nodes * (num_nodes - 1)
        assert abs(n_direct_confounders / number_of_pairs - p_confounder) < 0.05, (
            f"Expected rate of confounders 0.2 +- 0.05, instead got: "
            f"{abs(n_direct_confounders/number_of_pairs)}"
        )


def test_given_p_confounded_0_then_confounders_matrix_empty():
    p_confounder = 0.0
    num_nodes = 20
    p_edge = 0.4
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    adjacency = graph_generator.get_random_graph()
    confounded_adj = _ConfoundedMixin.confound_adjacency(adjacency, p_confounder)
    assert np.sum(confounded_adj[:, :num_nodes]) == 0


###################### Test autoregressive model ######################
def test_given_dataset_when_adding_time_effect_of_order_one_then_previous_sample_is_added():
    X = np.ones((2, 2))
    X = _AutoregressiveMixin.add_time_lag(X, order=1, weight_a=1, weight_b=1)
    assert np.allclose(X, np.array([[1.0, 1.0], [2.0, 2.0]]), 0.00005)


###################### Test measurement error model ######################
def test_given_dataset_when_adding_measurement_error_with_gamma_one_then_sample_value_change():
    X = np.ones((2, 2))
    X_error = _MeasurementErrorMixin.add_measure_error(X, gamma=1)
    assert not np.allclose(2 * X, X_error, 0.00005)
