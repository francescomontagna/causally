import random
import pytest
import numpy as np

from causally.graph.random_graph import ErdosRenyi, CustomGraph
from causally.scm.noise import Normal, Uniform
from causally.scm.scm import AdditiveNoiseModel
from causally.scm.context import SCMContext
from causally.scm.causal_mechanism import NeuralNetMechanism, LinearMechanism
from causally.scm.scm_property import (
    _MeasurementErrorMixin,
    _AutoregressiveMixin,
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


@pytest.fixture
def custom_adj():
    # Directed acyclic graph maximally dense
    return np.array(
        [
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 0, 0],
        ]
    )


###################### Test unfaithful model ######################
def test_given_unfaithful_model_when_generating_data_then_path_cancelling_is_observed(
    custom_adj,
):
    graph_generator = CustomGraph(custom_adj)
    noise_generator = Uniform(1, 1)
    causal_mechanism = LinearMechanism(min_weight=1, max_weight=1)

    # Make assumption: unfaithful model
    context = SCMContext()
    context.unfaithful_model(p_unfaithful=1)

    # Generate the data
    model = AdditiveNoiseModel(
        num_samples=100,
        graph_generator=graph_generator,
        noise_generator=noise_generator,
        causal_mechanism=causal_mechanism,
        scm_context=context,
        seed=42,
    )

    # Sample from the model
    dataset, _ = model.sample()
    assert np.allclose(
        dataset.mean(axis=0), np.array([2.0, 0.0, 4.0, 1.0, 0.0]), 0.0005
    )
    assert np.allclose(dataset.std(axis=0), np.array([0.0, 0.0, 0.0, 0.0, 0.0]), 0.0005)


def test_given_fully_connected_matrix_when_model_unfaithful_then_path_cancelling_in_moral_triplets(
    custom_adj,
):
    p_unfaithful = 1
    num_nodes = 5
    p_edge = 1
    num_samples = 2
    graph_generator = CustomGraph(custom_adj)
    noise_generator = Normal(0, 1)
    mechanism = NeuralNetMechanism()
    context = SCMContext()
    context.unfaithful_model(p_unfaithful)
    scm = AdditiveNoiseModel(
        num_samples, graph_generator, noise_generator, mechanism, context
    )

    _, _ = scm.sample()  # sampling creates the unfaithful adjacency
    unfaithful_adj = scm.scm_context.unfaithful_adjacency

    target_adj = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
        ]
    )
    assert np.allclose(unfaithful_adj, target_adj)


def test_given_adj_when_p_unfaithful_0_then_adjacency_and_unfaithful_adjacency_are_equal():
    p_unfaithful = 1e-9
    num_nodes = 20
    p_edge = 1
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    noise_generator = Normal(0, 1)
    mechanism = NeuralNetMechanism()
    context = SCMContext()
    context.unfaithful_model(p_unfaithful)
    scm = AdditiveNoiseModel(1000, graph_generator, noise_generator, mechanism, context)

    _, adjacency = scm.sample()  # sampling creates the unfaithful adjacency
    unfaithful_adj = scm.scm_context.unfaithful_adjacency
    assert np.allclose(adjacency, unfaithful_adj, 0.0005)


###################### Test confounded model ######################
def test_given_p_confounded_when_generating_graphs_then_rate_of_confounded_pairs_is_p_confounded():
    p_confounder = 0.2
    num_nodes = 20
    p_edge = 0.4
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    noise_generator = Normal(0, 1)
    mechanism = NeuralNetMechanism()
    context = SCMContext()
    context.confounded_model(p_confounder)

    # sum_confounded_pairs = 0
    for _ in range(10):
        scm = AdditiveNoiseModel(
            10, graph_generator, noise_generator, mechanism, context
        )

        _, _ = scm.sample()  # sampling creates the confounded adjacency
        confounded_adj = scm.scm_context.confounded_adjacency
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
    p_confounder = 0.001
    num_nodes = 20
    p_edge = 0.4
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    noise_generator = Normal(0, 1)
    mechanism = NeuralNetMechanism()
    context = SCMContext()
    context.confounded_model(p_confounder)
    scm = AdditiveNoiseModel(10, graph_generator, noise_generator, mechanism, context)
    _, _ = scm.sample()  # sampling creates the confounded adjacency
    confounded_adj = scm.scm_context.confounded_adjacency
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
