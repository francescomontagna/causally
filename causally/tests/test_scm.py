import torch
import numpy as np
from causally.scm.scm import LinearModel, AdditiveNoiseModel, PostNonlinearModel
from causally.scm.causal_mechanism import (
    LinearMechanism,
    NeuralNetMechanism,
    GaussianProcessMechanism,
)
from causally.graph.random_graph import ErdosRenyi
from causally.scm.noise import Normal

# Random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def test_given_linear_and_anm_with_linear_mechanism_when_sample_then_same_dataset():
    # Additive noise model
    anm = AdditiveNoiseModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=LinearMechanism(),
        seed=SEED,
    )

    X_anm, _ = anm.sample()

    # Linear
    lingam = LinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        seed=SEED,
    )

    X_lingam, _ = lingam.sample()

    assert np.array_equal(X_anm, X_lingam)


def test_given_linear_and_pnl_with_identity_and_linear_mechanism_when_sample_then_same_dataset():
    # Post nonlinear model
    pnl = PostNonlinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=LinearMechanism(),
        invertible_function=lambda x: x,
        seed=SEED,
    )

    X_pnl, _ = pnl.sample()

    # Linear
    lingam = LinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        seed=SEED,
    )

    X_lingam, _ = lingam.sample()

    assert np.array_equal(X_pnl, X_lingam)


def test_given_anm_and_pnl_with_identity_and_nn_mechanism_when_sample_then_same_dataset():
    # Post nonlinear model
    pnl = PostNonlinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=NeuralNetMechanism(),
        invertible_function=lambda x: x,
        seed=SEED,
    )

    X_pnl, _ = pnl.sample()

    # Additive noise model
    anm = AdditiveNoiseModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=NeuralNetMechanism(),
        seed=SEED,
    )

    X_anm, _ = anm.sample()

    assert np.array_equal(X_pnl, X_anm)


def test_given_anm_and_pnl_with_identity_and_gp_mechanism_when_sample_then_same_dataset():
    # Post nonlinear model
    pnl = PostNonlinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=GaussianProcessMechanism(),
        invertible_function=lambda x: x,
        seed=SEED,
    )

    X_pnl, _ = pnl.sample()

    # Additive noise model
    anm = AdditiveNoiseModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=Normal(0, 1),
        causal_mechanism=GaussianProcessMechanism(),
        seed=SEED,
    )

    X_anm, _ = anm.sample()

    assert np.array_equal(X_pnl, X_anm)
