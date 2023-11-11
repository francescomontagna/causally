import torch
import numpy as np
from causally.scm.causal_mechanism import LinearMechanism, NeuralNetMechanism

# Random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.random.manual_seed(SEED)


def test_given_linear_mechanism_when_predict_then_output_is_linear_reg():
    mechanism = LinearMechanism()
    X = np.random.standard_normal((10, 1))
    y = mechanism.predict(X)

    linear_coeff = mechanism.linear_reg.coef_
    assert all(
        y == (X * linear_coeff).squeeze()
    ), "Unexpected value of the linear mechanism output."


def test_given_linear_mechanisms_then_weights_in_required_range():
    mechanism = LinearMechanism(min_weight=-1, max_weight=+1, min_abs_weight=0.05)
    X = np.random.standard_normal((10, 5))
    _ = mechanism.predict(X)

    linear_coeffs = mechanism.linear_reg.coef_
    assert (
        all(linear_coeffs < 1)
        and all(linear_coeffs > -1)
        and all(abs(linear_coeffs) > 0.05)
    ), "Coefficients not in the expected range [-1, -0.05] U [0.05, 1]."


def test_given_nn_mechanisms_then_weights_mean_zero_and_std_one():
    mechanism = NeuralNetMechanism(0, 1, hidden_dim=1000)
    X = np.random.standard_normal((10, 1))
    _ = mechanism.predict(X)

    weights = mechanism.model[0].weight.detach()
    assert np.allclose(
        weights.mean(), 0, atol=0.05
    ), f"Empirical mean is {weights.mean()}, expected closer to 0."
    assert np.allclose(
        weights.std(), 1, atol=0.05
    ), f"Empirical mean is {weights.mean()}, expected closer to 1."
