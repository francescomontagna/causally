import torch
import numpy as np
from scipy.stats import shapiro 
from causally.scm.noise import MLPNoise, Normal, CustomNoise

SEED = 42
torch.random.manual_seed(SEED)
np.random.seed(SEED)


def test_given_gauss_noise_sample_when_loc_zero_scale_one_then_sample_mean_zero_and_std_one():
    normal_sampler = Normal(loc=0, std=1)
    sample = normal_sampler.sample((2000, 1))
    assert np.allclose(
        sample.mean(), 0, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 0."
    assert np.allclose(
        sample.std(), 1, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 1."


def test_given_mlp_noise_sample_when_checking_shape_then_has_two_dimensions():
    mlp_sampler = MLPNoise(hidden_dim=100)
    sample = mlp_sampler.sample((1000, 1))
    assert sample.shape == (
        1000,
        1,
    ), f"Sample shape is {sample.shape}, expected (1000, 1) instead"


def test_given_mlp_noise_sample_when_standardized_is_true_then_mean_zero_and_std_one():
    mlp_sampler = MLPNoise(hidden_dim=100, standardize=True)
    sample = mlp_sampler.sample((1000, 1))
    assert np.allclose(
        sample.mean(), 0, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 0."
    assert np.allclose(
        sample.std(), 1, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 1."


def test_given_std_normal_to_custom_noise_when_sample_data_are_std_normal_with_low_p_value():
    sampler = CustomNoise(lambda x: 1/np.sqrt(2*np.pi)*np.exp(-0.5*x**2))
    sample = sampler.sample((1000,))
    _, p_val = shapiro(sample)
    assert p_val > 0.05, "Sample data not normal with statistical significance"
    assert np.allclose(
        sample.mean(), 0, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 0."
    assert np.allclose(
        sample.std(), 1, atol=0.05
    ), f"Empirical mean is {sample.mean()}, expected closer to 1."