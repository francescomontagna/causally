import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.scm import PostNonlinearModel, AdditiveNoiseModel, LinearModel
from datasets.causal_mechanisms import LinearMechanism, NeuralNetMechanism, GaussianProcessMechanism
from datasets.random_graphs import ErdosRenyi
from datasets.random_noises import MLPNoise, Normal

# Random seeds for reproducibility
SEED=42
np.random.seed(SEED)
torch.random.manual_seed(SEED)

def main():
    pnl = PostNonlinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=MLPNoise(a_weight=-1, b_weight=1, bias=False, standardize=True),
        causal_mechanism=NeuralNetMechanism(),
        invertible_function=lambda x: x**3,
    )

    X_pnl, _ = pnl.sample()
    plt.scatter(X_pnl[:, 0], X_pnl[:, -1], s=3)
    plt.savefig("./pnl_nn.png")
    plt.close("all")

    pnl = PostNonlinearModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(2, p_edge=1),
        noise_generator=MLPNoise(a_weight=-1, b_weight=1, bias=False, standardize=True),
        causal_mechanism=NeuralNetMechanism(),
        invertible_function=lambda x: x**3,
    )

    X_pnl, _ = pnl.sample()
    plt.scatter(X_pnl[:, 0], X_pnl[:, -1], s=3)
    plt.savefig("./pnl_nn_2.png")
    plt.close("all")


if __name__ == "__main__":
    main()

