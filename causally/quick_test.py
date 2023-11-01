import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets.scm import PostNonlinearModel, AdditiveNoiseModel, LinearModel
from datasets.causal_mechanisms import LinearMechanism, NeuralNetMechanism, GaussianProcessMechanism
from datasets.random_graphs import ErdosRenyi
from datasets.random_noises import MLPNoise, Normal
from datasets.scm_properties import ConfoundedModel, UnfaithfulModel, AutoregressiveModel, MeasurementErrorModel

# Random seeds for reproducibility
SEED=42
np.random.seed(SEED)
torch.random.manual_seed(SEED)

def main():
    num_nodes = 10
    p_edge = 0.4
    model = AdditiveNoiseModel(
        num_samples=1000,
        graph_generator=ErdosRenyi(num_nodes, p_edge=p_edge),
        noise_generator=MLPNoise(a_weight=-1, b_weight=1, bias=False, standardize=True),
        causal_mechanism=NeuralNetMechanism(),
    )

    X, _ = model.sample()
    plt.scatter(X[:, 0], X[:, -1], s=3)
    plt.savefig("./anm.png")
    plt.close("all")

    # Generate with violations
    # model.add_misspecificed_property(UnfaithfulModel(p_unfaithful=0.75))
    # model.add_misspecificed_property(ConfoundedModel(p_confounder=0.2))
    # model.add_misspecificed_property(MeasurementErrorModel(gamma=0.8))
    model.add_misspecificed_property(AutoregressiveModel(order=1))

    X, _ = model.sample()
    plt.scatter(X[:, 0], X[:, -1], s=3)
    plt.savefig("./anm_timino.png")
    plt.close("all")


if __name__ == "__main__":
    main()

