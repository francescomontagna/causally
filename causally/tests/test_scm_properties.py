import numpy as np
import random

from causally.graph.random_graphs import ErdosRenyi
from causally.scm.scm_properties import ConfoundedModel, UnfaithfulModel

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

###################### Test UnfaithfulModel ######################
def test_given_fully_connected_matrix_when_model_unfaithful_then_path_cancelling_in_moral_triplets():
    adjacency = np.triu(np.ones((5, 5)), k=1)
    unfaithful_model = UnfaithfulModel(p_unfaithful=1)
    unfaithful_adj, _ = unfaithful_model.unfaithful_adjacency(adjacency)

    target_adj = np.array([
        [0,1,0,0,0],
        [0,0,1,1,1],
        [0,0,0,1,0],
        [0,0,0,0,1],
        [0,0,0,0,0]
    ])
    assert np.allclose(unfaithful_adj, target_adj)


###################### Test ConfoundedModel ######################
def test_given_p_confounded_when_generating_graphs_then_rate_of_confounded_pairs_is_p_confounded():
    p_confounder = 0.2
    num_nodes = 20
    p_edge = 0.4
    graph_generator = ErdosRenyi(num_nodes, p_edge=p_edge)
    confounder_model = ConfoundedModel(p_confounder)

    # sum_confounded_pairs = 0
    for _ in range(10):
        adjacency = graph_generator.get_random_graph()
        confounded_adj = confounder_model.confound_adjacency(adjacency)
        confounders_matrix = confounded_adj[num_nodes:, num_nodes:]
        n_direct_confounders = 0
        for confounder in range(num_nodes):
            n_direct_confounders += int(confounders_matrix[confounder, :].sum())
        
        number_of_pairs = num_nodes*(num_nodes-1)
        assert abs(n_direct_confounders/number_of_pairs - p_confounder) < 0.05,\
            f"Expected rate of confounders 0.2 +- 0.05, instead got: "\
            f"{abs(n_direct_confounders/number_of_pairs)}"