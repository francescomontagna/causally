import pytest
import random
import math
import numpy as np
import networkx as nx

from datasets.random_graphs import ErdosRenyi, BarabasiAlbert, GaussianRandomPartition

seed = 42
random.seed(seed)
np.random.seed(seed)


# ******************* Test acyclicity and order ******************* #
def test_acyclicity_nontrivial_order():
    """Test that the returned graph acyclicity and nontrivial topological order
    """
    def num_errors(order, adj):
        err = 0
        for i in range(len(order)):
            err += adj[order[i+1:], order[i]].sum()
        return err

    for num_nodes in range(10, 100):
        graph_type = np.random.choice(["ER", "SF", "GRP"])
        if graph_type == "ER":
            generator = ErdosRenyi(
                num_nodes=num_nodes,
                expected_degree=int(np.random.choice(range(1,4)))
            )
        if graph_type == "SF":
            generator = BarabasiAlbert(
                num_nodes=num_nodes,
                expected_degree=int(np.random.choice(range(1,4)))
            )
        if graph_type == "GRP":
            generator = GaussianRandomPartition(
                num_nodes=num_nodes,
                p_in=0.4,
                p_out=0.1,
                n_clusters=math.floor(num_nodes/4) # ~4 nodes per cluster
            )

        A = generator.get_random_graph()

        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(A, create_using=nx.DiGraph)),\
        "The generated random graph is not acyclic! No topological order can be defined"

        trivial_order = range(num_nodes)
        assert num_errors(trivial_order, A) > 0, \
        f"The adjacency matrix of {graph_type} graph has trivial order."


# ******************* Test expected number of nodes ******************* #
def test_er_has_expected_number_of_nodes():
    """Test ErdosRenyi class generated graphs with the required number of nodes.
    """
    nodes_values = [5, 10, 20, 50]
    for num_nodes in nodes_values:
        generator = ErdosRenyi(
            num_nodes=num_nodes,
            p_edge=0.4
        )
        A = generator.get_random_graph(seed)
        assert num_nodes == A.shape[0], "ER graph generated with wrong number of nodes"\
        f"Expected {num_nodes} nodes, got {A.shape[0]} instead."


def test_sf_has_expected_number_of_nodes():
    """Test BarabasiAlbert class generated graphs with the required number of nodes.
    """
    nodes_values = [5, 10, 20, 50]
    for num_nodes in nodes_values:
        generator = BarabasiAlbert(
            num_nodes=num_nodes,
            expected_degree=1
        )
        A = generator.get_random_graph(seed)
        assert num_nodes == A.shape[0], "SF graph generated with wrong number of nodes"\
        f"Expected {num_nodes} nodes, got {A.shape[0]} instead."


def test_grp_has_expected_number_of_nodes():
    """Test GaussianRandomPartition class generated graphs with the required number of nodes.
    """
    nodes_values = [5, 10, 20, 50]
    for num_nodes in nodes_values:
        generator = GaussianRandomPartition(
            num_nodes=num_nodes,
            p_in=0.05,
            p_out=0.4,
            n_clusters=2
        )
        A = generator.get_random_graph()
        assert num_nodes == A.shape[0], "GRP graph generated with wrong number of nodes"\
        f"Expected {num_nodes} nodes, got {A.shape[0]} instead."


# ******************* Test ER graphs generation ******************* #

def test_er_small_sparse_number_of_edges_and_degree_regularity():
    """Test small sparse ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 5
        - maximum number of edges: <=5 (probability of 5 or more edges is 0.01)
        - average number of edges: nn {2, 3}
        - regularity of edges: on average max out and income degree difference is less than 1
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 5
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        p_edge=0.1
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] +  [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.mean(logs["num_edges"]) <= 3
    assert np.max(logs["num_edges"]) <= 5
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 1 # regularity condition


def test_er_small_dense_number_of_edges_and_degree_regularity():
    """Test small dense ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 5
        - edge probability: 0.4
        - maximum number of edges: >= 8 (probability of 8 or more edges is 0.2)
        - average number of edges: 5 +- 1
        - regularity of edges: on average max out and income degree difference is less than 1
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 5
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        p_edge=0.4
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.mean(logs["num_edges"]) >= 4 and np.mean(logs["num_edges"]) <= 6 # mean
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 1 # regularity


def test_er_medium_sparse_number_of_edges_and_degree_regularity():
    """Test medium sparse ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 10
        - expected_degree: 1
        - minimum number of edges: 9
        - maximum number of edges: 11
        - regularity of edges: on average max out and income degree difference is less than 1
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 10
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        expected_degree=1
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.max(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.max(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # max edges
    assert np.min(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.min(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 1 # regularity


def test_er_medium_dense_number_of_edges_and_degree_regularity():
    """Test medium dense ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 10
        - expected_degree: 2
        - minimum number of edges: 19
        - maximum number of edges: 21
        - regularity of edges: on average max out and income degree difference is less than 1.5
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 10
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        expected_degree=2
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.max(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.max(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # max edges
    assert np.min(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.min(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 1.5 # regularity


def test_er_large_sparse_number_of_edges_and_degree_regularity():
    """Test large sparse ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 50
        - expected_degree: 2
        - minimum number of edges: 99
        - maximum number of edges: 101
        - regularity of edges: on average max out and income degree difference is less than 1.5
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 50
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        expected_degree=2
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.max(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.max(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # max edges
    assert np.min(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.min(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 1.5 # regularity condition


def test_er_large_dense_number_of_edges_and_degree_regularity():
    """Test large dense ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes: 50
        - expected degree: 8
        - mean number of edges: 399
        - maximum number of edges: 401
        - regularity of edges: on average max out and income degree difference is less than 2.5
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 50
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    
    # Construct ER generator
    generator = ErdosRenyi(
        num_nodes=num_nodes,
        expected_degree=8
    )

    # Generate graphs and store logs
    for seed in run_seeds:
        A = generator.get_random_graph(seed)
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert np.max(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.max(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # max edges
    assert np.min(logs["num_edges"]) >= generator.expected_degree*num_nodes-1 \
        and np.min(logs["num_edges"]) <= generator.expected_degree*num_nodes+1 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 2.5 # regularity


# ******************* Test SF graphs generation ******************* #
def test_sf_medium_sparse():
    """Test generation of medium sparse SF graphs (10 nodes).
    Test specifics:
        - number of nodes: 10
        - density parameter: 2
        - mean number of edges: 10 +- 1
        - maximum number of edges: 10 +- 1
        - low maximum of input degree: <= 2
        - large minimum of max_output_degree: <= 4
        - large mean distance between max input and output degree: >= 3
    """ 
    n_graphs = 100
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "num_nodes" : [],
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    for seed in run_seeds:
        generator = VanillaGenerator(
            graph_type="SF",
            num_nodes = 10,
            graph_size = "medium",
            graph_density="sparse",
            num_samples=10,
            noise_distr="Gauss",
            noise_std_support=(.5, 1.),
            seed=seed
        )
        generator.simulate_dag()
        A = generator.adjacency
        num_nodes = A.shape[0]
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["num_nodes"] = logs["num_nodes"] + [num_nodes]
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert generator.get_density_param() == 1
    assert min(logs["num_nodes"]) == max(logs["num_nodes"]) & max(logs["num_nodes"]) == 10
    assert np.mean(logs["num_edges"]) >= 9 and np.mean(logs["num_edges"]) <= 11 # mean edges
    assert np.max(logs["num_edges"]) >= 9 and np.max(logs["num_edges"]) <= 11 # max edges
    assert np.min(logs["num_edges"]) >= 9 and np.min(logs["num_edges"]) <= 11 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) >= 3 # irregular graph
    assert np.mean(np.array(logs["max_income_degree"])) <= 2 # low input degree
    assert np.mean(np.array(logs["max_outcome_degree"])) >= 4 # large output degree


def test_sf_large_sparse():
    """Test generation of large sparse SF graphs (30 nodes).
    Test specifics:
        - number of nodes: 10
        - density parameter: 2
        - mean number of edges: 10 +- 1
        - maximum number of edges: 10 +- 1
        - low maximum of input degree: <= 2
        - large minimum of max_output_degree: >= 6
        - large mean distance between max input and output degree: >= 3
    """ 
    n_graphs = 100
    run_seeds = np.random.randint(10000, size=n_graphs)
    logs = {
        "num_nodes" : [],
        "max_income_degree" : [],
        "max_outcome_degree" : [],
        "num_edges" : [],
    }
    for seed in run_seeds:
        generator = VanillaGenerator(
            graph_type="SF",
            num_nodes = 30,
            graph_size = "large",
            graph_density="sparse",
            num_samples=10,
            noise_distr="Gauss",
            noise_std_support=(.5, 1.),
            seed=seed
        )
        generator.simulate_dag()
        A = generator.adjacency
        num_nodes = A.shape[0]
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["num_nodes"] = logs["num_nodes"] + [num_nodes]
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    assert generator.get_density_param() == 1
    assert min(logs["num_nodes"]) == max(logs["num_nodes"]) & max(logs["num_nodes"]) == 30
    assert np.mean(logs["num_edges"]) >= 29 and np.mean(logs["num_edges"]) <= 31 # mean edges
    assert np.max(logs["num_edges"]) >= 29 and np.max(logs["num_edges"]) <= 31 # max edges
    assert np.min(logs["num_edges"]) >= 29 and np.min(logs["num_edges"]) <= 31 # min edges
    assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) >= 5 # irregular graph
    assert np.mean(np.array(logs["max_income_degree"])) <= 1 # low input degree
    assert np.mean(np.array(logs["max_outcome_degree"])) >= 10 # large output degree


##################### Test GRP graphs generation #####################
def test_gpr_sample_number_of_clusters():
    p_in = 0.4
    p_out = 0.04
    
    # 10 nodes
    d = 10
    grp = GaussianRandomPartition(d, p_in, p_out)
    assert grp._sample_number_of_clusters() == 2, f"Medium graphs with {d} nodes must have 2 clusters!"

    # 20 nodes 
    d = 20
    grp = GaussianRandomPartition(d, p_in, p_out)
    for _ in range(10):
        n_clusters = grp._sample_number_of_clusters()
        assert n_clusters >= 3 and n_clusters <= 5, f"Medium graphs with {d} nodes must have n_clusters in [3, 4, 5]!"

    # 30 nodes 
    d = 30
    grp = GaussianRandomPartition(d, p_in, p_out)
    for _ in range(10):
        n_clusters = grp._sample_number_of_clusters()
        assert n_clusters >= 3 and n_clusters <= 5, f"Medium graphs with {d} nodes must have n_clusters in [3, 4, 5]!"

    # 50 nodes 
    d = 50
    grp = GaussianRandomPartition(d, p_in, p_out)
    for _ in range(10):
        n_clusters = grp._sample_number_of_clusters()
        assert n_clusters >= 4 and n_clusters <= 6, f"Medium graphs with {d} nodes must have n_clusters in [4, 5, 6]!"


def test_sample_cluster_sizes():
    p_in = 0.4
    p_out = 0.04
    
    for d in [10, 20, 30, 50]:
        for _ in range(10):
            grp = GaussianRandomPartition(d, p_in, p_out)
            n_clusters = grp._sample_number_of_clusters()
            assert np.min(grp._sample_cluster_sizes(n_clusters)) >= 3, f"Unexpected behaviour! There is a cluster with less than 3 nodes!"


def test_clusters_er_behaviour():
    """Check that clusters are approimatively regular graphs
    """
    import torch
    torch.manual_seed(seed)
    p_in = 0.4
    p_out = 0.04
    
    for num_nodes in [10, 20, 30, 50]:
        logs = {
            "num_nodes" : [],
            "max_income_degree" : [],
            "max_outcome_degree" : [],
            "num_edges" : [],
        }

        grp = GaussianRandomPartition(num_nodes, p_in, p_out)
        n_clusters = grp._sample_number_of_clusters()
        clusters_size = grp._sample_cluster_sizes(n_clusters)
        for c in clusters_size:
            A = grp._sample_er_cluster(cluster_size=c)
            max_income_degree = np.max(A.sum(axis=0))
            max_outcome_degree = np.max(A.sum(axis=1))
            logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
            logs["max_outcome_degree"] = logs["max_outcome_degree"] +  [max_outcome_degree]

        assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 2,\
        "ER clusters showing irregular behaviour!"


def test_sparsity_between_clusters():
    """Check the the disjoint union works as expetced
    1. Mark the nodes in cllusters before the union.
    2. Check connection in the same cluster, between different clusters.
    """
    p_in = 0.4
    p_out = {
        10 : 0.06,
        20 : 0.06,
        30 : 0.03,
        50 : 0.03
    }
    
    for num_nodes in [10, 20, 30, 50]:
        grp = GaussianRandomPartition(num_nodes, p_in, p_out[num_nodes])
        n_clusters = grp._sample_number_of_clusters()
        clusters_size = grp._sample_cluster_sizes(n_clusters)
        
        # Initialize with the first cluster and remove it from the list
        A = grp._sample_er_cluster(clusters_size[0])
        clusters_size = np.delete(clusters_size, [0])
        m = A.shape[0]
        for c in clusters_size:
            A = grp._disjoint_union(A, c)
            n = A.shape[0]

            # Check that between clusters connections are sparse (Upper trianguler matrix)
            A_between_clusters = A[:m, m:]
            assert A_between_clusters.sum() <= 2
            m=n