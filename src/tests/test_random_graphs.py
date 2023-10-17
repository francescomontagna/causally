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
        A = generator.get_random_graph()
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
        A = generator.get_random_graph()
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
def test_given_er_generator_when_sampling_bivariate_with_p_one_then_x_causes_y():
    er_generator = ErdosRenyi(2, p_edge=1)
    A = er_generator.get_random_graph()
    assert A.sum() == 1, f"Expected number of edges 1, got instead {A.sum}"

def test_er_small_sparse_number_of_edges_and_degree_regularity():
    """Test small sparse ER graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes parameter: 5
        - edge probability parameter: 0.1
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
        - number of nodes parameter: 5
        - edge probability parameter: 0.4
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
        - number of nodes parameter: 10
        - expected degree parameter: 1
    """ 
    # Hyperparameters and logging
    n_graphs = 100
    num_nodes = 10
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
        - number of nodes parameter: 10
        - expected degree parameter: 2
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
        - number of nodes parameter: 50
        - expected degree parameter: 2
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
        - number of nodes parameter: 50
        - expected degree parameter: 8
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
    for _ in range(n_graphs):
        A = generator.get_random_graph()
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
def test_sf_medium_sparse_irregular_degree():
    """Test medium sparse SF graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes parameter: 10
        - expected degree parameter: 1
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
    
    # Construct SF generator
    generator = BarabasiAlbert(
        num_nodes=num_nodes,
        expected_degree=1
    )

    # Generate graphs and store logs
    for _ in range(n_graphs):
        A = generator.get_random_graph()
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    avg_in_out_diff = np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"])))
    assert  avg_in_out_diff >= 3, f"Mean difference between max input and output degree: {avg_in_out_diff}."\
    " SF graphs shouldn't be regular!" # irregular graph
    assert np.mean(np.array(logs["max_outcome_degree"])) >= 4 # large output degree


def test_sf_large_sparse_irregular_degree():
    """Test large sparse SF graphs have regular degree and expected number of edges.
    Test specifics:
        - number of nodes parameter: 50
        - expected degree parameter: 2
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
    
    # Construct SF generator
    generator = BarabasiAlbert(
        num_nodes=num_nodes,
        expected_degree=2
    )

    # Generate graphs and store logs
    for _ in range(n_graphs):
        A = generator.get_random_graph()
        num_edges = A.sum()
        max_income_degree = np.max(A.sum(axis=0))
        max_outcome_degree = np.max(A.sum(axis=1))
        logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
        logs["max_outcome_degree"] = logs["max_outcome_degree"] + [max_outcome_degree]
        logs["num_edges"] = logs["num_edges"] + [num_edges]

    avg_in_out_diff = np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"])))
    assert  avg_in_out_diff >= 5, f"Mean difference between max input and output degree: {avg_in_out_diff}."\
    " SF graphs shouldn't be regular!" # irregular graph
    assert np.mean(np.array(logs["max_outcome_degree"])) >= 10 # large output degree


##################### Test GRP graphs generation #####################
def test_gpr_has_expected_number_of_clusters():
    """Test GRP graph has the number of clusters passed as input argument.
    """
    p_in = 0.4
    p_out = 0.04
    nodes_values = [5, 10, 20, 50]
    n_clusters_and_min_size = {
        5: (2, 2),
        10: (3, 3),
        20: (5, 4),
        50: (10, 4),
    }
    
    for num_nodes in nodes_values:
        n_clusters, min_cluster_size = n_clusters_and_min_size[num_nodes]
        generator = GaussianRandomPartition(
            num_nodes, p_in, p_out, n_clusters=n_clusters, min_cluster_size=min_cluster_size
        )
        assert len(generator.size_of_clusters) == n_clusters, "Wrong number of clusters."\
        f" Expected {n_clusters}, got {len(generator.size_of_clusters)} instead."


def test_grp_min_cluster_size():
    """Test GRP graph has the requested minimum cluster size
    """
    p_in = 0.4
    p_out = 0.04
    nodes_values = [5, 10, 20, 50]
    n_clusters_and_min_size = {
        5: (2, 2),
        10: (3, 3),
        20: (5, 4),
        50: (10, 4),
    }

    for num_nodes in nodes_values:
        n_clusters, min_cluster_size = n_clusters_and_min_size[num_nodes]
        generator = GaussianRandomPartition(
            num_nodes, p_in, p_out, n_clusters=n_clusters, min_cluster_size=min_cluster_size
        )
        smallest_cluster = min(generator.size_of_clusters)
        assert smallest_cluster >= min_cluster_size, \
        f"Min cluster size {smallest_cluster} is too large! Expected larger or equal {min_cluster_size}" 



def test_grp_clusters_er_regular_degree_behaviour():
    """Check that clusters are approximatively regular graphs.
    """
    p_in = 0.4
    p_out = 0.04
    nodes_values = [5, 10, 20, 50]
    n_clusters_and_min_size = {
        5: (2, 2),
        10: (3, 3),
        20: (5, 4),
        50: (10, 4),
    }
    logs = {
        "max_income_degree" : [],
        "max_outcome_degree" : [],
    }

    for num_nodes in nodes_values:
        n_clusters, min_cluster_size = n_clusters_and_min_size[num_nodes]
        generator = GaussianRandomPartition(
            num_nodes, p_in, p_out, n_clusters=n_clusters, min_cluster_size=min_cluster_size
        )
        for _ in generator.size_of_clusters:
            A = generator.get_random_graph(seed=num_nodes)
            max_income_degree = np.max(A.sum(axis=0))
            max_outcome_degree = np.max(A.sum(axis=1))
            logs["max_income_degree"] = logs["max_income_degree"] + [max_income_degree]
            logs["max_outcome_degree"] = logs["max_outcome_degree"] +  [max_outcome_degree]

        assert np.mean(abs(np.array(logs["max_income_degree"]) - np.array(logs["max_outcome_degree"]))) <= 2,\
        "ER clusters showing irregular behaviour!"


def test_sparsity_between_clusters():
    """Test that connection between different clusters are sparse
    """
    nodes_values = [10, 20, 50]
    p_in = 0.4
    p_out = {
        10 : 0.06,
        20 : 0.06,
        50 : 0.03
    }
    n_clusters_and_min_size = {
        10: (3, 3),
        20: (5, 4),
        50: (10, 4),
    }
    
    for num_nodes in nodes_values:
        n_clusters, min_cluster_size = n_clusters_and_min_size[num_nodes]
        generator = GaussianRandomPartition(
            num_nodes, p_in, p_out[num_nodes], n_clusters=n_clusters, min_cluster_size=min_cluster_size
        )
        clusters_size = generator.size_of_clusters
        
        # Initialize with the first cluster and remove it from the list
        A = generator._sample_er_cluster(clusters_size[0])
        clusters_size = np.delete(clusters_size, [0])
        m = A.shape[0]
        for c in clusters_size:
            A = generator._disjoint_union(A, c)
            n = A.shape[0]

            # Check that between clusters connections are sparse (Upper trianguler matrix)
            A_between_clusters = A[:m, m:]
            assert A_between_clusters.sum() <= 2
            m=n