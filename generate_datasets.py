import argparse
import os
import shutil
import causally.datasets.random_graphs as rg
import causally.datasets.random_noises as rn 
import causally.datasets.causal_mechanisms as cm
import causally.datasets.scm as scm

from causally.utils.data import generate_and_store_dataset

WORKSPACE = os.path.join(os.getcwd(), "..")

def args_sanity_check(args):
    if args.p_density is None or args.m_density is None:
        ValueError("One argument between `-p` and `-m` must be unassigned.")

    if args.m_density is None and args.graph_type == "SF":
        raise ValueError("SF graphs can not accept `-p` as density parameter. Provide a valid value for `-m`")

    # TODO: add more!



if __name__ == "__main__":
    # TODO: this script does not allow for mixed configurations! Need to fix
    # TODO: replace command line arguments with json folder reading
    parser = argparse.ArgumentParser(description="Datasets generation and storage")

    parser.add_argument(
        "--seed",
        '-s',
        default=42, 
        type=int, 
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        "--n_datasets",
        default=100, 
        type=int, 
        help="Number graphs samples in the dataset."
    )

    parser.add_argument(
        "--graph_n_samples",
        default=1000, 
        type=int, 
        help="Number samples per graph."
    )

    parser.add_argument(
        "--graph_type",
        default="ER",
        type=str,
        help="Algorithm for generation of synthetic graphs. Accepted values are ['ER', 'SF', 'GRP']."
    )

    parser.add_argument(
        "--noise_distr",
        default="gauss",
        type=str,
        help="Distribution of th noise terms. Accepted values are ['gauss', 'gp', 'nn-tanh', 'nn-relu']."
    )

    parser.add_argument(
        "--scm",
        default="anm",
        type=str,
        help="Structural causal model (linear, nonlinear additive, post-nonlinear). Accepted values are ['linear', 'anm', 'pnl']."
    )

    parser.add_argument(
        "--mechanisms",
        "-f",
        default="nn",
        type=str,
        help="Methods for generation of the causal mechanism (gaussian process, neural net). Accepted values are ['gp', 'nn']."
    )

    parser.add_argument(
        "--n_nodes",
        default=5, 
        type=int, 
        help="Number of nodes in the graph."
    )

    parser.add_argument(
        "--p_density",
        "-p",
        default=None,
        type=float, 
        help="`p`density parameter, probability of connecting a pair of nodes.",
    )

    parser.add_argument(
        "--m_density",
        "-m", 
        default=None,
        type=int, 
        help="`m` density parameter, expected degree of each node.",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        default=os.path.join(WORKSPACE, "storage", "data"), 
        type=str, 
        help="Base folder for storage of the data."
    )

    parser.add_argument(
        "--dataset_name",
        "-n",
        type=str,
        help='Name of the dataset. This is used as name of the folder for the dataset storage.',
        required=True
    )

    # Parse and check arguments
    args = parser.parse_args()
    args_sanity_check(args)

    # Create dataset directory, else clear it
    dataset_directory = os.path.join(args.output_folder, args.dataset_name)
    if os.path.exists(dataset_directory):
        shutil.rmtree(dataset_directory)
    os.makedirs(dataset_directory) # create also intermediate directories

    # Sample and store datasets
    for id in range(args.n_datasets):
        data_file = os.path.join(dataset_directory, f"data_{id}.npy")
        groundtruth_file = os.path.join(dataset_directory, f"groundtruth_{id}.npy")
        
        # Noise generator
        if args.noise_distr == 'gauss':
            # Sample noise on CPU to avoid NaN (PyTorch bug: https://discuss.pytorch.org/t/why-am-i-getting-a-nan-in-normal-mu-std-rsample/117401/8)
            noise_generator = rn.Normal(0, 1)
        elif args.noise_distr == "nn":
            noise_generator = rn.MLPNoise()
        else:
            raise ValueError(f"Unsupported noise type {args.noise_distr}.")

        # Graph generator
        if args.graph_type == "ER":
            graph_generator = rg.ErdosRenyi(
                num_nodes=args.n_nodes,
                expected_degree=args.m_density,
                p_edge = args.p_density
            )
        elif args.graph_type == "SF":
            graph_generator = rg.BarabasiAlbert(
                num_nodes=args.n_nodes,
                expected_degree=args.m_density
            )
        elif args.graph_type == "GRP":
            raise ValueError("Currently GRP graph generation is not supported. TODO: fix this")
        

        # Causal mechanism generator
        if args.mechanisms == "nn":
            causal_mechanism = cm.NeuralNetMechanism()
        elif args.mechanisms == "gp":
            causal_mechanism = cm.GaussianProcessMechanism()
        else: 
            raise ValueError(f"Unsupported causal mechanism {args.mechanisms}.")
        

        # Model generator
        if args.scm == "anm":
            model = scm.AdditiveNoiseModel(
                num_samples=args.graph_n_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                causal_mechanism=causal_mechanism,
                seed=args.seed+id
            )
        elif args.scm == "linear":
            model = scm.LinearModel(
                num_samples=args.graph_n_samples,
                graph_generator=graph_generator,
                noise_generator=noise_generator,
                seed=args.seed+id
            )
        elif args.scm == "pnl":
            raise ValueError("Currently PNL graph generation is not supported. TODO: fix this")
        

        generate_and_store_dataset(data_file, groundtruth_file, model)
