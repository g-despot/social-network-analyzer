import argparse
import logging
import networkx as nx
import src.community_detection.community_detection as community_detection
import src.link_prediction.link_prediction as link_prediction
import src.node_classification.node_classification as node_classification
import src.utils as utils
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


logger = logging.getLogger('sna')
logging.getLogger().addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

_here = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    """Parse the application arguments.

    Returns:
        args (argparse.Namespace): Application arguments.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')

    parser.add_argument('--input', default='facebook_pages.edgelist',
                        help='Input graph in edge list format in the /input directory.')
    parser.add_argument('--output', default='facebook_pages.txt',
                        help='File name to save the graph embeddings in the /embeddings directory.')
    parser.add_argument('--results', default='facebook_pages.csv',
                        help='File name to save the evaluation results in the /results directory.')
    parser.add_argument('--dimensions', default=128, type=int,
                        help='Dimensionality of the word vectors. (default: 128)')
    parser.add_argument('--walk-length', default=64, type=int,
                        help='The number of nodes in each walk. (default: 64)')
    parser.add_argument('--num-walks', default=32, type=int,
                        help='Number of walks from each node. (default: 32)')
    parser.add_argument('--p', default=2.0, type=float,
                        help='Node2vec return parameter p. (default: 2)')
    parser.add_argument('--q', default=1.0, type=float,
                        help='Node2vec in-out parameter q. (default: 1)')
    parser.add_argument('--workers', default=1, type=int,
                        help='Number of worker threads to train the model. (default: 1)')
    parser.add_argument('--seed', default=0, type=int,
                        help='A seed for the random number generator. (default: 0)')
    parser.add_argument('--test-percentage', default=0.1, type=float,
                        help='Percentage of graph edges that should be used for testing classifiers.')
    parser.add_argument('--train-percentage', default=0.1, type=float,
                        help='Percentage of graph edges that should be used for training classifiers.')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Maximum distance between the current and predicted word within a sentence. (default: 10)')
    parser.add_argument('--weighted', type=bool, default=False,
                        help='Denotes if the graph is weighted. (default: False)')
    parser.add_argument('--directed', type=bool, default=False,
                        help='Denotes if the graph is directed. (default: False)')
    parser.add_argument('--iter', default=1, type=int,
                        help='Number of iterations (epochs) over the corpus.')
    parser.add_argument('--method', default='node2vec_eliorc',
                        choices=['node2vec_snap',
                                 'node2vec_eliorc',
                                 'node2vec_custom',
                                 'deepwalk_phanein',
                                 'deepwalk_custom',
                                 'label-propagation',
                                 'girvan-newman'],
                        help='The graph embedding algorithm and specific implementation.')
    parser.add_argument('--classifier', default='logisticalregression',
                        choices=['logisticalregression',
                                 'randomforest',
                                 'gradientboost', ],
                        help='The classifier for evaluation.')
    parser.add_argument('--evaluation', default='node-classification',
                        choices=['link-prediction',
                                 'node-classification',
                                 'community-detection'],
                        help='Social network analysis technique to be used.')
    parser.add_argument('--embed', type=utils.str2bool, nargs='?',
                        const=True, default=False,
                        help='Denotes if the embedding should be calculated or loaded from an existing file.')
    parser.add_argument('--node-ml-target',
                        default='ml_target',
                        help='The node ML target label.')
    parser.add_argument('--k', default=None, type=int,
                        help='Number of node samples to estimate betweenness. (default: None)')
    parser.add_argument('--converging', default=10, type=int,
                        help='When to cut off the Girvan-Newman algorithm if modularity is decreasing. (default: 10)')
    args = parser.parse_args()

    args.dataset = args.input
    args.input = _here.parent.joinpath("input/" + args.input)
    args.output = _here.parent.joinpath("embeddings/" + args.output)
    if args.results:
        args.results = _here.parent.joinpath("results/" + args.results)

    return args


def main():
    """Load the graph and perform the selected social network analysis method."""

    args = parse_args()

    #graph = utils.load_graph(args.weighted, args.directed, args.input)
    graph = nx.karate_club_graph()
    utils.print_graph_info(graph, "original graph")

    graph.remove_nodes_from(list(nx.isolates(graph)))
    utils.print_graph_info(graph, "graph without isolates")

    if args.evaluation == 'node-classification':
        logger.info(f'\nNode classification procedure')
        node_classification.run(graph, args)

    elif args.evaluation == 'link-prediction':
        logger.info(f'\nLink prediction procedure')
        link_prediction.run(graph, args)

    elif args.evaluation == 'community-detection':
        community_detection.run(graph, args)

    else:
        raise ValueError(f'Invalid analysis method: {args.method}')


if __name__ == "__main__":
    """Main entry point of the application."""

    main()
