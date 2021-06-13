import argparse
import logging
import src.utils as utils
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path


logger = logging.getLogger('prepare-data')
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

    parser.add_argument('--input-edges', default='git/git_edges.csv',
                        help='Graph input edges in the /input directory.')
    parser.add_argument('--input-nodes', default='git/git_target.csv',
                        help='Graph input nodes in the /input directory.')
    parser.add_argument('--output', default='git/git.gpickle',
                        help='File name to save the graph in the /input directory.')
    parser.add_argument('--column-one', default='id_1',
                        help='Name of the first edge id column.')
    parser.add_argument('--column-two', default='id_2',
                        help='Name of the second edge id column.')
    parser.add_argument('--node-name', default='', help='Node name.')
    parser.add_argument('--node-ml-target',
                        default='ml_target',
                        help='Node machine learning target label.')
    parser.add_argument('--format', default='adjacency-list',
                        choices=['adjacency-list', 'nodes-edges'],
                        help='Format of the graph being transformed.')

    args = parser.parse_args()

    args.input_edges = _here.parent.joinpath("input/" + args.input_edges)
    args.input_nodes = _here.parent.joinpath("input/" + args.input_nodes)
    args.output = _here.parent.joinpath("input/" + args.output)

    return args


def main():
    """Utility program for preparing input files."""

    args = parse_args()

    logger.info(f'\nTransformation started.')
    start = time.time()

    if args.format == 'adjacency-list':
        utils.transform_graph_from_adjacency_list(args)

    if args.format == 'nodes-edges':
        utils.transform_graph_from_multiple(args)

    time_diff = time.time() - start
    logger.info(
        f'\nTransformation finished in {time_diff:.2f} seconds.')
    logger.info(f'\nGraph saved as {args.output}')


if __name__ == "__main__":
    """Main entry point of the utility program."""

    main()
