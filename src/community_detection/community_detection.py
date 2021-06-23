import argparse
import csv
import logging
import networkx as nx
import src.community_detection.girvan_newman as girvan_newman
import src.community_detection.k_means as k_means
import src.community_detection.label_propagation as label_propagation
import time
from networkx.algorithms.community.centrality import girvan_newman as girvan_newman_nx
from networkx.algorithms.community.label_propagation import label_propagation_communities as label_propagation_communities_nx


logger = logging.getLogger('sna')


def save_evaluation_results(dataset: str,
                            elbow: float,
                            multipliers: list,
                            number_of_communities: int,
                            file_path: str) -> None:
    """Save the community detection evaluation results in JSON format.

    Args:
        dataset (str): The name of the input dataset.
        elbow (float): The calculated elbow point.
        multipliers (float): The multipliers for calculating alternative cluster sizes.
        number_of_communities (int): The input number of clusters.
        file_path (str): Path to the file with the community detection evaluation results.
    """

    json_results = [dataset,
                    elbow,
                    multipliers,
                    number_of_communities]

    with open(file_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(json_results)


def run(graph: nx.classes.graph.Graph,
        args: argparse.Namespace) -> None:
    """Perform the community detection evaluation.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        args (argparse.Namespace): The provided application arguments.
    """

    number_of_communities = 0
    start = time.time()

    if args.community_method == 'girvan_newman_custom':
        logger.info(f'\nCommunity detection procedure: Girvan-Newman')
        _, number_of_communities = girvan_newman.detect(graph, args)

    elif args.community_method == 'label_propagation_custom':
        logger.info(f'\nCommunity detection procedure: Label propagation')
        _, number_of_communities = label_propagation.detect(graph, args)

    elif args.community_method == 'girvan_newman_nx':
        logger.info(f'\nCommunity detection procedure: Girvan-Newman NetworkX')
        communities = girvan_newman_nx(graph)
        for _ in communities:
            number_of_communities = number_of_communities + 1

    elif args.community_method == 'label_propagation_nx':
        logger.info(
            f'\nCommunity detection procedure: Label propagation NetworkX')
        communities = label_propagation_communities_nx(graph)
        for _ in communities:
            number_of_communities = number_of_communities + 1

    else:
        raise ValueError(
            f'Invalid community detection algorithm: {args.community_method}')

    logger.info(
        f'Initial community detection finished in {time.time() - start:.2f} seconds.')
    logger.info(f'\nNumber of communities: {number_of_communities}')

    if number_of_communities > 0:
        elbow, multipliers = k_means.detect(graph, number_of_communities, args)
        save_evaluation_results(
            args.input, elbow, multipliers, number_of_communities, args.results)
