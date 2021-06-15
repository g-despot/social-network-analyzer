import argparse
import logging
import networkx as nx
import random
from collections import Counter
from typing import Dict


logger = logging.getLogger('sna')


def choose_neighbor(neighbors: list,
                    labels: Dict[int, int]) -> int:
    """
    Choose a neighbor from a propagation starting node.

    Args:
        neigbours (list): List of neighbors.
        labels (Dict): Dictionary with community labels as values and nodes as keys.

    Returns:
        (int): Community label of the node.
    """

    scores = {}
    for neighbor in neighbors:
        neighbor_label = labels[neighbor]
        if neighbor_label in scores.keys():
            scores[neighbor_label] = scores[neighbor_label] + 1
        else:
            scores[neighbor_label] = 1

    top = []
    for key, val in scores.items():
        if val == max(scores.values()):
            top.append(key)

    return random.sample(top, 1)[0]


def propagation_step(graph: nx.classes.graph.Graph,
                     nodes: list,
                     communities: Dict[int, int]) -> tuple:
    """
    One step of the label propagation algorithm.

    Args:
        graph (networkx.classes.graph.Graph): The graph to be clustered.
        nodes (list): List of nodes.
        labels (Dict[int, int]): Dictionary with community labels as values and nodes as keys.

    Returns:
        (Dict): Dictionary with community labels as values and nodes as keys.
    """

    random.shuffle(nodes)
    next_communities = {}

    for node in nodes:
        neighbors = nx.neighbors(graph, node)
        pick = choose_neighbor(neighbors, communities)
        next_communities[node] = pick

    return next_communities


def detect(graph: nx.classes.graph.Graph,
           args: argparse.Namespace) -> Dict[int, int]:
    """
    Run the label propagation algorithm.

    Args:
        graph (networkx.classes.graph.Graph): A NetworkX graph object.
        args (argparse.Namespace): The provided application arguments.

    Returns:
        number_of_communities (int): Number of communities in the graph.
    """

    nodes = []
    for node in graph.nodes():
        nodes.append(node)

    communities = {}
    for i, node in enumerate(graph.nodes()):
        communities[node] = i

    for _ in range(args.iter):
        communities = propagation_step(graph, nodes, communities)

    number_of_communities = len(Counter(communities.values()).keys())
    logger.info(f'\nNumber of communities: {number_of_communities}')

    return number_of_communities
