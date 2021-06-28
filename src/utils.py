import argparse
import logging
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
from random import sample
from sklearn.manifold import TSNE
from tqdm import tqdm
from typing import Dict


logger = logging.getLogger('sna')


def load_graph(is_weighted: bool,
               is_directed: bool,
               file_path: str) -> nx.classes.graph.Graph:
    """Load the graph from a file in the input directory.

    Args:
        is_weighted (boolean): Denotes if the graph should be weighted.
        is_directed (boolean): Denotes if the graph should be directed.
        file_path (str): Path to the file with the input graph.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    if is_weighted:
        G = nx.read_gpickle(file_path)
    else:
        G = nx.read_gpickle(file_path)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if is_directed:
        G = G.to_undirected()

    return G


def transform_graph_from_multiple_files(args: argparse.Namespace) -> nx.classes.graph.Graph:
    """Load the graph from multiple files and save it in pickle format to the input directory.

    Args:
        args (argparse.Namespace): The provided application arguments.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    edges = pd.read_csv(args.input_edges, sep=',')
    G = nx.from_pandas_edgelist(edges, args.column_one, args.column_two)

    nodes = pd.read_csv(args.input_nodes, sep=',')
    nx.set_node_attributes(G, pd.Series(
        nodes.ml_target, index=nodes.id).to_dict(), args.node_ml_target)
    nx.set_node_attributes(G, pd.Series(
        nodes.id, index=nodes.id).to_dict(), 'id')

    nx.write_gpickle(G, args.output)

    return G


def transform_graph_from_adjacency_list(args: argparse.Namespace) -> nx.classes.graph.Graph:
    """Load the graph from an adjacency list and save it in pickle format to the input directory.

    Args:
        args (argparse.Namespace): The provided application arguments.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    edges = pd.read_csv(args.input_edges, sep=',')
    G = nx.from_pandas_edgelist(edges, args.column_one, args.column_two)

    nx.write_gpickle(G, args.output)

    return G


def sample_graph(args: argparse.Namespace) -> nx.classes.graph.Graph:
    """Load the graph in pickle format, sample the new graph and save it to the input directory.

    Args:
        args (argparse.Namespace): The provided application arguments.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    G = nx.read_gpickle(args.input)
    H = G.copy()
    samples = sample(list(G.nodes()), 10000)

    for n in tqdm(H):
        if n not in samples:
            G.remove_node(n)

    nx.write_gpickle(G, args.output)

    return G


def get_labels(G: nx.classes.graph.Graph,
               ml_target: str) -> Dict:
    """Get the all the node labels from a graph with the ML target as dictionary values.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        ml_target (str): The node ML target label.

    Returns:
        labels (networkx.classes.graph.Graph): A dictionary of node ids as keys and the ML target label as values.
    """

    labels = {}
    for n in G.nodes(data=True):
        labels[n[1]['id']] = n[1][ml_target]
    return labels


def print_graph_info(G: nx.classes.graph.Graph,
                     graph_name: str) -> None:
    """Print information about the graph.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        graph_name (str): The name of the graph.
    """

    number_of_nodes = nx.number_of_nodes(G)
    number_of_edges = nx.number_of_edges(G)
    density = nx.density(G)

    logger.info(f'\nInformation about the {graph_name}')
    logger.info(
        f'Number of nodes: {number_of_nodes}\tNumber of edges: {number_of_edges}\tDensity: {density}\n')


def load_embedding(file_path: str) -> Dict:
    """Load the node embeddings from a file.

    Args:
        file_path (str): Path to the file with the node embeddings.

    Results:
        embedding_dict (dict): A dictionary of node embedding vectors with nodes as keys.
    """

    embedding_dict = {}
    first_line = True
    with open(file_path) as f:
        for line in f:
            if first_line:
                first_line = False
                continue
            vector = [float(i) for i in line.strip().split()]
            embedding_dict[vector[0]] = vector[1:]
        f.close()

    return embedding_dict


def str2bool(argument: str) -> bool:
    """Transform a string argument to a boolean value.

    Args:
        argument (str): String argument that represents a boolean value.

    Results:
        (bool): A boolean value.
    """

    if isinstance(argument, bool):
        return argument
    if argument.lower() in ('true', 't'):
        return True
    elif argument.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'The argument must be a boolean value.')


def calculate_node_degrees(G: nx.classes.graph.Graph,
                           adj_matrix: scipy.sparse) -> Dict:
    """Calculate the degree of every node.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        adj_matrix (scipy.sparse): Graph adjacency matrix.

    Returns:
        degree_dict (networkx.classes.graph.Graph): A dictionary of node ids as keys and degrees as values.
    """

    degree_dict = {}
    B = adj_matrix.sum(axis=1)
    for cnt, node_id in enumerate(list(G.nodes())):
        degree_dict[node_id] = B[cnt, 0]
    return degree_dict


def visualize_embeddings(embeddings, node_targets):
    """Calculate the degree of every node.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        adj_matrix (scipy.sparse): Graph adjacency matrix.

    Returns:
        degree_dict (networkx.classes.graph.Graph): A dictionary of node ids as keys and degrees as values.
    """

    tsne = TSNE(n_components=2)
    two_dimensional_embeddings = tsne.fit_transform(embeddings)

    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colors = [label_map[target] for target in node_targets]

    plt.scatter(
        two_dimensional_embeddings[:, 0],
        two_dimensional_embeddings[:, 1],
        c=node_colors,
        cmap="jet",
        alpha=0.7,
    )

    plt.show()
