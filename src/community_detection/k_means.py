import argparse
import logging
import matplotlib.pyplot as plt
import networkx as nx
import src.embedding_algorithms.embedding as embedding
import src.utils as utils
import time
from kneed import KneeLocator
from sklearn.cluster import KMeans


logger = logging.getLogger('sna')


def detect(graph: nx.classes.graph.Graph,
           number_of_communities: int,
           args: argparse.Namespace) -> tuple:
    """Perform the k means clustering algorithm on the node embeddings.

    Args:
        graph (networkx.classes.graph.Graph): A NetworkX graph object.
        number_of_communities (int): The number of communities in the graph.
        args (argparse.Namespace): The provided application arguments.

    Returns:
        number_of_communities (int): Number of communities in the graph.
    """

    if args.embed:
        logger.info(f'\nEmbedding algorithm started.')
        start = time.time()

        embedding.create_embedding(args, graph)
        time_diff = time.time() - start
        logger.info(
            f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embeddings_dict = utils.load_embedding(args.output)
    logger.info(f'\nEmbedding loaded.')

    embeddings_list = list(embeddings_dict.values())

    sse_list = []
    multipliers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
    for offset in multipliers:
        kmeans = KMeans(
            init="random",
            n_clusters=int(number_of_communities * offset),
            n_init=10,
            max_iter=300
        )

        kmeans.fit(embeddings_list)
        sse_list.append(kmeans.inertia_)

        # Multiplier for calculating the adjusted number of clusters
        logger.info(f'\nMultiplier: {offset}')

        # Sum of squared distances of samples to their closest cluster center
        logger.info(f'Inertia: {kmeans.inertia_}')

        # Number of iterations required to converge
        logger.info(f'Iterations: {kmeans.n_iter_}')

    kl = KneeLocator(
        multipliers, sse_list, curve="convex", direction="decreasing"
    )
    logger.info(f'\nElbow point in the SSE curve: {kl.elbow}')

    if args.visuals:
        plt.plot(multipliers, sse_list, label='Sum of the squared error (SSE)')
        plt.legend()
        plt.xticks(multipliers)
        plt.xlabel("Adjusted number of clusters")
        plt.show()

    return kl.elbow, multipliers
