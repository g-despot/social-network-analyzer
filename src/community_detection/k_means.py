import logging
import matplotlib.pyplot as plt
import src.embedding as embedding
import src.utils as utils
import time
from sklearn.cluster import KMeans


logger = logging.getLogger('sna')


def detect(graph, number_of_communities, args):
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

    sse = []
    offset_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8]
    for offset in offset_list:
        kmeans = KMeans(
            init="random",
            n_clusters=int(number_of_communities * offset),
            n_init=10,
            max_iter=300
        )

        kmeans.fit(embeddings_list)
        sse.append(kmeans.inertia_)

        #logger.info(f'\nOffset: {offset}')

        # Sum of squared distances of samples to their closest cluster center
        #logger.info(f'Inertia: {kmeans.inertia_}')

        # Number of iterations required to converge
        #logger.info(f'Iterations: {kmeans.n_iter_}')

    plt.plot(offset_list, sse, label='Sum of the squared error')
    plt.legend()
    plt.xticks(offset_list)
    plt.xlabel("Adjusted number of clusters")
    plt.show()
