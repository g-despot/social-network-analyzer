import logging
import src.embedding as embedding
import src.node_classification.evaluation as evaluation
import src.utils as utils
import time
from sklearn.model_selection import train_test_split


logger = logging.getLogger('sna')


def run(graph, args):
    """Run node classification on the graph.

    Args:
        graph (networkx.classes.graph.Graph): A NetworkX graph object.
        args (argparse.Namespace): The provided application arguments.
    """

    if args.embed:
        logger.info(f'\nEmbedding algorithm started.')
        start = time.time()

        embedding.create_embedding(args, graph)
        time_diff = time.time() - start
        logger.info(
            f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embeddings = utils.load_embedding(args.output)
    logger.info(f'\nEmbedding loaded.')

    labels = utils.get_labels(graph, args.node_ml_target)

    X = []
    y = []
    for x in embeddings.keys():
        X.append(embeddings[x])
        y.append(labels[int(x)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, test_size=0.25)

    logger.info(f'\nEmbedding evaluation started.')
    start = time.time()

    result = evaluation.evaluate(args.classifier,
                                 X_train,
                                 y_train,
                                 X_test,
                                 y_test)

    time_diff = time.time() - start
    logger.info(
        f'Embedding evaluation finished in {time_diff:.2f} seconds.')

    accuracy = evaluation.evaluate_model(result["classifier"],
                                         X_test,
                                         y_test)

    logger.info(
        f"Scores on test set.")
    logger.info(f"accuracy_score: {accuracy}")

    if(args.results):
        evaluation.save_evaluation_results(args.dataset,
                                           args.method,
                                           args.classifier,
                                           (accuracy),
                                           args.results)
