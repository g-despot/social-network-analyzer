import src.utils as utils
import src.node_classification.node_classification_evaluation as node_classification_evaluation
import src.embedding as embedding
import logging
import time
from sklearn.model_selection import train_test_split


logger = logging.getLogger('sna')


def run(graph, args):

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

    result = node_classification_evaluation.evaluate(args.classifier,
                                                     X_train,
                                                     y_train,
                                                     X_test,
                                                     y_test)

    time_diff = time.time() - start
    logger.info(
        f'Embedding evaluation finished in {time_diff:.2f} seconds.')

    accuracy = node_classification_evaluation.evaluate_model(result["classifier"],
                                                             X_test,
                                                             y_test)

    logger.info(
        f"Scores on test set.")
    logger.info(f"accuracy_score: {accuracy}")

    if(args.results):
        node_classification_evaluation.save_evaluation_results(args.dataset,
                                                               args.method,
                                                               args.classifier,
                                                               (accuracy),
                                                               args.results)


"""
logger.info(f'\nEmbedding algorithm evaluation - node classification.')

        #graph = utils.load_graph(args.weighted, args.directed, args.input)

        dataset = datasets.Cora()
        graph, node_subjects = dataset.load(
            largest_connected_component_only=True)
        graph = graph.to_networkx()
        utils.print_graph_info(graph, "original graph")

        graph.remove_nodes_from(list(nx.isolates(graph)))
        utils.print_graph_info(graph, "graph without isolates")
        logger.info(f'\nEmbedding algorithm started.')
        start = time.time()

        embedding.create_embedding(args, graph)
        time_diff = time.time() - start
        logger.info(f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')
        embeddings = utils.load_embedding(args.output)
        node_ids = list(embeddings.keys())
        X = list(embeddings.values())
        logger.info(f'\n{node_subjects}')
        labels = node_subjects.values.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25)

        logger.info(f'\nEmbedding evaluation started.')
        start = time.time()

        result = node_classification_evaluation.evaluate(args.classifier,
                                                         X_train,
                                                         y_train,
                                                         X_test,
                                                         y_test)

        time_diff = time.time() - start
        logger.info(
            f'Embedding evaluation finished in {time_diff:.2f} seconds.')

        logger.info(
            f"\naccuracy score on train set: {result['accuracy']}.")

        logger.info(f'\nEmbedding algorithm started.')
        start = time.time()

        embedding.create_embedding(args, graph)
        time_diff = time.time() - start
        logger.info(
            f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

        embedding_test = utils.load_embedding(args.output)

        accuracy = node_classification_evaluation.evaluate_model(result["classifier"],
                                                                 X_test,
                                                                 y_test)

        logger.info(
            f"Scores on test set.")
        logger.info(f"accuracy_score: {accuracy}")

        if(args.results):
            node_classification_evaluation.save_evaluation_results(args.dataset,
                                                                   args.method,
                                                                   args.classifier,
                                                                   (accuracy),
                                                                   args.results)
"""
