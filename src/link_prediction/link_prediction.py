import src.utils as utils
import src.link_prediction.link_prediction_evaluation as link_prediction_evaluation
import src.embedding as embedding
import logging
import time
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split

logger = logging.getLogger('sna')

def run(graph, args):
    edge_splitter_test = EdgeSplitter(graph)

    graph_test, X_test_edges, y_test = edge_splitter_test.train_test_split(
        p=args.test_percentage, method="global"
    )

    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, X_edges, y = edge_splitter_train.train_test_split(
        p=args.train_percentage, method="global"
    )
    X_train_edges, X_model_selection_edges, y_train, y_model_selection = train_test_split(
        X_edges, y, train_size=0.75, test_size=0.25)

    logger.info(f'\nEmbedding algorithm started.')
    start = time.time()

    embedding.create_embedding(args, graph_train)
    time_diff = time.time() - start
    logger.info(
        f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embeddings = utils.load_embedding(args.output)

    logger.info(f'\nEmbedding evaluation started.')
    start = time.time()
    results = link_prediction_evaluation.evaluate(args.classifier,
                                                    embeddings,
                                                    X_train_edges,
                                                    y_train,
                                                    X_model_selection_edges,
                                                    y_model_selection)

    time_diff = time.time() - start
    logger.info(
        f'Embedding evaluation finished in {time_diff:.2f} seconds.')

    best_result = max(results, key=lambda result: result["roc_auc"])

    logger.info(
        f"\nBest roc_auc_score on train set using '{best_result['binary_operator'].__name__}': {best_result['roc_auc']}.")

    logger.info(f'\nEmbedding algorithm started.')
    start = time.time()

    embedding.create_embedding(args, graph_test)
    time_diff = time.time() - start
    logger.info(
        f'\nEmbedding algorithm finished in {time_diff:.2f} seconds.')

    embedding_test = utils.load_embedding(args.output)

    roc_auc, average_precision, accuracy, f1 = link_prediction_evaluation.evaluate_model(best_result["classifier"],
                                                                                            embedding_test,
                                                                                            best_result["binary_operator"],
                                                                                            X_test_edges,
                                                                                            y_test)

    logger.info(
        f"Scores on test set using '{best_result['binary_operator'].__name__}'.")
    logger.info(f"roc_auc_score: {roc_auc}")
    logger.info(f"average_precision_score: {average_precision}")
    logger.info(f"accuracy_score: {accuracy}")
    logger.info(f"f1_score on test set using: {f1}\n")

    if(args.results):
        link_prediction_evaluation.save_evaluation_results(args.dataset,
                                                            args.method,
                                                            args.classifier,
                                                            (roc_auc, average_precision,
                                                            accuracy, f1),
                                                            args.results)