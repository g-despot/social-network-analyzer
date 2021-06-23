import csv
import logging
import numpy as np
import src.embedding_algorithms.operators as operators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Callable


logger = logging.getLogger('sna')


def edge_to_feature(edges: list,
                    embedding: dict,
                    binary_operator: Callable) -> list:
    """Transform edges to vectors by applying a binary operator on the embeddings.

    Args:
        edges (list): A list of edges to transform.
        embedding (dict): Dictionary of embedding vectors.
        binary_operator (Callable): The binary operator that is applied on the embeddings of the source and target nodes of each sampled edge.

    Returns:
        features (list): List of edge vector representations.
    """

    features = []
    for start_node, end_node in edges:
        features.append(binary_operator(
            np.array(embedding[start_node]), np.array(embedding[end_node])))

    return features


def link_prediction_classifier(classifier: str) -> Pipeline:
    """Create the classifier.

    Args:
        classifier (str): The classifier for link prediction evaluation.

    Returns:
        (sklearn.pipeline.Pipeline): Pipeline of transforms with a final estimator.
    """

    if classifier == 'logisticalregression':
        clf = LogisticRegressionCV(
            Cs=10, cv=10, scoring="roc_auc", max_iter=10000)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", clf)])

    elif classifier == 'randomforest':
        clf = RandomForestClassifier()
        param_grid = {'n_estimators': [
            10, 50, 100, 150, 200], 'max_depth': [5, 10, 15]}
        gridsearch = GridSearchCV(clf, param_grid=param_grid)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", gridsearch)])

    elif classifier == 'gradientboost':
        clf = GradientBoostingClassifier()
        param = {'learning_rate': [0.01, .05, .1, 0.15, 0.2]}
        gridsearch = GridSearchCV(clf, param_grid=param)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", gridsearch)])

    else:
        raise ValueError(f'Invalid classifier: {classifier}')


def evaluate_model(clf: Pipeline,
                   embedding: dict,
                   binary_operator: Callable,
                   X_test_edges: np.ndarray,
                   y_test: np.ndarray) -> tuple:
    """Calculate link prediction model scores.

    Args:
        classifier (str): The classifier for link prediction evaluation.
        embedding (dict): Dictionary of embedding vectors.
        binary_operator (Callable): The binary operator that is applied on the embeddings of the source and target nodes of each sampled edge.
        X_train_edges (np.ndarray): Testing data edges.
        y_train (np.ndarray): Testing targets.

    Returns:
        (tuple): Link prediction scores: roc_auc, average_precision, accuracy and f1.
    """

    X_test = edge_to_feature(X_test_edges,
                             embedding, binary_operator)

    predicted = clf.predict_proba(X_test)
    positive_column = list(clf.classes_).index(1)
    y_score = predicted[:, positive_column]
    y_pred = clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return roc_auc, average_precision, accuracy, f1


def train_and_evaluate_model(classifier: str,
                             embedding: dict,
                             binary_operator: Callable,
                             X_train_edges: np.ndarray,
                             y_train: np.ndarray,
                             X_model_selection_edges: np.ndarray,
                             y_model_selection: np.ndarray) -> dict:
    """Train the classifier and evaluate the link prediction model.

    Args:
        classifier (str): The classifier for link prediction evaluation.
        binary_operator (Callable): The binary operator that is applied on the embeddings of the source and target nodes of each sampled edge.
        X_train_edges (np.ndarray): Training data edges.
        y_train (np.ndarray): Training targets.
        embedding (dict): Dictionary of embedding vectors.
        X_model_selection_edges (np.ndarray): Model selection data edges.
        y_model_selection (np.ndarray): Model selection targets.

    Returns:
        result (dict): A single link prediction result.
    """

    clf = link_prediction_classifier(classifier)
    X_train = edge_to_feature(X_train_edges, embedding, binary_operator)
    clf.fit(X_train, y_train)

    roc_auc, average_precision, accuracy, f1 = evaluate_model(clf,
                                                              embedding,
                                                              binary_operator,
                                                              X_model_selection_edges,
                                                              y_model_selection)

    result = {"classifier": clf,
              "binary_operator": binary_operator,
              "roc_auc": roc_auc,
              "average_precision": average_precision,
              "accuracy": accuracy,
              "f1": f1}

    return result


def evaluate(classifier: str,
             embedding: dict,
             X_train_edges: np.ndarray,
             y_train: np.ndarray,
             X_model_selection_edges: np.ndarray,
             y_model_selection: np.ndarray) -> list:
    """Evaluate the input data.

    Args:
        classifier (str): The classifier for link prediction evaluation.
        embedding (dict): Dictionary of embedding vectors.
        X_train_edges (np.ndarray): Training data edges.
        y_train (np.ndarray): Training targets.
        X_model_selection_edges (np.ndarray): Model selection data edges.
        y_model_selection (np.ndarray): Model selection targets.

    Returns:
        results (list): Link prediction evaluation results.
    """

    binary_operators = [operators.hadamard,
                        operators.l1,
                        operators.l2,
                        operators.avg]

    results = []
    for binary_operator in binary_operators:
        results.append(train_and_evaluate_model(classifier,
                                                embedding,
                                                binary_operator,
                                                X_train_edges,
                                                y_train,
                                                X_model_selection_edges,
                                                y_model_selection))

    return results


def save_evaluation_results(dataset: str,
                            method: str,
                            classifier: str,
                            evaluation_results: tuple,
                            file_path: str) -> None:
    """Save the link prediction evaluation results in JSON format.

    Args:
        dataset (str): The name of the input dataset.
        method (str): The name of the embedding method.
        evaluation_results (tuple): Results of the embedding evaluation.
        file_path (str): Path to the file with the embedding evaluation results.
    """

    roc_auc, average_precision, accuracy, f1 = evaluation_results
    json_results = [dataset,
                    method,
                    classifier,
                    roc_auc,
                    average_precision,
                    accuracy,
                    f1]

    with open(file_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(json_results)
