import csv
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger('sna')


def link_prediction_classifier(classifier: str) -> Pipeline:
    """Create the classifier.

    Args:
        classifier (str): The classifier for node classification evaluation.

    Returns:
        (sklearn.pipeline.Pipeline): Pipeline of transforms with a final estimator.
    """

    if classifier == 'logisticalregression':
        clf = LogisticRegressionCV(
            Cs=10, cv=10, scoring="roc_auc", multi_class="ovr", max_iter=10000)
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
                   X_test_nodes: np.ndarray,
                   y_test: np.ndarray) -> tuple:
    """Calculate node classification model scores.

    Args:
        classifier (str): The classifier for node classification evaluation.
        X_train_nodes (np.ndarray): Testing data nodes.
        y_train (np.ndarray): Testing targets.

    Returns:
        (float): Node classification accuracy score.
    """

    X_test = X_test_nodes

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def evaluate(classifier: str,
             X_train_nodes: np.ndarray,
             y_train: np.ndarray,
             X_model_selection_nodes: np.ndarray,
             y_model_selection: np.ndarray) -> list:
    """Evaluate the input data.

    Args:
        classifier (str): The classifier for node classification evaluation.
        X_train_nodes (np.ndarray): Training data nodes.
        y_train (np.ndarray): Training targets.
        X_model_selection_nodes (np.ndarray): Model selection data nodes.
        y_model_selection (np.ndarray): Model selection targets.

    Returns:
        results (Dict): Node classification evaluation results.
    """

    clf = link_prediction_classifier(classifier)
    clf.fit(np.array(X_train_nodes), np.array(y_train))

    accuracy = evaluate_model(clf,
                              X_model_selection_nodes,
                              y_model_selection)

    result = {"classifier": clf,
              "accuracy": accuracy}

    return result


def save_evaluation_results(dataset: str,
                            method: str,
                            classifier: str,
                            evaluation_results: tuple,
                            file_path: str) -> None:
    """Save the node classification evaluation results in JSON format.

    Args:
        dataset (str): The name of the input dataset.
        method (str): The name of the embedding method.
        evaluation_results (tuple): Results of the embedding evaluation.
        file_path (str): Path to the file with the embedding evaluation results.
    """

    accuracy = evaluation_results
    json_results = [dataset,
                    method,
                    classifier,
                    accuracy]

    with open(file_path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(json_results)
