from typing import Literal

from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors
from autoop.core.ml.model.classification.naive_bayes import NaiveBayes
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.decision_tree_regressor import (
    DecisionTreeRegressor,
)
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)

REGRESSION_MODELS = [
    "MultipleLinearRegression",
    "DecisionTreeRegressor",
    "LassoRegression",
]  # add your models as str here

CLASSIFICATION_MODELS = [
    "DecisionTreeClassifier",
    "KNearestNeighbors",
    "NaiiveBayes",
]  # add your models as str here


def get_model(
    model_name: Literal[
        "MultipleLinearRegression",
        "DecisionTreeRegressor",
        "LassoRegression",
        "DecisionTreeClassifier",
        "KNearestNeighbors",
        "NaiiveBayes",
    ]
) -> Model:
    """Factory function to get a model by name."""
    if model_name == "MultipleLinearRegression":
        return MultipleLinearRegression()
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor()
    elif model_name == "LassoRegression":
        return LassoRegression()
    elif model_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif model_name == "KNearestNeighbors":
        return KNearestNeighbors()
    elif model_name == "NaiveBayes":
        return NaiveBayes()
    else:
        raise ValueError(f"Model {model_name} not found.")
