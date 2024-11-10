from typing import Literal

import numpy as np
from sklearn import tree

from autoop.core.ml.model import Model


class DecisionTreeClassifier(Model):
    """
    A decision tree classifier model from scikit-learn.

    Read this: https://scikit-learn.org/1.5/modules/tree.html
    """

    def __init__(
        self,
        criterion: Literal["gini", "entropy"] = "gini",
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Literal["auto", "sqrt", "log2"] = "sqrt",
    ):
        """Initializes the model."""
        super().__init__()
        self._type = "classification"

        self._hyper_params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

        self._clf = tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fits the model with input shape checks.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground: np.ndarray: The ground truths to fit the model with.
        """
        super().fit(observations, ground)

        # Ensure ground is 1-dimensional (for a single-target classifier)
        if ground.ndim != 1:
            raise ValueError(
                "The ground array must be 1-dimensional for classification."
            )

        self._clf.fit(observations, ground)

        # Capture model attributes
        self._model_attrs = {
            "classes": self._clf.classes_,
            "feature_importances": self._clf.feature_importances_,
            "max_features": self._clf.max_features_,
            "n_classes": self._clf.n_classes_,
            "n_features_in_fit": self._clf.n_features_in_,
        }  # TODO: possibly get the tree itself?

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data \
            with shape validation.

        Arguments:
            X: np.ndarray: The input data to predict the target variable for.

        Returns:
            np.ndarray: The predicted target variable.
        """
        super().predict(observations)
        # Validate that X has the correct number of features
        if observations.shape[1] != self._model_attrs["n_features_in_fit"]:
            raise ValueError(
                f"Expected {self._model_attrs['n_features_in_fit']} \
                    features, but got {observations.shape[1]}."
            )
        return self._clf.predict(observations)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the probability of each class.

        Arguments:
            X: np.ndarray: The input data to predict the target variable for.

        Returns:
            np.ndarray: The predicted probabilities of each target variable.
        """
        # Check that the model has been fitted
        if "n_features_in_fit" not in self._model_attrs:
            raise ValueError(
                "The model has not been fitted yet. \
                             Please call fit() before predict_proba()."
            )

        # Validate that X has the correct number of features
        if X.shape[1] != self._model_attrs["n_features_in_fit"]:
            raise ValueError(
                f"Expected {self._model_attrs['n_features_in_fit']} \
                             features, but got {X.shape[1]}."
            )

        return self._clf.predict_proba(X)

    def plot_tree(self):
        """
        Plots the decision tree.

        Need to implement this with streamlit somehow.
        """
        # TODO: Implement this with streamlit. # noqa
        tree.plot_tree(self._clf, filled=True)
