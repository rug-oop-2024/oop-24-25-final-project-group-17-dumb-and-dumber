from typing import Literal

import numpy as np
from sklearn import tree

from autoop.core.ml.model.model import Model


class DecisionTreeRegressor(Model):
    """
    A decision tree regressor model from scikit-learn.

    Read this: https://scikit-learn.org/1.5/modules/tree.html
    """

    def __init__(
        self,
        max_depth: int = None,
        criterion: Literal[
            "squared_error", "friedman_mse", "absolute_error", "poisson"
        ] = "mse",
        splitter: Literal["best", "random"] = "best",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Literal["sqrt", "log2"] = None,
        random_state: int = None,
    ):
        """Initializes the model."""
        super().__init__()
        self._model = tree.DecisionTreeRegressor(
            max_depth=max_depth,
            criterion=criterion,
            splitter=splitter,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )

        self._hyper_params = {
            "max_depth": max_depth,
            "criterion": criterion,
            "splitter": splitter,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fits the model.

        Arguments:
            observations (np.ndarray): The observations to fit the model with.
            ground (np.ndarray): The ground truths to fit the model with.
        """
        super().fit(observations, ground)
        self._model.fit(observations, ground)

        # Capture model attributes
        self._model_attrs = {
            "feature_importances": self._model.feature_importances_,
            "max_features": self._model.max_features_,
            "n_features_in_fit": self._model.n_features_in_,
            "feature_names_in_fit": self._model.feature_names_in_,
        }  # TODO: possibly get the tree itself?

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for the given input data.

        Arguments:
            observations (np.ndarray): The observations to predict
            the target variable for.

        Returns:
            np.ndarray: The predicted target variable.
        """
        super().predict(observations)
        return self._model.predict(observations)
