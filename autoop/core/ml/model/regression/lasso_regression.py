from typing import Literal

import numpy as np
from sklearn.linear_model import Lasso

from autoop.core.ml.model.model import Model


class LassoRegression(Model):
    """
    A facade Lasso regression model from scikit-learn.

    Read this:
    https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.Lasso.html
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        warm_start: bool = False,
        positive: bool = False,
        selection: Literal["cyclic", "random"] = "cyclic",
    ):
        """Initializes the model."""
        super().__init__()
        self._type = "regression"
        self._model = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            selection=selection,
        )

        self._hyper_params = {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "max_iter": max_iter,
            "tol": tol,
            "warm_start": warm_start,
            "positive": positive,
            "selection": selection,
        }

    def initialize(self, hyper_params: dict) -> None:
        """Initializes the model with hyper-parameters."""
        self._model = Lasso(**hyper_params)

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fits the model.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground: np.ndarray: The ground truths to fit the model with.
        """
        super().fit(observations, ground)

        # Ensure ground is 1-dimensional (single target)
        if ground.ndim != 1:
            raise ValueError(
                "The ground array must be 1-dimensional for a single-target regression."
            )

        self._model.fit(observations, ground)

        # Capture model attributes
        self._model_attrs = {
            "coef": self._model.coef_,
            "intercept": self._model.intercept_,
            "n_iter": self._model.n_iter_,
            "dual_gap": self._model.dual_gap_
            if isinstance(self._model.dual_gap_, float)
            else None,
            "n_features_in": getattr(self._model, "feature_names_in_", None),
            "feature_names_in": self._model.feature_names_in_
            if isinstance(self._model, np.ndarray)
            else None,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data."""
        super().predict(observations)
        # Check that X has the correct number of features
        if observations.shape[1] != self._model_attrs["n_features_in"]:
            raise ValueError(
                f"Expected {self._model_attrs['n_features_in']} \
                             features, but got {observations.shape[1]} features."
            )

        return self._model.predict(observations)

    @property
    def coefficients(self) -> np.ndarray:
        """Returns the coefficients of the model."""
        return self._model_attrs["coef"]

    @property
    def intercept(self) -> float:
        """Returns the intercept of the model."""
        return self._model.intercept_
