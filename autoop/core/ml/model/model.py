from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from pydantic import PrivateAttr

from autoop.core.ml.artifact import Artifact


class Model(Artifact):
    """The abstract class for any machine learning model."""

    _parameters: dict = PrivateAttr(default={})
    _hyper_params: dict = PrivateAttr(default={})
    _model_attrs: dict = PrivateAttr(default={})
    _type: Literal["classification", "regression"] = PrivateAttr()
    
    def to_artifact(self, name: str, model: Any) -> Artifact:
        """Convert the model to an artifact."""
        pass

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fit the model.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground: np.ndarray: The ground truths to fit the model with.
        """
        # Check if observations and ground have compatible shapes
        if observations.shape[0] != ground.shape[0]:
            raise ValueError(
                "The number of samples in observations and ground_truth must match."
            )

        self._parameters = {
            "observations": observations,
            "ground_truth": ground,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the target variable for the given input data.

        Arguments:
            observations: np.ndarray: The input data to predict the target variable for.

        Returns:
            np.ndarray: The predicted target variable.
        """
        # Check that the model has been fitted
        if "observations" not in self._parameters:
            raise ValueError(
                "The model has not been fitted yet. Please call fit() before predict()."
            )

    def save(self) -> None:
        """Save the model."""
        # TODO: Implement the save method.
        # Or in the artifact class
        pass

    def read(self) -> Any:
        """Read the model."""
        # TODO: Implement the read method.
        # Or in the artifact class
        pass

    # -------- GETTERS -------- # noqa
    @property
    def coefficients(self) -> dict:
        """Returns the parameters of the model."""
        return deepcopy(self._parameters)

    @property
    def hyper_params(self) -> dict:
        """Returns the hyper-parameters of the model."""
        return deepcopy(self._hyper_params)

    @property
    def model_attrs(self) -> dict:
        """Returns the model attributes."""
        return deepcopy(self._model_attrs)

    @property
    def type(self) -> Literal["classification", "regression"]:
        """Returns the type of the model."""
        return self._type

    # -------- SETTERS -------- #
    @coefficients.setter
    def coefficients(self, parameters: dict) -> None:
        """Sets the parameters of the model."""
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary")
        self._parameters = parameters
