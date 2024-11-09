from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Any, Literal
from pydantic import PrivateAttr

import numpy as np

from autoop.core.ml.artifact import Artifact


class Model(Artifact):
    """The abstract class for any machine learning model."""

    _parameters: dict = PrivateAttr(default={})
    _hyper_params: dict = PrivateAttr(default={})

    def to_artifact(self, name: str, model: Any) -> Artifact:
        """Convert the model to an artifact."""
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the target variable for the given input data."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

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
    
    # -------- GETTERS -------- #
    @property
    def coefficients(self) -> dict:
        """Returns the parameters of the model."""
        return deepcopy(self._parameters)
    
    @property
    def hyper_params(self) -> dict:
        """Returns the hyper-parameters of the model."""
        return deepcopy(self._hyper_params)
    
    # -------- SETTERS -------- #
    @coefficients.setter
    def coefficients(self, parameters: dict) -> None:
        """Sets the parameters of the model."""
        if not isinstance(parameters, dict):
            raise TypeError("parameters must be a dictionary")
        self._parameters = parameters

    @hyper_params.setter
    def hyper_params(self, hyper_params: dict) -> None:
        """Sets the hyper-parameters of the model."""
        if not isinstance(hyper_params, dict):
            raise TypeError("hyper_params must be a dictionary")
        self._hyper_params = hyper_params
    