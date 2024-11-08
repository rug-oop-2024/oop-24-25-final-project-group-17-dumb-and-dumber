from abc import abstractmethod
from copy import deepcopy
from typing import Any, Literal

import numpy as np

from autoop.core.ml.artifact import Artifact


class Model:
    """The abstract class for any machine learning model."""

    def to_artifact(self, name: str, model: Any) -> Artifact:
        """Convert the model to an artifact."""
        pass

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict the target variable for the given input data."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

    pass  # your code (attribute and methods) here
