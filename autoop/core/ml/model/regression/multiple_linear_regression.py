from autoop.core.ml.model.model import Model

from sklearn.linear_model import LinearRegression
import numpy as np


class MultipleLinearRegression(Model):
    """A multiple linear regression model from scikit-learn."""

    def __init__(self):
        """Initializes the model."""
        self._model = LinearRegression()

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """Fits the model."""	
        self._parameters = {
            "observations": observations,
            "ground_truth": ground
        }
        self._model.fit(observations, ground)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data."""
        predictions = self._model.predict(X)
        self._parameters["predictions"] = predictions
        return predictions
