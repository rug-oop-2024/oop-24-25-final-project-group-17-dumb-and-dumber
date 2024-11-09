from autoop.core.ml.model import Model

from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class KNearestNeighbors(Model):
    """A K-nearest neighbors classifier model from scikit-learn."""

    def __init__(self, n_neighbors: int = 5):
        """Initializes the model."""
        super().__init()
        self._model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """Fits the model."""
        self._parameters = {
            "observations": observations,
            "ground_truth": ground,
        }
        self._model.fit(observations, ground)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data."""
        predictions = self._model.predict(X)
        self._parameters["predictions"] = predictions
        return predictions