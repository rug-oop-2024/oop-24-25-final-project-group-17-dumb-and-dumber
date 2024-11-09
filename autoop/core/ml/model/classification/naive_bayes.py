from sklearn.naive_bayes import GaussianNB
import numpy as np
from autoop.core.ml.model import Model

class NaiveBayesModel(Model):
    """
    A naive Bayes model from scikit-learn.

    I have no idea how this works. 
    so we probably shouldnt use it rn.
    """
    
    def __init__(self):
        """Initializes the model."""
        self._model = GaussianNB()

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