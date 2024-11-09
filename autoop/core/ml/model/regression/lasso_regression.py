from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso
import numpy as np

class LassoRegression(Model):
    """A facade Lasso regression model from scikit-learn."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 0.0001):
        """Initializes the model."""
        self._model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)

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
    
    @property
    def coefficients(self) -> np.ndarray:
        """Returns the coefficients of the model."""
        return self._model.coef_
    
    @property
    def intercept(self) -> float:
        """Returns the intercept of the model."""
        return self._model.intercept_