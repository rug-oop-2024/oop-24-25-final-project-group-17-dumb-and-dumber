from autoop.core.ml.model.model import Model

from sklearn import tree
import numpy as np

class DecisionTreeRegressorModel(Model):
    """
    A decision tree regressor model from scikit-learn.
    
    Read this: https://scikit-learn.org/1.5/modules/tree.html
    """

    def __init__(self, max_depth: int = None):
        """Initializes the model."""
        super().__init__()
        self._model = tree.DecisionTreeRegressor(max_depth=max_depth)

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """Fits the model."""
        self._parameters = {
            "observations": observations,
            "ground_truth": ground
        }
        self._model.fit(observations, ground)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data."""
        return self._model.predict(X)