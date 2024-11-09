from autoop.core.ml.model import Model
from sklearn import tree
import numpy as np

class DecisionTree(Model):
    """A decision tree classifier model from scikit-learn."""

    def __init__(self):
        """Initializes the model."""
        super().__init()
        self._clf = tree.DecisionTreeClassifier()

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """Fits the model."""
        self._parameters = {
            "observations": observations,
            "ground_truth": ground,
        }
        self._clf.fit(observations, ground) 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target variable for the given input data."""
        return self._clf.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts the probability of each class for the given input data."""
        return self._clf.predict_proba(X)
    
    def plot_tree(self):
        """
        Plots the decision tree.
        
        Need to implement this with streamlit somehow.
        """
        # TODO: Implement this with streamlit.
        tree.plot_tree(self._clf, filled=True)