from autoop.core.ml.model import Model
from sklearn import tree
from typing import Literal
import numpy as np

class DecisionTree(Model):
    """A decision tree classifier model from scikit-learn."""

    def __init__(
            self, 
            criterion: Literal["gini", "entropy"] = "gini", 
            max_depth: int = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: Literal["auto", "sqrt", "log2"] = "sqrt",
            ):
        """Initializes the model."""
        super().__init__()

        self._hyper_params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

        self._clf = tree.DecisionTreeClassifier(
            criterion=criterion, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            )

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