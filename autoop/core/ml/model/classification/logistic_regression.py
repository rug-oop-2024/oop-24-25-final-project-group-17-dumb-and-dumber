from sklearn.linear_model import LogisticRegression
import numpy as np
from autoop.core.ml.model import Model
from typing import Optional


class SoftmaxRegressionModel(Model):
    """
    A Multinomial Logistic Regression model from scratch.
    
    better for more or three categories
    """
    
    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000) -> None:
        """Initializes the model."""
        super().__init__()
        
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        self._hyper_params = {
            "learning_rate": learning_rate,
            "num_iterations": num_iterations,
        }

        self.theta: Optional[np.ndarray] = None

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for each sets of scores in z.

        Arguments:
            z (np.ndarray): A numpy array of shape (n, m) where n is the number of classes and m is the number of samples.

        Returns:
            np.ndarray: A numpy array of shape (n, m) containing the softmax probabilities.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) # Apparently I need this for numerical stability
        return exp_z / np.sum(np.exp(z), axis=1, keepdims=True)
    
    def compute_cost(self, observations: np.ndarray, ground: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.

        Arguments:
            observations (np.ndarray): The feature matrix
            ground (np.ndarray): The ground truth labels

        Returns:
            float: The cross-entropy loss.
        """
        m = observations.shape[0]
        h = self.softmax(observations.dot(self.theta))
        ground_one_hot = np.eye(self.theta.shape[1])[ground]
        cost = -(1 / m) * np.sum(ground_one_hot * np.log(h))
        return cost
    
    def gradient_descent(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Perform gradient descent to minimize the cost function.

        Arguments:
            observations (np.ndarray): The feature matrix
            ground (np.ndarray): The ground truth labels
        """
        m = observations.shape[0]
        ground_one_hot = np.eye(self.theta.shape[1])[ground]

        for _ in range(self.num_iterations):
            h = self.softmax(observations.dot(self.theta))
            gradient = (1 / m) * observations.T.dot(h - ground_one_hot)
            self.theta -= self.learning_rate * gradient

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fits the model.
        
        Arguments:
            observations (np.ndarray): The feature matrix
            ground (np.ndarray): The ground truth labels
        """
        self._parameters = {
            "observations": observations,
            "ground_truth": ground,
        }

        observations = np.c_[np.ones((observations.shape[0], 1)), observations]
        self.theta = np.zeros((observations.shape[1], len(np.unique(ground))))
        self.gradient_descent(observations, ground)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the probabilities for each class.

        Arguments:
            X (np.ndarray): The feature matrix

        Returns:
            np.ndarray: The predicted probabilities
        """
        X = np.c_[np.ones((X.shape[0], 1)), X]
        return self.softmax(X.dot(self.theta))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for the given input data.

        Arguments:
            X (np.ndarray): The feature matrix

        Returns:
            np.ndarray: The predicted target variable
        """
        return np.argmax(self.predict_proba(X), axis=1)
    

class LogisticRegressionModel(Model):
    """A logistic regression model from scikit-learn."""

    def __init__(self):
        """Initializes the model."""
        self._model = LogisticRegression()

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
    
