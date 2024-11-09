from autoop.core.ml.model import Model
from pydantic import Field
import numpy as np


class KNearestNeighbors(Model):
    """A K-nearest neighbors classifier model from assignment 1."""

    k: int = Field(5, description="The number of neighbors to consider.")

    def __init__(self, n_neighbors: int = 5):
        """Initializes the model."""
        super().__init__()

        self.k = n_neighbors

        self._hyper_params = {
            "n_neighbors": n_neighbors,
        }

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fit the model with observations and ground truths.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground_truths: np.ndarray: The ground truths to fit the model with.
        """
        self._parameters = {
            "observations": observations,
            "ground_truth": ground,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict outcomes based on the observations.

        Arguments:
            observations: np.ndarray: The observations to predict outcomes for.

        Returns:
            np.ndarray: The predicted outcomes.
        """
        return np.array(
            [self._predict(observation) for observation in observations]
        )  # noqa
    
    def _predict(self, observation: np.ndarray) -> int:
        """
        Predict the classification of a single observation.

        Arguments:
            observation: np.ndarray: The observation to predict the
            classification of.

        Returns:
            np.ndarray: The predicted classification.
        """
        
        # Find the distance between the test point and all the training points.
        obs = self.parameters["observations"]
        distances = [self._distance(observation, pt) for pt in obs]

        # Convert the distances to a numpy array. # noqa: SC100
        distances = np.array(distances)

        # Sort the distances and find the k nearest point's indices.
        nearest_indices = distances.argsort()[: self.k]

        # Get the ground truths of the k nearest points.
        nearest_classes = np.array(self.parameters["ground_truths"])[
            nearest_indices
        ]  # noqa

        # Take the average of the k nearest points (their classifications).
        return np.round(nearest_classes.mean())
    
    def _distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """
        Calculate the Euclidean distance between two points.

        Arguments:
            pt1: np.ndarray: The first point.
            pt2: np.ndarray: The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.linalg.norm(pt1 - pt2)