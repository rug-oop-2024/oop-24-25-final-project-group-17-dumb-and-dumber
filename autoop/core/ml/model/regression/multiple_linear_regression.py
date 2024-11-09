from autoop.core.ml.model.model import Model
import numpy as np


class MultipleLinearRegression(Model):
    """A multiple linear regression model from scikit-learn."""

    def __init__(self):
        """Initializes the model and parents."""
        super().__init__()

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Fit the model with observations and ground truths.

        Make sure the number of samples is in the row dimension,
            while the variables are in the column dimension.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground_truths: np.ndarray: The ground truths to fit the model with.
        """
        self._parameters = {
            "observations": observations,
            "ground_truths": ground_truths,
        }

        x_wave = self._add_ones_column(observations)
        x_wave_transpose = x_wave.T
        self.coefficients = (
            np.linalg.inv(x_wave_transpose @ x_wave) @ x_wave_transpose @ ground_truths
        )

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict outcomes based on the observations.

        Arguments:
            observations: np.ndarray: The observations to predict outcomes for.

        Returns:
            np.ndarray: The predicted outcomes.
        """
        x_wave = self._add_ones_column(observations)
        result = x_wave @ self.coefficients
        return np.round(np.array(result), 2)
    
    @staticmethod
    def _add_ones_column(matrix: np.ndarray) -> np.ndarray:
        """
        Add a column of ones to the matrix.

        Arguments:
            matrix: np.ndarray: The matrix to add a column of ones to.

        Returns:
            np.ndarray: The matrix with a column of ones added.
        """
        temp = np.ones((matrix.shape[0], 1))
        return np.hstack((temp, matrix))

    # TODO: Implement shape checks for incoming observations and ground_truths
