import numpy as np

from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """A multiple linear regression model from assignment 1."""

    def __init__(self):
        """Initializes the model and parents."""
        super().__init__()
        self._type = "regression"
        self._model_attrs = {
            "coefficients": None,
            "fitted": False,
            "n_features_in": None,
        }

    def initialize(self, hyper_params: dict) -> None:
        pass

    def fit(self, observations: np.ndarray, ground_truths: np.ndarray) -> None:
        """
        Fit the model with observations and ground truths.

        Make sure the number of samples is in the row dimension,
            while the variables are in the column dimension.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground_truths: np.ndarray: The ground truths to fit the model with.
        """
        super().fit(observations, ground_truths)

        x_wave = self._add_ones_column(observations)
        x_wave_transpose = x_wave.T
        coefficients = (
            np.linalg.inv(x_wave_transpose @ x_wave) @ x_wave_transpose @ ground_truths
        )
        self._model_attrs["coefficients"] = coefficients
        self._model_attrs["fitted"] = True
        self._model_attrs["n_features_in"] = observations.shape[1]

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict outcomes based on the observations.

        Arguments:
            observations: np.ndarray: The observations to predict outcomes for.

        Returns:
            np.ndarray: The predicted outcomes.
        """
        super().predict(observations)

        # Validate input dimensionality # noqa
        if observations.shape[1] != self._model_attrs["n_features_in"]:
            raise ValueError(
                f"Expected {self._model_attrs['n_features_in']} features, "
                f"got {observations.shape[1]}."
            )

        x_wave = self._add_ones_column(observations)
        result = x_wave @ np.array(self._model_attrs["coefficients"])
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

    @property
    def intercept(self) -> float:
        """Returns the intercept of the model."""
        return (
            self._model_attrs["coefficients"][0]
            if self._model_attrs["fitted"]
            else None
        )

    @property
    def coefficients(self) -> np.ndarray:
        """Returns the coefficients of the model, excluding the intercept."""
        return (
            self._model_attrs["coefficients"][1:]
            if self._model_attrs["fitted"]
            else None
        )

    @property
    def n_features_in(self) -> int:
        """Returns the number of features in the model."""
        return self._model_attrs["n_features_in"]
