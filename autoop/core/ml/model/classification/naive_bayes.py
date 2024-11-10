import numpy as np
from pydantic import PrivateAttr
from sklearn.naive_bayes import GaussianNB

from autoop.core.ml.model import Model


class NaiveBayes(Model):
    """
    A naive Bayes model from scikit-learn.

    Read this: https://scikit-learn.org/1.5/modules/naive_bayes.html
    """

    def __init__(self, priors: np.ndarray = None, var_smoothing: float = 1e-9) -> None:
        """
        Initializes the model.

        Arguments:
            priors: np.ndarray: The prior probabilities of the classes.
            var_smoothing: float: The portion of the largest variance of
            all features to add to variances for calculation stability.
        """
        super().__init__()
        self._model = GaussianNB(priors=priors, var_smoothing=var_smoothing)
        self._hyper_params = {
            "priors": priors,
            "var_smoothing": var_smoothing,
        }

    def fit(self, observations: np.ndarray, ground: np.ndarray) -> None:
        """
        Fits the model.

        Arguments:
            observations: np.ndarray: The observations to fit the model with.
            ground: np.ndarray: The ground truths to fit the model with.
        """
        super().fit(observations, ground)
        self._model.fit(observations, ground)

        # Capture model attributes
        self._model_attrs = {
            "class_prior": self._model.class_prior_,
            "class_count": self._model.class_count_,
            "theta": self._model.theta_,
            "classes": self._model.classes_,
            "varience": self._model.var_,
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predicts the target variable for the given input data.

        Arguments:
            observations: np.ndarray: The input data to predict
            the target variable for.

        Returns:
            np.ndarray: The predicted target variable.
        """
        super().predict(observations)
        return self._model.predict(observations)
