from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "root_mean_squared_error",
    "r2_score",
    "accuracy",
    "precision",
    "recall",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    """
    Factory function to get a metric by name.

    Args:
        name (str): Name of the metric. One of the METRICS.
    """
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    elif name == "r2_score":
        return R2Score()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    else:
        raise ValueError(
            f"\nUnknown metric: {name} \n Supported metrics are: {list(METRICS)}"
        )


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Call the metric."""
        pass

    def evaluate(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Evaluate the metric."""
        return self(ground_truth, predics)

    def __str__(self):
        """Return the name of the metric."""
        return self.__class__.__name__


# concrete implementations of the Metric class


class MeanSquaredError(Metric):
    """Measure the mean squared error of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the mean squared error of the model."""
        return np.mean((ground_truth - predics) ** 2)


class Accuracy(Metric):
    """Measure the accuracy model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the accuracy model."""
        return np.mean(ground_truth == predics)


class MeanAbsoluteError(Metric):
    """Measure the mean absolute error of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the mean absolute error of the model."""
        return np.mean(np.abs(ground_truth - predics))


class RootMeanSquaredError(Metric):
    """Measure the root mean squared error of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the root mean squared error of the model."""
        return np.sqrt(np.mean((ground_truth - predics) ** 2))


class R2Score(Metric):
    """Measure the R2 score of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the R2 score of the model."""
        ss_res = np.sum((ground_truth - predics) ** 2)
        ss_tot = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        return 1 - (ss_res / ss_tot)


class Precision(Metric):
    """Measure the precision of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the precision of the model."""
        true_positive = np.sum((predics == 1) & (ground_truth == 1))
        false_positive = np.sum((predics == 1) & (ground_truth == 0))
        if true_positive + false_positive > 0:
            return true_positive / (true_positive + false_positive)
        return 0.0


class Recall(Metric):
    """Measure the recall of the model."""

    def __call__(self, ground_truth: np.ndarray, predics: np.ndarray) -> float:
        """Measure the recall of the model."""
        true_positive = np.sum((predics == 1) & (ground_truth == 1))
        false_negative = np.sum((predics == 0) & (ground_truth == 1))
        if true_positive + false_negative > 0:
            return true_positive / (true_positive + false_negative)
        return 0.0
