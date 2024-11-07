# flake8: noqa
import unittest

from autoop.core.ml.metric import (
    Accuracy,
    MeanAbsoluteError,
    MeanSquaredError,
    Precision,
    R2Score,
    Recall,
    RootMeanSquaredError,
    get_metric,
)


class TestGetMetric(unittest.TestCase):
    def test_get_mean_squared_error(self):
        metric = get_metric("mean_squared_error")
        self.assertIsInstance(metric, MeanSquaredError)

    def test_get_accuracy(self):
        metric = get_metric("accuracy")
        self.assertIsInstance(metric, Accuracy)

    def test_get_mean_absolute_error(self):
        metric = get_metric("mean_absolute_error")
        self.assertIsInstance(metric, MeanAbsoluteError)

    def test_get_root_mean_squared_error(self):
        metric = get_metric("root_mean_squared_error")
        self.assertIsInstance(metric, RootMeanSquaredError)

    def test_get_r2_score(self):
        metric = get_metric("r2_score")
        self.assertIsInstance(metric, R2Score)

    def test_get_precision(self):
        metric = get_metric("precision")
        self.assertIsInstance(metric, Precision)

    def test_get_recall(self):
        metric = get_metric("recall")
        self.assertIsInstance(metric, Recall)

    def test_get_unknown_metric(self):
        with self.assertRaises(ValueError):
            get_metric("unknown_metric")

    def test_unknown_metric_error_message(self):
        with self.assertRaises(ValueError) as context:
            get_metric("unknown_metric")
        print(context.exception)
        self.assertIn("Unknown metric", str(context.exception))


if __name__ == "__main__":
    unittest.main()
