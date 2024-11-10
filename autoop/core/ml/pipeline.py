import pickle
from typing import TYPE_CHECKING, List

import numpy as np

from autoop.functional.preprocessing import preprocess_features

if TYPE_CHECKING:
    from autoop.core.ml.artifact import Artifact
    from autoop.core.ml.dataset import Dataset
    from autoop.core.ml.feature import Feature
    from autoop.core.ml.metric import Metric
    from autoop.core.ml.model.model import Model


class Pipeline:
    """A pipeline for training and evaluating a model."""

    def __init__(
        self,
        metrics: List["Metric"],
        dataset: "Dataset",
        model: "Model",
        input_features: List["Feature"],
        target_feature: "Feature",
        split=0.8,
    ):
        """Initializes the pipeline."""
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.feature_type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for categorical target feature"
            )
        if target_feature.feature_type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self):
        """Returns a string representation of the pipeline."""
        return f"""
            Pipeline(
                model={self._model.type},
                input_features={list(map(str, self._input_features))},
                target_feature={str(self._target_feature)},
                split={self._split},
                metrics={list(map(str, self._metrics))},
            )
            """

    @property
    def model(self):
        """Returns the model used in the pipeline."""
        return self._model

    @property
    def artifacts(self) -> List["Artifact"]:
        """Returns artifacts generated during the pipeline execution."""
        artifacts = []
        artifact: Artifact
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
        )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
        )
        return artifacts

    def _register_artifact(self, name: str, artifact):
        """Register an artifact generated during the pipeline execution."""
        self._artifacts[name] = artifact

    def _preprocess_features(self):
        """Preprocess the features in the dataset."""
        target_feature_name, target_data, artifact = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features, self._dataset)
        for feature_name, _, artifact in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector, sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (_, data, _) in input_results]

    def _split_data(self):
        """Split the data into training and testing sets."""
        split = self._split
        self._train_X = [
            vector[: int(split * len(vector))] for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)) :] for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[: int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)) :]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """Concatenate the input vectors into a single matrix."""
        return np.concatenate(vectors, axis=1)

    def _train(self):
        """Train the model on the training set."""
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self):
        """Evaluate the model on the test set."""
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def _save(self, name: str, version: str = "1.0.0") -> "Artifact":
        """Save the pipeline."""
        pipeline_data = {
            "dataset": self._dataset,
            "model": self._model,
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
            "metrics": self._metrics,
        }

        pipeline_data = pickle.dumps(pipeline_data)
        path = f"{name}_{version}.pkl"

        return Artifact(
            type="pipeline",
            name=name,
            data=pipeline_data,
            version=version,
            asset_path=path,
        )

    def execute(self) -> dict:
        """Executes the pipeline."""
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
