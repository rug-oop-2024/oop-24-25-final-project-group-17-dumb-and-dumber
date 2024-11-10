import io

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """The class for a dataset artifact."""

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the Dataset object."""
        super().__init__(*args, **kwargs)
        self._type = "dataset"

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """Create a dataset artifact from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Dataset":
        """Convert an Artifact instance to a Dataset instance if possible."""
        if artifact.type != "dataset":
            raise ValueError(
                f"Cannot convert artifact of type '{artifact.type}' to Dataset."
            )

        return Dataset(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=artifact.read(),
            version=artifact.version,
            tags=artifact.tags,
            metadata=artifact.metadata,
        )

    def read(self) -> pd.DataFrame:
        """Read the data of the artifact."""
        _bytes = super().read()
        csv = _bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """Save the data of the artifact."""
        _bytes = data.to_csv(index=False).encode()
        return super().save(_bytes)
