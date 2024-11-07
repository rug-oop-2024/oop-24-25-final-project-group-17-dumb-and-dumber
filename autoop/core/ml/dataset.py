import io
from abc import ABC, abstractmethod

import pandas as pd

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """The class for a dataset artifact."""

    def __init__(self, *args, **kwargs):
        """Initializes the Dataset object."""
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ):
        """Create a dataset artifact from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
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
