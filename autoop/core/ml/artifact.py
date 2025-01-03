# TODO: We need to implement the save and read methods.
import base64  # noqa
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr


class Artifact(BaseModel, ABC):
    """The abstract class for any data artifact."""

    # Public attributes for initialization
    name: str = Field("", title="Name of the artifact")
    data: bytes = Field(b"")

    # Private attributes for internal use
    _name: str = PrivateAttr(default_factory=str)
    _type: str = PrivateAttr(default_factory=str)
    _asset_path: str = PrivateAttr(default_factory=str)
    _data: bytes = PrivateAttr(default_factory=bytes)
    _version: str = PrivateAttr(default_factory=str)
    _tags: list[str] = PrivateAttr(default_factory=list)
    _metadata: dict[str, str] = PrivateAttr(default_factory=dict[str,str])

    def __init__(self, **data):
        """Initialises the Artifact object."""
        super().__init__(**data)
        self._name = self.name
        self._type = data.get("type", "")
        self._asset_path = data.get("asset_path", "")
        self._data = data.get("data", b"")
        self._version = data.get("version", "1.0.0")
        self._tags = data.get("tags", [])
        self._metadata = data.get("metadata", dict())

    def read(self):
        """Reads the data of the artifact."""
        return self._data

    def save(self, data: bytes):
        """Saves the data of the artifact."""
        self._data = data

    # TODO: Implement the get method.
    def get(self) -> Literal["OneHotEncoder", "StandardScaler"]:
        """
        Get something in the artifact.

        Example:
            artifact_type = artifact.get("type")
        """
        pass

    # -------- GETTERS -------- # noqa
    @property
    def id(self) -> str:
        """ID of the artifact."""
        return str(base64.b64encode(self.asset_path.encode())) + self.version

    @property
    def name(self) -> str:
        """Name of the artifact."""
        return self._name

    @property
    def asset_path(self) -> str:
        """Asset path of the artifact."""
        return self._asset_path

    @property
    def version(self) -> str:
        """Return the version of the artifact."""
        return self._version
    
    @property
    def tags(self) -> list[str]:
        """Return the tags of the artifact."""
        return self._tags
    
    @property
    def metadata(self) -> dict[str, str]:
        """Return the metadata of the artifact."""
        return self._metadata
    
    @property
    def type(self) -> str:
        """Return the type of the artifact."""
        return self._type

    # -------- SETTERS --------

    @name.setter
    def name(self, name: str):
        self._name = name

    @asset_path.setter
    def asset_path(self, asset_path: str):
        self._asset_path = asset_path

    @version.setter
    def version(self, version: str):
        self._version = version
