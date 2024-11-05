# TODO: We need to implement the save and read methods.
import base64  # noqa
from abc import ABC, abstractmethod

from pydantic import BaseModel, PrivateAttr


class Artifact(BaseModel, ABC):
    """The abstract class for any data artifact."""

    _name: str = PrivateAttr(default_factory=str)
    _asset_path: str = PrivateAttr(default_factory=str)
    _data: bytes = PrivateAttr(default_factory=bytes)
    _version: str = PrivateAttr(default_factory=str)

    def __init__(self, name: str, asset_path: str, data: bytes, version: str, **kwargs):
        """Initialises the Artifact object."""
        super().__init__(**kwargs)
        self._name = name
        self._asset_path = asset_path
        self._data = data
        self._version = version

    @abstractmethod
    def read(self):
        """Reads the data of the artifact."""
        return self._data

    @abstractmethod
    def save(self):
        """Saves the data of the artifact."""
        pass

    # -------- GETTERS -------- # noqa

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
