from typing import Literal

from pydantic import BaseModel, PrivateAttr


class Feature(BaseModel):
    """
    Feature class to represent a feature in a dataset.

    Needs to be PrivateAttr but right now pydantic is being a bitch.
    """

    _name: str = PrivateAttr()
    _type: Literal["numerical", "categorical"] = PrivateAttr()

    def __init__(
        self, name: str, feature_type: Literal["numerical", "categorical"], **kwargs
    ):
        """Initialises the Feeature object."""
        super().__init__(**kwargs)
        self._name = name
        self._type = feature_type

    def __str__(self):
        """Returns the Feature object as a str."""
        
        feature_type_display = self.feature_type.capitalize()
        return f"{self.name} ({feature_type_display})"

    # -------- GETTERS -------- # noqa

    @property
    def name(self) -> str:
        """Name of the Feature."""
        return self._name

    @property
    def feature_type(self) -> Literal["numerical", "categorical"]:
        """
        The type of Feature.

        Note:
            Literal of "numerical" or "categorical"
        """
        return self._type

    # -------- SETTERS -------- # noqa

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @feature_type.setter
    def feature_type(self, value: Literal["numerical", "categorical"]) -> None:
        self._type = value
