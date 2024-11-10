from typing import Literal
from enum import Enum
from sklearn.datasets import fetch_openml
import pandas as pd

class DATASET(Enum):
    """
    Datasets available.
    
    contains:
    - IRIS
    - ADULT
    """	

    IRIS = "iris"
    ADULT = "adult"
    WINE = "wine"
    DIGITS = "mnist_784"

class DatasetHandler:
    def __init__(self):
        pass
    
    def load_dataset(self, dataset: DATASET | str) -> pd.DataFrame:
        if isinstance(dataset, str):
            dataset = DATASET(dataset)
        data = fetch_openml(
            name=dataset.value, version=1, parser="auto")
        self.df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        return self.df
    
    
    
    def get_all_datasets(self) -> dict[str, ]:
        """
        Get all the datasets.

        NOTE: DONT DO THIS AT HOME :D
        PARENTAL ADVISORY: EXPLICIT CONTENT
        """
        self.datasets = {
            "iris": self.load_dataset(DATASET.IRIS),
            "adult": self.load_dataset(DATASET.ADULT),
            "wine": self.load_dataset(DATASET.WINE),
            "mnist_784": self.load_dataset(DATASET.DIGITS)
        }
        return self.datasets