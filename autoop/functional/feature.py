from typing import List

from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Assumption: only categorical and numerical features and no NaN values.

    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.

    Okay bois game logic right here.

    """
    # 1. Get the data from the dataset. Its a pandas dataframe. # noqa
    # 2. Get the columns of the dataframe. # noqa
    # 3. For each column, check if the column is numerical or categorical.
    # 4. Create a Feature object with the column name and type.
    # 5. Return a list of Feature objects.

    # 1.
    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be of type Dataset")
    data = dataset.read()

    # 2.
    # TODO: We might need to check the datatype here.
    columns = data.columns

    # 3. & 4.
    features = []
    for col in columns:
        if data[col].dtype in [int, float]:
            features.append(Feature(name=col, feature_type="numerical"))
        else:
            features.append(Feature(name=col, feature_type="categorical"))

    # 5.
    return features
