import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from app.core.get_datasets import DatasetHandler, DATASET
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

handler = DatasetHandler()

# your code here

def selectExistingDataset():
    cutoff = 5

    datasetNames = [dataset.value for dataset in DATASET]
    dataset_name = st.selectbox("Select a dataset", datasetNames)
    
    # Load the selected dataset
    df = handler.load_dataset(dataset_name)
    
    # Layout with two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head()) 
    
    with col2:
        st.subheader("Features")
        for column in df.columns[:cutoff]:
            st.write(column)
        if len(df.columns) > cutoff:
            with st.expander("View more"):
                for column in df.columns[cutoff:]:
                    st.write(column)
    
    # Confirmation button
    if st.button("Confirm Dataset Selection"):
        st.write(f"Dataset '{dataset_name}' selected.")

        dataset = Dataset.from_dataframe(
            name=dataset_name,
            asset_path=f"{dataset_name}.csv",
            data=df,
        )
        automl.registry.register(dataset)
        st.success("Dataset registered successfully.")


def start():

    user_upload = st.file_uploader("Upload a CSV file", type=["csv"])
    if user_upload:
        df = pd.read_csv(user_upload)
        dataset = Dataset.from_dataframe(
            name="user_upload",
            asset_path="user_upload.csv",
            data=df,
        )
        automl.registry.register(dataset)
        st.success("Dataset uploaded successfully.")
        st.write(df.head())
        st.write(dataset)
    else:
        st.write("No file uploaded.")
        st.write("You can select one from the dowpdown below.")
        selectExistingDataset()

if __name__ == "__main__":
    start()
    