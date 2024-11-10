from typing import TYPE_CHECKING, List

import pandas as pd
import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import METRICS, Metric, get_metric
from autoop.core.ml.model.classification.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors
from autoop.core.ml.model.classification.naive_bayes import NaiveBayes
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.decision_tree_regressor import (
    DecisionTreeRegressor,
)
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    """Write helper text with specific styling."""
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


def initialize_session_state():
    """Initialize the session state variables."""
    if "confirmed" not in st.session_state:
        st.session_state.confirmed = False
    if "selected_dataset" not in st.session_state:
        st.session_state.selected_dataset = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = False
    if "input_features" not in st.session_state:
        st.session_state.input_features = []
    if "target_feature" not in st.session_state:
        st.session_state.target_feature = None
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = None
    if "split" not in st.session_state:
        st.session_state.split = 0.8
    if "model_locked" not in st.session_state:
        st.session_state.model_locked = False
    if "selected_metrics" not in st.session_state:
        st.session_state.selected_metrics = []
    if "model_hyperparams" not in st.session_state:
        st.session_state.model_hyperparams = {}
    if "initialise_model" not in st.session_state:
        st.session_state.initialise_model = None
    if "features" not in st.session_state:
        st.session_state.features = []


def main():
    """Main function to run the Streamlit app."""
    st.write("# ⚙ Modelling")
    write_helper_text(
        "In this section, you can design a machine learning pipeline "
        "to train a model on a dataset."
    )

    # Initialize AutoML system and datasets # noqa
    automl = AutoMLSystem.get_instance()
    datasets = automl.registry.list(type="dataset")  # noqa: SC100

    initialize_session_state()

    select_and_train_model(datasets)


def select_dataset(datasets):
    """Allow the user to select a dataset from the list."""
    st.header("Dataset Selection")
    dataset_names = [dataset.name for dataset in datasets]

    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_name = st.selectbox(
            "Select a dataset",
            dataset_names,
            index=dataset_names.index(st.session_state.selected_dataset)
            if st.session_state.selected_dataset
            else 0,
            key="dataset_selectbox",
        )
    with col2:
        confirm = st.button("Confirm", key="dataset_confirm_button")
        if confirm:
            st.session_state.confirmed = True
            st.session_state.selected_dataset = dataset_name
            # Reset feature selection when dataset changes # noqa
            st.session_state.selected_features = False


def select_features(datasets):
    """Allow the user to select input and target features."""
    if st.session_state.confirmed:
        try:
            curr_data = next(
                (d for d in datasets if d.name == st.session_state.selected_dataset),
                None,
            )

            if not isinstance(curr_data, Dataset):
                curr_data = Dataset.from_artifact(curr_data)
            features = detect_feature_types(curr_data)

            st.session_state.features = features  # Store features in session state

            st.write(f"**Selected dataset:** {curr_data.name}")

            st.subheader("Features")

            input_features = st.multiselect(
                "Select Input Features", features, key="input_features_multiselect"
            )
            target_feature = st.selectbox(
                "Select Target Feature", features, key="target_feature_selectbox"
            )

            if target_feature:
                target_feature_type = target_feature.feature_type.capitalize()
                st.write(f"**Target Feature Type:** {target_feature_type}")

            if st.button("Confirm Feature Selection", key="feature_confirm_button"):
                st.session_state.input_features = input_features
                st.session_state.target_feature = target_feature

                # Reset model selection when features change
                st.session_state.selected_model = None
                st.session_state.model_locked = False
                st.session_state.model_hyperparams = {}

                if not st.session_state.input_features:
                    st.warning("Please select at least one input feature.")
                    st.session_state.selected_features = False
                elif st.session_state.target_feature in st.session_state.input_features:
                    st.warning("Target feature cannot be an input feature.")
                    st.session_state.selected_features = False
                elif not st.session_state.target_feature:
                    st.warning("Please select a target feature.")
                    st.session_state.selected_features = False
                else:
                    st.session_state.selected_features = True
                    st.success(
                        "Features selected successfully. Proceed to Model Selection."
                    )
        except Exception as e:
            st.error(f"An error occurred during feature selection: {e}")
            st.session_state.selected_features = False


def select_model():
    """Allow the user to select a machine learning model and its hyperparameters."""
    if st.session_state.selected_features:
        st.subheader("Model Selection")

        split = st.slider(
            "Train-Test Split Ratio",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state.split,
            key="split_slider",
        )
        st.session_state.split = split

        target_feature_type = st.session_state.target_feature.feature_type

        if target_feature_type == "categorical":
            model_options = [
                "Decision Tree Classifier",
                "K-Nearest Neighbors Classifier",
                "Naive Bayes Classifier",
            ]
        elif target_feature_type == "numerical":
            model_options = [
                "Decision Tree Regressor",
                "Lasso Regression",
                "Multiple Linear Regression",
            ]
        else:
            st.error(f"Unsupported target feature type: {target_feature_type}")
            return

        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=model_options.index(st.session_state.selected_model)
            if st.session_state.selected_model
            else 0,
            key="model_selectbox",
        )

        st.session_state.selected_model = selected_model

        # Display hyperparameters before locking model selection # noqa
        display_hyperparameters(selected_model, target_feature_type)

        # Now show the "Lock Model Selection" button
        if st.button("Lock Model Selection", key="lock_model_button"):
            st.session_state.model_locked = True

            # Initialize the model and store in session state
            st.session_state.initialise_model = initialize_model(selected_model)

        if st.session_state.model_locked:
            st.success(f"Model '{st.session_state.selected_model}' locked.")
            # Display the model's hyperparameters # noqa
            st.write("### Model Hyperparameters")
            st.json(st.session_state.model_hyperparams)
        else:
            st.info("Please lock the model selection to proceed.")


def display_hyperparameters(selected_model, target_feature_type):
    """
    Display hyperparameters based on the selected model.

    Note:
        This function could be improved by using a dictionary to store
        the hyperparameters for each model.
    """
    st.subheader("Hyperparameters")

    # Initialize the model based on the selected model
    if selected_model == "Decision Tree Classifier":
        criterion = st.selectbox("Criterion", ["gini", "entropy"], key="dtc_criterion")
        max_depth = st.slider(
            "Max Depth", min_value=1, max_value=20, value=5, key="dtc_max_depth"
        )
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=10,
            value=2,
            key="dtc_min_samples_split",
        )
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=10,
            value=1,
            key="dtc_min_samples_leaf",
        )
        max_features = st.selectbox(
            "Max Features", [None, "sqrt", "log2"], key="dtc_max_features"
        )

        # Store hyperparameters in session state # noqa
        st.session_state.model_hyperparams = {
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

    elif selected_model == "K-Nearest Neighbors Classifier":
        n_neighbors = st.slider(
            "Number of Neighbors",
            min_value=1,
            max_value=20,
            value=5,
            key="knn_n_neighbors",
        )
        st.session_state.model_hyperparams = {"n_neighbors": n_neighbors}

    elif selected_model == "Naive Bayes Classifier":
        st.write("Using default parameters for Naive Bayes Classifier.")
        st.session_state.model_hyperparams = {}

    elif selected_model == "Decision Tree Regressor":
        max_depth = st.slider(
            "Max Depth", min_value=1, max_value=20, value=5, key="dtr_max_depth"
        )
        criterion = st.selectbox(
            "Criterion",
            ["squared_error", "friedman_mse", "absolute_error", "poisson"],
            key="dtr_criterion",
        )
        splitter = st.selectbox("Splitter", ["best", "random"], key="dtr_splitter")
        min_samples_split = st.slider(
            "Min Samples Split",
            min_value=2,
            max_value=10,
            value=2,
            key="dtr_min_samples_split",
        )
        min_samples_leaf = st.slider(
            "Min Samples Leaf",
            min_value=1,
            max_value=10,
            value=1,
            key="dtr_min_samples_leaf",
        )
        max_features = st.selectbox(
            "Max Features", [None, "sqrt", "log2"], key="dtr_max_features"
        )
        st.session_state.model_hyperparams = {
            "max_depth": max_depth,
            "criterion": criterion,
            "splitter": splitter,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }

    elif selected_model == "Lasso Regression":
        alpha = st.slider(
            "Alpha (Regularization Strength)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            key="lasso_alpha",
        )
        fit_intercept = st.checkbox(
            "Fit Intercept", value=True, key="lasso_fit_intercept"
        )
        max_iter = st.slider(
            "Max Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            key="lasso_max_iter",
        )
        tol = st.number_input(
            "Tolerance",
            min_value=1e-6,
            max_value=1e-2,
            value=1e-4,
            format="%.6f",
            key="lasso_tol",
        )
        warm_start = st.checkbox("Warm Start", value=False, key="lasso_warm_start")
        positive = st.checkbox("Positive", value=False, key="lasso_positive")
        selection = st.selectbox(
            "Selection", ["cyclic", "random"], key="lasso_selection"
        )
        st.session_state.model_hyperparams = {
            "alpha": alpha,
            "fit_intercept": fit_intercept,
            "max_iter": max_iter,
            "tol": tol,
            "warm_start": warm_start,
            "positive": positive,
            "selection": selection,
        }

    elif selected_model == "Multiple Linear Regression":
        st.session_state.model_hyperparams = {}


def initialize_model(selected_model) -> Model:
    """Initialize the selected model with the specified hyperparameters."""
    hyperparams = st.session_state.get("model_hyperparams", {})
    if selected_model == "Decision Tree Classifier":
        return DecisionTreeClassifier(**hyperparams)
    elif selected_model == "K-Nearest Neighbors Classifier":
        return KNearestNeighbors(**hyperparams)
    elif selected_model == "Naive Bayes Classifier":
        return NaiveBayes(**hyperparams)
    elif selected_model == "Decision Tree Regressor":
        return DecisionTreeRegressor(**hyperparams)
    elif selected_model == "Lasso Regression":
        return LassoRegression(**hyperparams)
    elif selected_model == "Multiple Linear Regression":
        return MultipleLinearRegression(**hyperparams)
    else:
        return None


def select_metrics():
    """Allow the user to select evaluation metrics and build the pipeline."""
    st.subheader("Select Evaluation Metrics")
    # Use the list of metric names from the METRICS constant
    selected_metrics = st.multiselect(
        "Select Metrics", METRICS, key="metrics_multiselect"
    )
    st.session_state.selected_metrics = selected_metrics

    if st.button("Build Pipeline", key="train_model_button"):
        if not selected_metrics:
            st.warning("Please select at least one metric.")
        else:
            build_pipeline()


def build_pipeline():
    """Build and execute the machine learning pipeline based on user selections."""
    st.write("Training and evaluating the model...")
    try:
        initialise_model = st.session_state.get("initialise_model", None)
        if initialise_model is None:
            st.error(
                "Model is not initialized. Please complete the model selection \
                    and lock it before training."
            )
            return

        # Prepare data
        automl = AutoMLSystem.get_instance()
        datasets = automl.registry.list(type="dataset")
        curr_data = next(
            (d for d in datasets if d.name == st.session_state.selected_dataset), None
        )
        if not isinstance(curr_data, Dataset):
            curr_data = Dataset.from_artifact(curr_data)

        # Create Metric instances
        metrics = [get_metric(name) for name in st.session_state.selected_metrics]

        # Initialize Pipeline
        pipeline = Pipeline(
            metrics=metrics,
            dataset=curr_data,
            model=initialise_model,
            input_features=st.session_state.input_features,
            target_feature=st.session_state.target_feature,
            split=st.session_state.split,
        )

        # Execute Pipeline
        results = pipeline.execute()

        st.subheader("Pipeline Summary")
        st.code(str(pipeline), language="python")

        # Display metrics
        st.subheader("Evaluation Results")
        for metric_obj, score in results["metrics"]:
            st.write(f"**{metric_obj.__class__.__name__}:** {score}")

        # Optionally display predictions
        st.subheader("Predictions")
        predictions = results["predictions"]
        predictions_df = pd.DataFrame(predictions).T
        st.dataframe(predictions_df, height=100, width=800)

    except Exception as e:
        st.error(f"An error occurred during training: {e}")


def select_and_train_model(datasets):
    """Coordinate the selection and training of the machine learning model."""
    select_dataset(datasets)
    select_features(datasets)
    if st.session_state.confirmed and st.session_state.selected_features:
        select_model()
        if st.session_state.model_locked:
            select_metrics()


if __name__ == "__main__":
    main()
