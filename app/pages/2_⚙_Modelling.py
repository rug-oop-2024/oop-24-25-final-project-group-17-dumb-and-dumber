import streamlit as st
import pandas as pd
from typing import List, TYPE_CHECKING
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types

from autoop.core.ml.model.regression.decision_tree_regressor import DecisionTreeRegressor
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.multiple_linear_regression import MultipleLinearRegression

if TYPE_CHECKING:
    from autoop.core.ml.model.classification.decision_tree_classifier import DecisionTreeClassifier
    from autoop.core.ml.model.classification.k_nearest_neighbors import KNearestNeighbors
    from autoop.core.ml.model.classification.naive_bayes import NaiveBayes


st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Initialize AutoML system and datasets
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

# Session state keys to persist steps
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

def select_and_train_model():
    # Step 1: Dataset Selection
    st.header("Dataset Selection")
    
    # Get dataset names for dropdown
    dataset_names = [dataset.name for dataset in datasets]

    col1, col2 = st.columns([3, 1])
    with col1:
        # Display dataset dropdown
        dataset_name = st.selectbox(
            "Select a dataset", 
            dataset_names, 
            index=dataset_names.index(st.session_state.selected_dataset) 
            if st.session_state.selected_dataset else 0,
            key='dataset_selectbox'
        )
    with col2:
        # Confirmation button to lock in the selection
        confirm = st.button("Confirm", key='dataset_confirm_button')
        if confirm:
            st.session_state.confirmed = True
            st.session_state.selected_dataset = dataset_name
            st.session_state.selected_features = False  # Reset feature selection when dataset changes

    # Step 2: Feature Selection
    if st.session_state.confirmed:
        curr_data: Dataset = next((d for d in datasets if d.name == st.session_state.selected_dataset), None)
        
        # Convert to Dataset if necessary and detect feature types
        if not isinstance(curr_data, Dataset):
            curr_data = Dataset.from_artifact(curr_data)
        # Detect feature types
        features: List[Feature] = detect_feature_types(curr_data)

        st.write(f"Selected dataset: {curr_data.name}")

        st.subheader("Features")

        # Use Feature objects directly in dropdowns
        input_features = st.multiselect(
            "Select Input Features", 
            features, 
            key='input_features_multiselect'
        )
        target_feature = st.selectbox(
            "Select Target Feature", 
            features, 
            key='target_feature_selectbox'
        )

        # Display the type of the selected target feature
        if target_feature:
            target_feature_type = target_feature.feature_type.capitalize()
            st.write(f"**Target Feature Type:** {target_feature_type}")

        # Confirm button to lock in the feature selection
        if st.button("Confirm Feature Selection", key='feature_confirm_button'):
            # Convert selected Feature objects to names for session state storage
            st.session_state.input_features = [feature.name for feature in input_features]
            st.session_state.target_feature = target_feature.name if target_feature else None

            # Validate selections
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
                st.success("Features selected successfully. Proceed to Model Selection.")

    # Step 3: Model Selection
    if st.session_state.selected_features:
        st.subheader("Model Selection")

        # Determine the type of the target feature
        target_feature_type = next(
            (f.feature_type for f in features if f.name == st.session_state.target_feature),
            None
        )
        if target_feature_type == "categorical":
            # Classification models
            model_options = [
                "Decision Tree Classifier", 
                "K-Nearest Neighbors Classifier",
                "Naive Bayes Classifier"
                ]
        elif target_feature_type == "numerical":
            # Regression models
            model_options = [
                "Decision Tree Regressor", 
                "Lasso Regression",
                "Multiple Linear Regression"
                ]
        else:
            st.error(f"Unsupported target feature type: {target_feature_type}")
            return

        selected_model = st.selectbox(
            "Select Model", 
            model_options, 
            index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model else 0,
            key='model_selectbox'
        )

        # Display model-specific hyperparameters
        if selected_model == "Decision Tree Classifier":
            criterion = st.selectbox("Criterion", ["gini", "entropy"], key='dtc_criterion')
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, key='dtc_max_depth')
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, key='dtc_min_samples_split')
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, key='dtc_min_samples_leaf')
            max_features = st.selectbox("Max Features", ["auto", "sqrt", "log2"], key='dtc_max_features')
            initialise_model = DecisionTreeClassifier(
                criterion=criterion, 
                max_depth=max_depth, 
                min_samples_split=min_samples_split, 
                min_samples_leaf=min_samples_leaf, 
                max_features=max_features
                )
            
        elif selected_model == "K-Nearest Neighbors Classifier":
            n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=20, value=5, key='knn_n_neighbors')
            initialise_model = KNearestNeighbors(n_neighbors=n_neighbors)
        
        elif selected_model == "Naive Bayes Classifier":
            st.write("priors and var_smoothing are not supported yet.")
            initialise_model = NaiveBayes()

        elif selected_model == "Decision Tree Regressor":
            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5, key='dtr_max_depth')
            criterion = st.selectbox("Criterion", ["mse", "friedman_mse", "mae", "poisson"], key='dtr_criterion')
            splitter = st.selectbox("Splitter", ["best", "random"], key='dtr_splitter')
            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, key='dtr_min_samples_split')
            min_samples_leaf = st.slider("Min Samples Leaf", min_value=1, max_value=10, value=1, key='dtr_min_samples_leaf')
            max_features = st.selectbox("Max Features", [None, "sqrt", "log2"], key='dtr_max_features')
            initialise_model = DecisionTreeRegressor(
                max_depth=max_depth,
                criterion=criterion,
                splitter=splitter,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features
            )
            
        elif selected_model == "Lasso Regression":
            alpha = st.slider("Alpha (Regularization Strength)", min_value=0.01, max_value=1.0, value=0.1, key='lasso_alpha')
            fit_intercept = st.checkbox("Fit Intercept", value=True, key='lasso_fit_intercept')
            max_iter = st.slider("Max Iterations", min_value=100, max_value=10000, value=1000, key='lasso_max_iter')
            tol = st.slider("Tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, key='lasso_tol')
            warm_start = st.checkbox("Warm Start", value=False, key='lasso_warm_start')
            positive = st.checkbox("Positive", value=False, key='lasso_positive')
            selection = st.selectbox("Selection", ["cyclic", "random"], key='lasso_selection')
            initialise_model = LassoRegression(
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                warm_start=warm_start,
                positive=positive,
                selection=selection
            )
        
        elif selected_model == "Multiple Linear Regression":
            initialise_model = MultipleLinearRegression()

        # Store selected model and configuration
        st.session_state.selected_model = selected_model

        if st.button("Train Model", key='train_model_button'):
            st.write(f"Training {selected_model} model with configuration:", model_config)
            # Prepare data for training
            data_df = curr_data.read()
            X = data_df[st.session_state.input_features]
            y = data_df[st.session_state.target_feature]
            # Initialize and run the pipeline
            # pipeline = Pipeline(...)
            # results = pipeline.execute()
            # Display results
            st.write("Model training and evaluation complete.")
            # st.write("Metrics:")
            # for metric, value in results["metrics"]:
            #     st.write(f"{metric.name}: {value}")
            # st.write("Predictions:")
            # st.write(results["predictions"])

select_and_train_model()
