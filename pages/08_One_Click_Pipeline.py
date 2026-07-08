import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import pickle
import json
import re
import copy
import importlib.util
from datetime import datetime
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Import data_augmentation module
spec = importlib.util.spec_from_file_location("data_augmentation_temp", os.path.join(PARENT_DIR, "data_augmentation.py"))
data_augmentation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_augmentation_module)
DataAugmentor = data_augmentation_module.DataAugmentor

from midel import ReadingData, AutoModelSelector, WaveletDenoiser
from target_utils import prepare_targets, is_multi_target, select_target
from preprocess import SpectralData
from pca import DimensionalityReduction
from spectra_specific.NIRSpectra import NIRPreprocessingOptimizer
from spectra_specific.RamanSpectra1 import RamanPreprocessingOptimizer
from spectra_specific.FTIRSpectra import FTIRPreprocessingOptimizer


# ── helpers ──────────────────────────────────────────────────────────────────

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, set):
        return [convert_numpy_types(v) for v in sorted(obj, key=str)]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


def flatten_y(y):
    # Flatten to 1-D only for a single target; keep multi-target data 2-D so
    # multi-output models work instead of corrupting the targets.
    return prepare_targets(y)


def _run_preprocessing(data_type, X_train, X_test, X_train_np, X_test_np,
                       y_train_flat, y_test_flat, cv_folds, n_trials, all_params):
    """Run the automated preprocessing optimizer and apply the best pipeline.

    Returns (X_train_df, X_test_df, n_feat, optimizer, results). On failure it
    shows an error and returns (None, None, None, None, None).
    """
    if data_type in ("General", "Raman Spectroscopy"):
        # The Raman optimizer explores the full general spectroscopy toolkit
        # (baseline / smoothing / normalization / centering / derivatives), so
        # it also backs the "General" option.
        optimizer = RamanPreprocessingOptimizer(
            X=X_train_np, y=y_train_flat,
            cv_folds=cv_folds, n_trials=n_trials, random_state=42,
        )
    elif data_type == "NIR Spectroscopy":
        optimizer = NIRPreprocessingOptimizer(
            X_train=X_train_np, X_test=X_test_np,
            y_train=y_train_flat, y_test=y_test_flat,
            cv_folds=cv_folds, n_trials=n_trials, random_state=42,
        )
    elif data_type == "FTIR Spectroscopy":
        optimizer = FTIRPreprocessingOptimizer(
            X=X_train_np, y=y_train_flat,
            cv_folds=cv_folds, n_trials=n_trials, test_size=0.2, random_state=42,
        )
    else:
        st.error(f"No optimizer available for technique: {data_type}")
        return None, None, None, None, None

    progress_bar = st.progress(0)
    status_line = st.empty()

    def _progress_cb(pct, msg):
        try:
            progress_bar.progress(min(100, max(0, int(pct))) / 100)
            status_line.text(f"Preprocessing: {msg}")
        except Exception:
            pass

    results = optimizer.optimize(progress_callback=_progress_cb)
    progress_bar.empty()
    status_line.empty()

    if not results.get("success", False):
        err_msg = results.get('error') or 'Unknown error (no message captured)'
        st.error(f"Preprocessing failed: {err_msg}")
        with st.expander("Full error details"):
            st.code(str(results))
        return None, None, None, None, None

    if not hasattr(optimizer, 'apply_best_preprocessing'):
        st.error("Optimizer does not support applying the pipeline.")
        return None, None, None, None, None

    import inspect
    sig = inspect.signature(optimizer.apply_best_preprocessing)
    if 'fit_mode' in sig.parameters:
        X_train_processed = optimizer.apply_best_preprocessing(X_train_np, fit_mode=True)
        X_test_processed = optimizer.apply_best_preprocessing(X_test_np, fit_mode=False)
    else:
        X_train_processed = optimizer.apply_best_preprocessing(X_train_np)
        X_test_processed = optimizer.apply_best_preprocessing(X_test_np)

    n_feat_train = X_train_processed.shape[1]
    n_feat_test = X_test_processed.shape[1]
    if n_feat_train != n_feat_test:
        n_feat = min(n_feat_train, n_feat_test)
        X_train_processed = X_train_processed[:, :n_feat]
        X_test_processed = X_test_processed[:, :n_feat]
    else:
        n_feat = n_feat_train

    col_names = list(X_train.columns[:n_feat]) if n_feat <= len(X_train.columns) else [f"f_{i}" for i in range(n_feat)]
    X_train_df = pd.DataFrame(X_train_processed, columns=col_names, index=X_train.index[:X_train_processed.shape[0]])
    X_test_df = pd.DataFrame(X_test_processed, columns=col_names, index=X_test.index[:X_test_processed.shape[0]])

    all_params["preprocessing"] = {
        "technique": data_type,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "cv_score": results.get("cv_score"),
        "best_pipeline": results.get("best_pipeline"),
    }

    st.success(f"Preprocessing done — Best CV R²: {results.get('cv_score', 0):.4f}")

    with st.expander("Preprocessing details"):
        st.write("**Best pipeline:**")
        for i, step in enumerate(results.get("best_pipeline", [])):
            st.write(f"  {i+1}. {step.get('method', '?')} — {step.get('params', {})}")
        if "all_model_results" in results:
            st.dataframe(pd.DataFrame(results["all_model_results"]), use_container_width=True)

    try:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Before preprocessing", "After preprocessing"))
        for i in range(min(15, X_train_np.shape[0])):
            fig.add_trace(go.Scatter(y=X_train_np[i], mode='lines', line={'color': 'blue', 'width': 1}, opacity=0.5, showlegend=False), row=1, col=1)
        for i in range(min(15, X_train_processed.shape[0])):
            fig.add_trace(go.Scatter(y=X_train_processed[i], mode='lines', line={'color': 'red', 'width': 1}, opacity=0.5, showlegend=False), row=1, col=2)
        fig.update_xaxes(title_text="Feature index", row=1, col=1)
        fig.update_yaxes(title_text="Intensity", row=1, col=1)
        fig.update_xaxes(title_text="Feature index", row=1, col=2)
        fig.update_yaxes(title_text="Intensity", row=1, col=2)
        fig.update_layout(height=400, margin={'l': 20, 'r': 20, 't': 40, 'b': 20}, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass

    return X_train_df, X_test_df, n_feat, optimizer, results


# ── main page ────────────────────────────────────────────────────────────────

def main():
    st.title("One-Click Automated ")
    st.markdown(
        "Upload your data, pick a few options, and the entire **preprocessing → "
        "model training** pipeline runs automatically."
    )

    _p = "ocp_"

    st.header("1 · Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV, XLSX or TXT file", type=["csv", "xlsx", "txt"], key=f"{_p}uploader"
    )

    if uploaded_file is not None:
        if f"{_p}raw_data" not in st.session_state or st.session_state.get(f"{_p}last_file") != uploaded_file.name:
            try:
                rd = ReadingData()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                data = rd.read_data(tmp_path)
                os.unlink(tmp_path)

                st.session_state[f"{_p}raw_data"] = data.copy()
                st.session_state[f"{_p}last_file"] = uploaded_file.name
                
                for k in [f"{_p}pipeline_done", f"{_p}trained_model", f"{_p}predictions",
                          f"{_p}model_params", f"{_p}preprocessing_params"]:
                    st.session_state.pop(k, None)
                st.success(f"Loaded **{uploaded_file.name}** — {data.shape[0]} rows × {data.shape[1]} columns")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return
        data = st.session_state[f"{_p}raw_data"]
        st.dataframe(data.head(), use_container_width=True)
    else:
        st.info("Please upload a data file to begin.")
        return

    st.header("2 · Select Targets")

    drop_cols = st.multiselect("Columns to drop (optional):", data.columns.tolist(), key=f"{_p}drop_cols")
    working_data = data.drop(columns=drop_cols) if drop_cols else data.copy()

    target_columns = st.multiselect(
        "Select **target** column(s):", working_data.columns.tolist(), key=f"{_p}targets"
    )
    if not target_columns:
        st.warning("Please select at least one target column to continue.")
        return

    X_full = working_data.drop(columns=target_columns)
    y_full = working_data[target_columns]

    # Spectral features must be numeric. Drop non-numeric feature columns
    # (sample IDs, labels) that would break preprocessing and modelling.
    _Xnum = X_full.apply(pd.to_numeric, errors='coerce')
    _bad_cols = [c for c in X_full.columns if _Xnum[c].isna().all()]
    if _bad_cols:
        X_full = X_full.drop(columns=_bad_cols)
        st.warning("Dropped non-numeric feature column(s): " + ", ".join(map(str, _bad_cols)))

    st.info(f"Features: {X_full.shape[1]} | Target(s): {len(target_columns)} | Samples: {X_full.shape[0]}")

    # Multi-target handling. Preprocessing optimization always needs a single
    # target; the model can either predict all targets jointly or just one.
    model_target_mode = "Combined (one multi-output model)"
    opt_target = None
    if len(target_columns) > 1:
        opt_target = st.selectbox(
            "Multiple targets selected - target to optimize preprocessing against:",
            target_columns, key=f"{_p}opt_target"
        )
        model_target_mode = st.radio(
            "How should the model handle the targets?",
            ("Combined (one multi-output model)", "Single target"),
            key=f"{_p}model_target_mode"
        )
    st.session_state[f"{_p}opt_target_val"] = opt_target
    st.session_state[f"{_p}model_target_mode_val"] = model_target_mode

    st.header("3 · Spectral Data Type")
    data_type = st.selectbox(
        "Select your spectral technique:",
        ["General", "Raman Spectroscopy", "FTIR Spectroscopy", "NIR Spectroscopy"],
        key=f"{_p}data_type",
    )

    technique_descriptions = {
        "General": "General spectroscopy preprocessing: auto-selects from baseline correction, "
                   "smoothing, SNV/MSC/Detrend/Area/Vector/Min-max/Pareto normalization, centering "
                   "and SG derivatives. Works across techniques.",
        "Raman Spectroscopy": "Baseline correction, smoothing, SNV/MSC normalization, SG derivatives",
        "FTIR Spectroscopy": "Peak-based normalization, area normalization, FTIR-optimized baseline correction",
        "NIR Spectroscopy": "Advanced scatter correction (SNV, MSC, Detrending), derivatives, noise reduction",
    }
    st.caption(f"{technique_descriptions[data_type]}")

    st.header("4 · Train-Test Split")
    test_size = st.slider("Test set fraction:", 0.10, 0.50, 0.30, 0.05, key=f"{_p}test_size")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training samples", int(X_full.shape[0] * (1 - test_size)))
    with col2:
        st.metric("Test samples", int(X_full.shape[0] * test_size))

    st.header("5 · Preprocessing Optimization")
    skip_preprocessing = st.checkbox(
        "Skip preprocessing (data is already preprocessed)",
        key=f"{_p}skip_prep",
        help="Use the uploaded data as-is and go straight to model training. "
             "Enable this only if your spectra are already preprocessed.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_trials = st.selectbox("Preprocessing trials:", [10, 25, 50, 100, 150, 200], index=1,
                                key=f"{_p}trials", disabled=skip_preprocessing)
    with col2:
        cv_folds = st.selectbox("CV folds:", [3, 5, 7, 10], index=1,
                                key=f"{_p}cv", disabled=skip_preprocessing)
    with col3:
        model_trials = st.number_input(
            "Model tuning iterations:", min_value=1, max_value=300, value=20, step=5,
            key=f"{_p}model_trials",
            help="Optuna trials per candidate model during automated model selection. "
                 "Higher = better tuning but slower.")
    if skip_preprocessing:
        st.info("Preprocessing will be skipped — the uploaded data is used as-is for model training.")

    st.header("6 · Dimensionality Reduction (optional)")
    enable_dim_reduction = st.checkbox("Enable dimensionality reduction (PCA)", key=f"{_p}enable_dr")
    dr_params = {}
    if enable_dim_reduction:
        col1, col2 = st.columns(2)
        with col1:
            dr_method = st.selectbox("PCA method:", ["variance", "elbow", "fixed"], key=f"{_p}dr_method")
        with col2:
            if dr_method == "variance":
                dr_params["variance_threshold"] = st.slider("Variance threshold:", 0.80, 0.99, 0.95, 0.01, key=f"{_p}dr_var")
            elif dr_method == "fixed":
                dr_params["n_components"] = st.slider("Number of components:", 2, 50, 10, key=f"{_p}dr_nc")
        dr_params["method"] = dr_method
        enable_scaling = st.checkbox("Apply standard scaling before PCA", value=True, key=f"{_p}dr_scale")
        dr_params["scale"] = enable_scaling

    st.header("7 · Data Augmentation (optional)")
    enable_augmentation = st.checkbox("Enable data augmentation", key=f"{_p}enable_aug")
    aug_params = {}
    if enable_augmentation:
        aug_method = st.selectbox(
            "Augmentation method:",
            ["Gaussian Noise", "Mixup", "Spectral Shift", "Add Spectra"],
            key=f"{_p}aug_method",
        )
        aug_params["method"] = aug_method
        col1, col2 = st.columns(2)
        with col1:
            aug_params["num_copies"] = st.number_input(
                "Number of synthetic samples:", 1, 2000, 100, key=f"{_p}aug_copies"
            )
        with col2:
            if aug_method == "Gaussian Noise":
                aug_params["std"] = st.number_input("Noise std:", 0.001, 1.0, 0.01, 0.001, key=f"{_p}aug_std")
            elif aug_method == "Mixup":
                aug_params["alpha"] = st.slider("Mixup alpha:", 0.1, 1.0, 0.4, 0.1, key=f"{_p}aug_alpha")
            elif aug_method == "Spectral Shift":
                aug_params["shift_range"] = st.slider("Shift range:", 1, 20, 5, key=f"{_p}aug_shift")

    st.header("8 · Model Training")
    task_type = st.radio("Task type:", ["Regression", "Classification"], horizontal=True, key=f"{_p}task_type")

    st.divider()
    run_clicked = st.button("Run Automated Pipeline", type="primary", use_container_width=True, key=f"{_p}run")

    if run_clicked:
        overall_progress = st.progress(0)
        status = st.empty()
        all_params = {}

        try:
            status.info("Splitting data...")
            overall_progress.progress(5)

            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full, test_size=test_size, random_state=42
            )
            all_params["train_test_split"] = {"test_size": test_size, "random_state": 42}
            st.success(f"Split: {X_train.shape[0]} train / {X_test.shape[0]} test")

            X_train_np = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
            X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)

            # The preprocessing optimizer scores against a single target.
            if opt_target is not None and is_multi_target(y_train):
                y_train_flat = flatten_y(select_target(y_train, opt_target))
                y_test_flat = flatten_y(select_target(y_test, opt_target))
            else:
                y_train_flat = flatten_y(y_train)
                y_test_flat = flatten_y(y_test)

            if np.any(np.isnan(X_train_np)) or np.any(np.isinf(X_train_np)):
                st.error("Training data contains NaN or Inf values. Please clean your data.")
                return
            if np.any(np.isnan(y_train_flat)) or np.any(np.isinf(y_train_flat)):
                st.error("Target data contains NaN or Inf values.")
                return

            optimizer = None

            if skip_preprocessing:
                # Data is already preprocessed — use it as-is, no optimizer.
                status.info("Skipping preprocessing (data already preprocessed)...")
                overall_progress.progress(30)
                X_train_df = X_train.copy()
                X_test_df = X_test.copy()
                n_feat = X_train_df.shape[1]
                all_params["preprocessing"] = {"applied": False, "skipped": True}
                st.success("Preprocessing skipped — using uploaded data as-is.")
            else:
                status.info(f"Running automated {data_type.split()[0]} preprocessing ({n_trials} trials)...")
                overall_progress.progress(10)
                X_train_df, X_test_df, n_feat, optimizer, results = _run_preprocessing(
                    data_type, X_train, X_test, X_train_np, X_test_np,
                    y_train_flat, y_test_flat, cv_folds, n_trials, all_params,
                )
                if X_train_df is None:
                    return

            overall_progress.progress(40)

            if enable_dim_reduction:
                status.info("Applying dimensionality reduction...")
                overall_progress.progress(50)

                dim_reducer = DimensionalityReduction(
                    X_train=X_train_df, X_test=X_test_df,
                    y_train=y_train, y_test=y_test,
                )
                if dr_params.get("scale", True):
                    dim_reducer.apply_scaling(scaling_method="standard")

                method = dr_params.get("method", "variance")
                if method == "variance":
                    X_train_dr, X_test_dr, n_comp = dim_reducer.pca_analysis(
                        method="variance",
                        variance_threshold=dr_params.get("variance_threshold", 0.95),
                        use_scaled=True,
                    )
                elif method == "elbow":
                    X_train_dr, X_test_dr, n_comp = dim_reducer.pca_analysis(
                        method="elbow", use_scaled=True,
                    )
                else:
                    X_train_dr, X_test_dr, n_comp = dim_reducer.pca_analysis(
                        method="fixed",
                        n_components=dr_params.get("n_components", 10),
                        use_scaled=True,
                    )

                X_train_df = X_train_dr if X_train_dr is not None else dim_reducer.X_train
                X_test_df = X_test_dr if X_test_dr is not None else dim_reducer.X_test

                all_params["dimensionality_reduction"] = {
                    "method": method,
                    "n_components": int(n_comp),
                    **{k: v for k, v in dr_params.items() if k != "method"},
                }
                st.success(f"Dimensionality reduction: {n_feat} → {n_comp} components")
            else:
                all_params["dimensionality_reduction"] = {"applied": False}

            overall_progress.progress(60)

            if enable_augmentation:
                status.info("Augmenting training data...")
                overall_progress.progress(65)

                aug_X = X_train_df.copy()
                aug_y = y_train.copy()

                augmentor = DataAugmentor(aug_X, aug_y)
                method = aug_params.get("method", "Gaussian Noise")
                nc = int(aug_params.get("num_copies multiply", 2))

                if method == "Gaussian Noise":
                    aug_X, aug_y = augmentor.gaussian_noise(num_copies=nc, std=float(aug_params.get("std", 0.01)))
                elif method == "Mixup":
                    aug_X, aug_y = augmentor.mixup(num_copies=nc, alpha=float(aug_params.get("alpha", 0.4)))
                elif method == "Spectral Shift":
                    shift = min(int(aug_params.get("shift_range", 5)), X_train_df.shape[1] // 10)
                    if shift <= 0:
                        shift = 1
                    aug_X, aug_y = augmentor.spectral_shift(num_copies=nc, shift_range=shift)
                elif method == "Add Spectra":
                    aug_X, aug_y = augmentor.add_spectra(num_copies=nc)

                orig_size = X_train_df.shape[0]
                X_train_df = aug_X
                y_train = aug_y

                all_params["augmentation"] = {
                    "method": method,
                    "num_copies": nc,
                    "original_samples": orig_size,
                    "augmented_samples": int(X_train_df.shape[0]),
                }
                st.success(f"Augmentation: {orig_size} → {X_train_df.shape[0]} training samples")
            else:
                all_params["augmentation"] = {"applied": False}

            overall_progress.progress(70)

            status.info("Training models (automated selection)...")
            overall_progress.progress(75)

            # Honour the chosen multi-target model mode: a single multi-output
            # model over all targets, or one model for a single chosen target.
            if model_target_mode == "Single target" and is_multi_target(y_train) and opt_target is not None:
                y_train_model = flatten_y(select_target(y_train, opt_target))
                y_test_model = flatten_y(select_target(y_test, opt_target))
            else:
                y_train_model = flatten_y(y_train)
                y_test_model = flatten_y(y_test)

            auto = AutoModelSelector(n_trials=int(model_trials))
            # run_regression/run_classification return
            # (best_model_name, best_model, best_score, all_scores).
            if task_type == "Regression":
                best_name, best_model, best_score, _ = auto.run_regression(X_train_df, X_test_df, y_train_model, y_test_model)
            else:
                best_name, best_model, best_score, _ = auto.run_classification(X_train_df, X_test_df, y_train_model, y_test_model)

            if best_model is None:
                st.error("Automated model selection failed: no model could be trained on this data.")
                return
            st.caption(f"Best model: {best_name} (score {best_score:.4f})")

            overall_progress.progress(95)

            status.info("Evaluating model...")
            predictions = best_model.predict(X_test_df)

            if task_type == "Regression":
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                r2 = r2_score(y_test_model, predictions)
                mae = mean_absolute_error(y_test_model, predictions)
                rmse = np.sqrt(mean_squared_error(y_test_model, predictions))

                metrics = {"r2_score": r2, "mae": mae, "rmse": rmse}
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² Score", f"{r2:.4f}")
                with col2:
                    st.metric("MAE", f"{mae:.4f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.4f}")

                try:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=y_test_model, y=predictions, mode='markers', marker={'color': 'blue', 'opacity': 0.6, 'line': {'color': 'black', 'width': 1}}, name='Predictions'))
                    lims_min = min(y_test_model.min(), predictions.min())
                    lims_max = max(y_test_model.max(), predictions.max())
                    fig.add_trace(go.Scatter(x=[lims_min, lims_max], y=[lims_min, lims_max], mode='lines', line={'color': 'red', 'dash': 'dash'}, name='Ideal'))
                    fig.update_layout(title="Predicted vs Actual", xaxis_title="Actual", yaxis_title="Predicted", template='plotly_white', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    pass
            else:
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(y_test_model, predictions)
                f1 = f1_score(y_test_model, predictions, average="weighted")
                metrics = {"accuracy": acc, "f1_score": f1}
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Accuracy", f"{acc:.4f}")
                with col2:
                    st.metric("F1 Score", f"{f1:.4f}")

            all_params["model"] = {
                "task_type": task_type.lower(),
                "training_method": "automated",
                "performance_metrics": convert_numpy_types(metrics),
            }

            st.session_state[f"{_p}trained_model"] = best_model
            st.session_state[f"{_p}predictions"] = predictions
            st.session_state[f"{_p}model_params"] = all_params
            st.session_state[f"{_p}preprocessing_params"] = all_params
            # Keep the FITTED preprocessing optimizer so prediction reproduces
            # training exactly (rebuilding from step params is only approximate
            # and produces feature-count mismatches).
            st.session_state[f"{_p}auto_optimizer"] = optimizer
            st.session_state[f"{_p}n_features_model"] = int(n_feat)
            st.session_state[f"{_p}X_train"] = X_train_df
            st.session_state[f"{_p}X_test"] = X_test_df
            st.session_state[f"{_p}y_train"] = y_train
            st.session_state[f"{_p}y_test"] = y_test
            st.session_state[f"{_p}pipeline_done"] = True

            overall_progress.progress(100)
            status.success("Pipeline complete!")

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            return

    if st.session_state.get(f"{_p}pipeline_done"):
        st.divider()
        st.header("9 · Save Model")

        user_name = st.session_state.get("user_name", "user")
        user_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', user_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model = st.session_state[f"{_p}trained_model"]
        all_params = st.session_state[f"{_p}model_params"]

        save_name = st.text_input("Model name:", value="OneClick_Model", key=f"{_p}save_name")

        if st.button("Save Model & Parameters", type="primary", key=f"{_p}save_btn"):
            try:
                model_filename = f"model_{user_name_clean}_{save_name}_{timestamp}.pkl"
                with open(model_filename, 'wb') as f:
                    pickle.dump(model, f)

                X_tr = st.session_state.get(f"{_p}X_train")
                y_tr = st.session_state.get(f"{_p}y_train")
                target_cols = target_columns

                params_to_save = {
                    "user_info": {
                        "user_name": user_name,
                        "creation_date": datetime.now().isoformat(),
                    },
                    "model_info": {
                        "model_name": save_name,
                        "timestamp": timestamp,
                        "model_filename": model_filename,
                    },
                    "pipeline_parameters": convert_numpy_types(all_params),
                    "data_info": {
                        "target_columns": [str(c) for c in target_cols],
                        "feature_columns": [str(c) for c in X_tr.columns] if X_tr is not None and hasattr(X_tr, "columns") else [],
                        "training_shape": list(X_tr.shape) if X_tr is not None else [],
                    },
                }

                json_filename = f"parameters_{user_name_clean}_{save_name}_{timestamp}.json"
                with open(json_filename, 'w') as f:
                    json.dump(params_to_save, f, indent=4, ensure_ascii=False, default=str)

                # Save the FITTED preprocessing optimizer so prediction reproduces
                # training preprocessing exactly (no feature-count mismatch).
                fitted_filename = None
                optimizer = st.session_state.get(f"{_p}auto_optimizer")
                if optimizer is not None:
                    fitted_objects = {
                        "auto_optimizer": optimizer,
                        "n_features_model": st.session_state.get(f"{_p}n_features_model"),
                    }
                    fitted_filename = f"fitted_objects_{user_name_clean}_{save_name}_{timestamp}.pkl"
                    with open(fitted_filename, 'wb') as f:
                        pickle.dump(fitted_objects, f)

                st.success(f"Saved! Model: `{model_filename}` | Params: `{json_filename}`")
                st.info("Upload **all three files** (model, parameters, fitted objects) on the "
                        "Prediction / Real-Time pages for exact, correct predictions.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    with open(model_filename, 'rb') as f:
                        st.download_button("Download Model (.pkl)", f.read(),
                                           file_name=os.path.basename(model_filename),
                                           mime="application/octet-stream")
                with col2:
                    with open(json_filename, 'r') as f:
                        st.download_button("Download Parameters (.json)", f.read(),
                                           file_name=os.path.basename(json_filename),
                                           mime="application/json")
                with col3:
                    if fitted_filename:
                        with open(fitted_filename, 'rb') as f:
                            st.download_button("Download Fitted Objects (.pkl)", f.read(),
                                               file_name=os.path.basename(fitted_filename),
                                               mime="application/octet-stream")
            except Exception as e:
                st.error(f"Error saving: {e}")

main()

from chatbot import render_chatbot
render_chatbot("08_One_Click_Pipeline")

