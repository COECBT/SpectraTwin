"""
Shared prediction helpers so that a model saved by EITHER the One-Click
pipeline (page 08) or the General/Full pipeline (pages 03 / 06) can be used for
prediction from the Model-Prediction page (07) and the Real-Time page (09).

The two pipelines save slightly different JSON layouts:

General / Full pipeline
    {
      "model_parameters":        {...},
      "preprocessing_parameters": {...},   # wavelet / spectral_steps / automated_* / ...
      "data_info": {"feature_columns": [...], "target_columns": [...]}
    }

One-Click pipeline
    {
      "model_info": {"model_name": ...},
      "pipeline_parameters": {
          "preprocessing": {"technique": ..., "best_pipeline": [ {method, params}, ... ]},
          "model":         {"task_type": ..., "performance_metrics": {...}}
      },
      "data_info": {"target_columns": [...]}
    }

``normalize_parameters`` maps either of these to one canonical structure so the
rest of the code only has to deal with a single shape.
"""

import os
import tempfile

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin

from preprocess import SpectralData
from midel import WaveletDenoiser


class RandomConvFeatures(BaseEstimator, TransformerMixin):
    """1D-CNN-style feature extractor (ROCKET-inspired), TensorFlow-free.

    Applies many random 1D convolution kernels and pools each kernel's output to
    two features (max and PPV = proportion of positive values), capturing local
    shift-invariant patterns like a CNN without training conv weights.

    NOTE: this lives in prediction_utils (an importable module, not a Streamlit
    page) so that models trained with it in the NN Builder can be UNPICKLED on
    the Prediction / Real-Time pages.
    """
    def __init__(self, n_kernels=200, kernel_sizes=(7, 9, 11), random_state=42):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        self.kernels_ = []
        for _ in range(self.n_kernels):
            klen = int(rng.choice(self.kernel_sizes))
            w = rng.normal(size=klen).astype(np.float32)
            w -= w.mean()
            b = float(rng.uniform(-1, 1))
            self.kernels_.append((w, b))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        feats = []
        for w, b in self.kernels_:
            klen = len(w)
            Xp = X if X.shape[1] >= klen else np.pad(X, ((0, 0), (0, klen - X.shape[1])))
            win = sliding_window_view(Xp, klen, axis=1)
            conv = win @ w + b
            feats.append(conv.max(axis=1))
            feats.append((conv > 0).mean(axis=1))
        return np.column_stack(feats).astype(np.float32)


def _noop(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# Parameter normalisation
# ---------------------------------------------------------------------------
def detect_source(parameters):
    """Return 'one-click' or 'general' for a loaded parameters dict."""
    if not isinstance(parameters, dict):
        return "general"
    if "pipeline_parameters" in parameters and "preprocessing_parameters" not in parameters:
        return "one-click"
    return "general"


def normalize_parameters(parameters, source="auto"):
    """Return a canonical dict with keys: model_parameters,
    preprocessing_parameters, data_info, user_info, model_info, source."""
    parameters = parameters or {}
    if source == "auto":
        source = detect_source(parameters)

    if source == "one-click":
        pipe = parameters.get("pipeline_parameters", {}) or {}
        prep = pipe.get("preprocessing", {}) or {}
        model_blk = pipe.get("model", {}) or {}

        preprocessing_parameters = {}
        best_pipeline = prep.get("best_pipeline")
        if best_pipeline:
            preprocessing_parameters["automated_technique"] = prep.get("technique", "")
            preprocessing_parameters["automated_pipeline"] = best_pipeline

        model_parameters = {
            "model_name": parameters.get("model_info", {}).get("model_name", "One-Click Model"),
            "model_type": model_blk.get("task_type", ""),
            "training_method": model_blk.get("training_method", "automated"),
            "performance_metrics": model_blk.get("performance_metrics", {}),
        }
        return {
            "model_parameters": model_parameters,
            "preprocessing_parameters": preprocessing_parameters,
            "data_info": parameters.get("data_info", {}) or {},
            "user_info": parameters.get("user_info", {}) or {},
            "model_info": parameters.get("model_info", {}) or {},
            "source": "one-click",
        }

    # General / Full pipeline (already canonical)
    return {
        "model_parameters": parameters.get("model_parameters", {}) or {},
        "preprocessing_parameters": parameters.get("preprocessing_parameters", {}) or {},
        "data_info": parameters.get("data_info", {}) or {},
        "user_info": parameters.get("user_info", {}) or {},
        "model_info": parameters.get("model_info", {}) or {},
        "source": "general",
    }


# ---------------------------------------------------------------------------
# Preprocessing replay
# ---------------------------------------------------------------------------
def _apply_spectral_method(spectral, method, params):
    method_map = {
        'AsLS':               lambda: spectral.AsLS(lam=params['lam'], p=params['p'], niter=params['niter']),
        'Polyfit':            lambda: spectral.polyfit(order=params['order'], niter=params['niter']),
        'Pearson':            lambda: spectral.pearson(u=params['u'], v=params['v']),
        'Rolling':            lambda: spectral.rolling(window=params['window']),
        'Savitzky-Golay':     lambda: spectral.SGSmooth(window=params['window'], poly=params['poly']),
        'SNV':                lambda: spectral.snv(),
        'MSC':                lambda: spectral.msc(),
        'Detrend':            lambda: spectral.detrend(order=params['order']),
        'Area':               lambda: spectral.area(),
        'Peak Normalization': lambda: spectral.peaknorm(wavenumber=params['wave']),
        'Vector':             lambda: spectral.vector(),
        'Min-max':            lambda: spectral.minmax(min_val=params['minv'], max_val=params['maxv']),
        'Pareto':             lambda: spectral.pareto(),
        'Mean (spectrum)':    lambda: spectral.mean_center(option=False),
        'Mean (wavelength)':  lambda: spectral.mean_center(option=True),
        'Last Point':         lambda: spectral.lastpoint(),
        'Derivative_Subtract': lambda: spectral.subtract(spectra=params['subtract_idx']),
        'Derivative_Reset':   lambda: spectral.reset(),
        'SG Derivative':      lambda: spectral.SGDeriv(window=params['window'], poly=params['poly'], order=params['order']),
    }
    func = method_map.get(method)
    if func:
        func()
    return func is not None


def _build_technique_order(dim_params):
    order = []
    for key in dim_params.keys():
        if key.startswith('scaling_'):
            order.append(('Scaling', int(key.split('_')[-1]), key))
        elif key.startswith('pca_'):
            order.append(('PCA Analysis', int(key.split('_')[-1]), key))
        elif key.startswith('feature_selection_'):
            order.append(('Feature Selection', int(key.split('_')[-1]), key))
    order.sort(key=lambda x: x[1])
    return order


def _apply_manual_spectral_steps(spectral, spectral_params, log=_noop):
    technique_order = []
    for key in spectral_params.keys():
        parts = key.split('_')
        if key.startswith('trim_'):
            technique_order.append(('Trim', int(parts[1]), key))
        elif key.startswith('baseline_'):
            technique_order.append(('Baseline Correction', int(parts[1]), key))
        elif key.startswith('smoothing_'):
            technique_order.append(('Smoothing', int(parts[1]), key))
        elif key.startswith('normalization_'):
            technique_order.append(('Normalization', int(parts[1]), key))
        elif key.startswith('center_'):
            technique_order.append(('Center', int(parts[1]), key))
        elif key.startswith('derivative_'):
            technique_order.append(('Derivative', int(parts[-1]), key))
        elif key.startswith('sg_derivative_'):
            technique_order.append(('SG Derivative', int(parts[-1]), key))

    technique_order.sort(key=lambda x: x[1])

    for technique, idx, key in technique_order:
        params = spectral_params[key]
        log(f"    Applying {technique} (step {idx + 1})...")

        if technique == 'Trim':
            if params['type'] == "Trim":
                spectral.trim(start=params['start'], end=params['end'])
            else:
                spectral.invtrim(start=params['start'], end=params['end'])
        elif technique == 'Baseline Correction':
            for method in params['methods']:
                if method == "AsLS":
                    p = params['parameters'][f'AsLS_{idx}']
                    spectral.AsLS(lam=p['lam'], p=p['p'], niter=int(p['niter']))
                elif method == "Polyfit":
                    p = params['parameters'][f'Polyfit_{idx}']
                    spectral.polyfit(order=int(p['order']), niter=int(p['niter']))
                elif method == "Pearson":
                    p = params['parameters'][f'Pearson_{idx}']
                    spectral.pearson(u=int(p['u']), v=int(p['v']))
        elif technique == 'Smoothing':
            for method in params['methods']:
                if method == "Rolling":
                    p = params['parameters'][f'Rolling_{idx}']
                    spectral.rolling(window=int(p['window']))
                elif method == "Savitzky-Golay":
                    p = params['parameters'][f'SG_{idx}']
                    spectral.SGSmooth(window=int(p['window']), poly=int(p['poly']))
        elif technique == 'Normalization':
            for method in params['methods']:
                if method == "SNV":
                    spectral.snv()
                elif method == "MSC":
                    spectral.msc()
                elif method == "Detrend":
                    p = params['parameters'][f'Detrend_{idx}']
                    spectral.detrend(order=p['order'])
                elif method == "Area":
                    spectral.area()
                elif method == "Peak Normalization":
                    p = params['parameters'][f'Peak_{idx}']
                    spectral.peaknorm(wavenumber=p['wave'])
                elif method == "Vector":
                    spectral.vector()
                elif method == "Min-max":
                    p = params['parameters'][f'Minmax_{idx}']
                    spectral.minmax(min_val=p['minv'], max_val=p['maxv'])
                elif method == "Pareto":
                    spectral.pareto()
        elif technique == 'Center':
            for method in params['methods']:
                if method == 'Mean (spectrum)':
                    spectral.mean_center(option=False)
                elif method == 'Mean (wavelength)':
                    spectral.mean_center(option=True)
                elif method == 'Last Point':
                    spectral.lastpoint()
        elif technique == 'Derivative':
            for option in params['options']:
                if option == "Subtract":
                    spectral.subtract(spectra=params['parameters']['subtract_idx'])
                elif option == "Reset":
                    spectral.reset()
        elif technique == 'SG Derivative':
            spectral.SGDeriv(window=int(params['window']), poly=int(params['poly']), order=int(params['order']))


def apply_preprocessing(data, preprocessing_params, fitted_objects=None, log=_noop):
    """Replay the saved preprocessing pipeline on new raw data."""
    processed_data = data.copy().astype(float)
    fitted_objects = fitted_objects or {}
    preprocessing_params = preprocessing_params or {}

    # Wavelet denoising
    if preprocessing_params.get('wavelet', {}).get('applied', False):
        log("  -> Applying wavelet denoising...")
        params = preprocessing_params['wavelet']
        fitted_denoiser = fitted_objects.get('wavelet_denoiser')
        if fitted_denoiser is not None:
            processed_data = fitted_denoiser.transform(processed_data)
        else:
            denoiser = WaveletDenoiser(
                wavelet=params['wavelet'], level=params['level'],
                threshold_mode=params['threshold_mode'])
            denoiser.fitted_threshold_ = params['fitted_threshold']
            processed_data = denoiser.transform(processed_data)

    # Trim steps
    trim_steps = [s for s in preprocessing_params.get('spectral_steps', [])
                  if s.startswith('Trim:') or s.startswith('Inverse Trim:')]
    if trim_steps:
        log("  -> Applying trimming...")
        processed_data = _replay_on_spectral(processed_data, lambda spectral: [
            (spectral.trim(start=float(s.split(': ')[1].split(' - ')[0]),
                           end=float(s.split(': ')[1].split(' - ')[1]))
             if s.startswith('Trim:')
             else spectral.invtrim(start=float(s.split(': ')[1].split(' - ')[0]),
                                   end=float(s.split(': ')[1].split(' - ')[1])))
            for s in trim_steps])

    # Automated pipeline (used by both automated preprocessing and One-Click)
    if 'automated_technique' in preprocessing_params and 'automated_pipeline' in preprocessing_params:
        technique = preprocessing_params['automated_technique']
        log(f"  -> Applying automated {technique} pipeline...")
        auto_optimizer = fitted_objects.get('auto_optimizer')
        if auto_optimizer is not None and hasattr(auto_optimizer, 'apply_best_preprocessing'):
            import inspect
            vals = processed_data.values if hasattr(processed_data, 'values') else np.asarray(processed_data)
            sig = inspect.signature(auto_optimizer.apply_best_preprocessing)
            if 'fit_mode' in sig.parameters:
                result = auto_optimizer.apply_best_preprocessing(vals, fit_mode=False)
            else:
                result = auto_optimizer.apply_best_preprocessing(vals)
            if result.shape[1] <= processed_data.shape[1]:
                processed_data = pd.DataFrame(result, columns=list(processed_data.columns[:result.shape[1]]))
            else:
                processed_data = pd.DataFrame(result, columns=[f"feature_{i}" for i in range(result.shape[1])])
            log("    Used fitted optimizer from training (exact reproduction).")
        else:
            if preprocessing_params.get('automated_pipeline'):
                log("    [warn] No fitted optimizer uploaded - reconstructing from step "
                    "parameters (approximate). Upload the fitted objects (.pkl) for exact results.")
            def _steps(spectral):
                try:
                    spectral.spc.columns = spectral.spc.columns.astype(float)
                    spectral.wav = spectral.spc.columns.copy()
                    spectral._wav_raw = spectral.spc.columns.copy()
                except (ValueError, TypeError):
                    pass
                for step in preprocessing_params['automated_pipeline']:
                    method = step['method']
                    params = step.get('params', {})
                    if not _apply_spectral_method(spectral, method, params):
                        log(f"    [warn] unknown method '{method}' - skipped")
            processed_data = _replay_on_spectral(processed_data, _steps)

    # Manual spectral steps
    elif 'spectral_steps' in preprocessing_params and preprocessing_params.get('spectral_parameters'):
        log("  -> Applying manual spectral preprocessing...")
        spectral_params = preprocessing_params['spectral_parameters']
        processed_data = _replay_on_spectral(
            processed_data, lambda spectral: _apply_manual_spectral_steps(spectral, spectral_params, log))

    # Dimensionality reduction (fitted objects strongly recommended)
    dim_params = preprocessing_params.get('dimensionality_parameters')
    if dim_params and preprocessing_params.get('dimensionality_steps'):
        log("  -> Applying dimensionality reduction...")
        processed_data = _apply_dimensionality(processed_data, dim_params, fitted_objects, log)

    return processed_data.astype(float)


def _replay_on_spectral(processed_data, step_fn):
    """Write the current frame to a temp CSV, load it as SpectralData, run
    step_fn(spectral), and return the resulting frame."""
    frame = processed_data.copy()
    try:
        frame.columns = [float(c) for c in frame.columns]
    except (ValueError, TypeError):
        pass
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        temp_path = tmp.name
        frame.to_csv(temp_path, index=False)
    try:
        spectral = SpectralData(temp_path)
        step_fn(spectral)
        return spectral.spc.copy()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _apply_dimensionality(processed_data, dim_params, fitted_objects, log=_noop):
    fitted_scaler = fitted_objects.get('dim_reducer_scaler')
    fitted_reducer = fitted_objects.get('dim_reducer_reducer')
    for technique, idx, key in _build_technique_order(dim_params):
        params = dim_params[key]
        vals = processed_data.values if hasattr(processed_data, 'values') else np.asarray(processed_data)
        if technique == 'Scaling':
            if fitted_scaler is not None:
                processed_data = pd.DataFrame(fitted_scaler.transform(vals))
            else:
                from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                scaler = {'standard': StandardScaler, 'minmax': MinMaxScaler,
                          'robust': RobustScaler}.get(params.get('method', 'standard'), StandardScaler)()
                processed_data = pd.DataFrame(scaler.fit_transform(vals))
                log("    [warn] re-fitted scaler on new data (no fitted scaler saved)")
        elif technique == 'PCA Analysis':
            if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                res = fitted_reducer.transform(vals)
                processed_data = pd.DataFrame(res, columns=[f'PC{i+1}' for i in range(res.shape[1])])
            else:
                log("    [warn] no fitted PCA saved - skipping PCA (feature mismatch likely)")
        elif technique == 'Feature Selection':
            if fitted_reducer is not None and hasattr(fitted_reducer, 'transform'):
                processed_data = pd.DataFrame(fitted_reducer.transform(vals))
            else:
                log("    [warn] no fitted feature selector saved - skipping")
    return processed_data


# ---------------------------------------------------------------------------
# Feature alignment
# ---------------------------------------------------------------------------
def _expected_n_features(model):
    """Number of input features the model was trained on, if discoverable."""
    if isinstance(model, dict):
        for key in ('regression_model', 'classification_model'):
            sub = model.get(key)
            n = getattr(sub, 'n_features_in_', None)
            if n is not None:
                return int(n)
        return None
    n = getattr(model, 'n_features_in_', None)
    return int(n) if n is not None else None


def _to_floatable(values):
    out = []
    for c in values:
        try:
            out.append(float(c))
        except (ValueError, TypeError):
            out.append(c)
    return out


def _align_features(processed, model, data_info, log=_noop):
    """Make ``processed`` have exactly the features the model expects.

    Strategy: (1) if it already matches, return as-is; (2) if the training
    ``feature_columns`` are known and all present, select/reorder to them
    (drops extra wavelengths, fixes order); (3) otherwise trim trailing extra
    columns to the model's expected count; (4) if it has too few, raise a
    clear error.
    """
    if not hasattr(processed, 'shape'):
        return processed
    cur = processed.shape[1]
    expected = _expected_n_features(model)

    if expected is not None and cur == expected:
        return processed

    feature_columns = (data_info or {}).get('feature_columns') or []
    if feature_columns and hasattr(processed, 'columns'):
        want = _to_floatable(feature_columns)
        proc_cols = _to_floatable(processed.columns)
        pos = {}
        for i, col in enumerate(proc_cols):
            pos.setdefault(col, i)
        if all(w in pos for w in want):
            idx = [pos[w] for w in want]
            log(f"  Aligned features to {len(idx)} training wavelengths (from {cur}).")
            return processed.iloc[:, idx]

    if expected is not None:
        if cur > expected:
            log(f"  [warn] Feature count {cur} > model's {expected}; trimming last {cur - expected}.")
            return processed.iloc[:, :expected] if hasattr(processed, 'iloc') else processed[:, :expected]
        if cur < expected:
            raise ValueError(
                f"Preprocessed data has {cur} features but the model expects {expected}. "
                "The new data likely differs from the training data, or a fitted "
                "objects (.pkl) file is needed to reproduce preprocessing exactly."
            )
    return processed


# ---------------------------------------------------------------------------
# End-to-end prediction
# ---------------------------------------------------------------------------
def run_prediction(model, parameters, new_data, fitted_objects=None,
                   skip_preprocessing=False, source="auto", log=_noop):
    """Preprocess ``new_data`` per the saved parameters and predict.

    Returns (predictions ndarray, processed_data DataFrame, normalized_params dict).
    """
    norm = normalize_parameters(parameters, source=source)

    if skip_preprocessing:
        processed = new_data.astype(float)
    else:
        pp = norm["preprocessing_parameters"]
        if pp:
            processed = apply_preprocessing(new_data, pp, fitted_objects, log=log)
        else:
            log("  [warn] no preprocessing parameters found - using raw data")
            processed = new_data.astype(float)

    # Align feature count with what the model expects. Small mismatches happen
    # when the new data has a slightly different number of wavelengths, or the
    # automated pipeline replays to a marginally different length.
    processed = _align_features(processed, model, norm["data_info"], log)

    prediction_data = processed.values.astype(float) if hasattr(processed, 'values') \
        else np.asarray(processed, dtype=float)

    if np.any(np.isnan(prediction_data)) or np.any(np.isinf(prediction_data)):
        raise ValueError("Data contains NaN/Inf after preprocessing.")

    # Two-part (zero-inflated) model saved as a dict
    if isinstance(model, dict) and 'classification_model' in model and 'regression_model' in model:
        clf = model['classification_model']
        reg = model['regression_model']
        binary = clf.predict(prediction_data)
        predictions = np.zeros(len(prediction_data), dtype=float)
        mask = binary == 1
        if np.any(mask):
            predictions[mask] = reg.predict(prediction_data[mask])
    else:
        predictions = model.predict(prediction_data)

    predictions = np.asarray(predictions)
    model_type = norm["model_parameters"].get("model_type", "")
    if model_type in ("regression", "zero_inflated"):
        predictions = np.maximum(predictions, 0)

    return predictions, processed, norm
