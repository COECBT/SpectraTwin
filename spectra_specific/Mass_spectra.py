import numpy as np
import pandas as pd
import optuna
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import logging
import traceback as _tb
from typing import Dict, List, Tuple, Optional, Any
import copy
from preprocess import SpectralData
import tempfile
import os
import io
import contextlib

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _tic_normalization(X: np.ndarray) -> np.ndarray:
    tic = np.sum(np.abs(X), axis=1, keepdims=True)
    return X / np.where(tic < 1e-12, 1.0, tic)

def _log1p_transform(X: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(X, 0, None))

def _sqrt_transform(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(X, 0, None))

def _background_subtraction(X: np.ndarray, quantile: float = 0.05) -> np.ndarray:
    bg = np.quantile(X, quantile, axis=1, keepdims=True)
    return np.clip(X - bg, 0, None)

class MassSpectralPreprocessingOptimizer:
    def __init__(self, X, y, mz_values=None, cv_folds=5, n_trials=50, test_size=0.2, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.mz_values = mz_values
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state
        self._validate_input_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=None
        )
        self.models = {
            'PLS': PLSRegression(),
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(random_state=self.random_state, verbosity=0),
        }
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        self.best_model_instance = None
        self.best_preprocessing_steps = []
        self.best_scaler = None
        self.optimization_history = []
        self.test_results = None
        self.all_model_results = {}
        self.current_trial = 0
        self.fitted_preprocessing_params = {}

    def _validate_input_data(self):
        if self.X.shape[0] != len(self.y):
            raise ValueError(f"X and y dimension mismatch: {self.X.shape[0]} vs {len(self.y)}")
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            raise ValueError("X contains NaN or infinite values")
        if np.any(np.isnan(self.y)) or np.any(np.isinf(self.y)):
            raise ValueError("y contains NaN or infinite values")
        if self.X.shape[1] < 2:
            raise ValueError("X must have at least 2 features")
        if self.X.shape[0] < 10:
            raise ValueError("Dataset too small. Need at least 10 samples.")

    def _apply_preprocessing_pipeline(self, X_data, pipeline: List[Dict], fit_mode=True):
        temp_file = None
        try:
            X_processed = np.array(X_data, dtype=float).copy()
            if fit_mode:
                self.fitted_preprocessing_params = {}
            ms_pre_steps = []
            standard_steps = []
            for step in pipeline:
                m = step['method']
                if m in ('TIC', 'Log1p', 'Sqrt', 'Background_Subtraction'):
                    ms_pre_steps.append(step)
                else:
                    standard_steps.append(step)
            for step in ms_pre_steps:
                m = step['method']
                p = step.get('params', {})
                X_before = X_processed.copy()
                try:
                    if m == 'TIC':
                        X_processed = _tic_normalization(X_processed)
                    elif m == 'Log1p':
                        X_processed = _log1p_transform(X_processed)
                    elif m == 'Sqrt':
                        X_processed = _sqrt_transform(X_processed)
                    elif m == 'Background_Subtraction':
                        X_processed = _background_subtraction(X_processed, quantile=p.get('quantile', 0.05))
                    if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                        X_processed = X_before
                except Exception:
                    X_processed = X_before
            if standard_steps:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    temp_file = tmp.name
                    pd.DataFrame(X_processed).to_csv(temp_file, index=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    spectral_data = SpectralData(temp_file)
                    for step_idx, step in enumerate(standard_steps):
                        method = step['method']
                        params = step.get('params', {})
                        try:
                            with contextlib.redirect_stdout(io.StringIO()):
                                if method == 'AsLS':
                                    spectral_data.AsLS(lam=params['lam'], p=params['p'], niter=params['niter'])
                                elif method == 'Polyfit':
                                    spectral_data.polyfit(order=params['order'], niter=params['niter'])
                                elif method == 'Pearson':
                                    spectral_data.pearson(u=params['u'], v=params['v'])
                                elif method == 'Rolling':
                                    spectral_data.rolling(window=params['window'])
                                elif method == 'Savitzky-Golay':
                                    spectral_data.SGSmooth(window=params['window'], poly=params['poly'])
                                elif method == 'SNV':
                                    spectral_data.snv()
                                elif method == 'MSC':
                                    spectral_data.msc()
                                elif method == 'Detrend':
                                    spectral_data.detrend(order=params['order'])
                                elif method == 'Area':
                                    spectral_data.area()
                                elif method == 'Vector':
                                    spectral_data.vector()
                                elif method == 'Peak Normalization':
                                    spectral_data.peaknorm(wavenumber=params['wave'])
                                elif method == 'Min-max':
                                    current = spectral_data.spc.values
                                    vmin = np.min(current, axis=1, keepdims=True)
                                    vmax = np.max(current, axis=1, keepdims=True)
                                    rng = np.where((vmax - vmin) < 1e-12, 1.0, vmax - vmin)
                                    scaled = (current - vmin) / rng
                                    scaled = scaled * (params['maxv'] - params['minv']) + params['minv']
                                    spectral_data.spc = pd.DataFrame(scaled)
                                elif method == 'Pareto':
                                    spectral_data.pareto()
                                elif method == 'Mean (spectrum)':
                                    spectral_data.mean_center(option=False)
                                elif method == 'Mean (wavelength)':
                                    spectral_data.mean_center(option=True)
                                elif method == 'Last Point':
                                    spectral_data.lastpoint()
                                elif method == 'Derivative_Subtract':
                                    spectral_data.subtract(spectra=params['subtract_idx'])
                                elif method == 'Derivative_Reset':
                                    spectral_data.reset()
                                elif method == 'SG Derivative':
                                    spectral_data.SGDeriv(window=params['window'], poly=params['poly'], order=params['order'])
                        except Exception:
                            continue
                X_processed = spectral_data.spc.values
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                return np.array(X_data, dtype=float).copy()
            return X_processed
        except Exception:
            return np.array(X_data, dtype=float).copy()
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def _create_automated_pipeline(self, trial) -> Tuple[List[Dict], str, Dict]:
        pipeline: List[Dict] = []
        if trial.suggest_categorical('use_background_sub', [True, False]):
            pipeline.append({'method': 'Background_Subtraction', 'params': {'quantile': trial.suggest_float('bg_quantile', 0.01, 0.10)}})
        if trial.suggest_categorical('use_intensity_transform', [True, False]):
            it = trial.suggest_categorical('intensity_transform', ['Log1p', 'Sqrt', 'TIC'])
            pipeline.append({'method': it, 'params': {}})
        if trial.suggest_categorical('use_baseline', [True, False]):
            bm = trial.suggest_categorical('baseline_method', ['AsLS', 'Polyfit'])
            if bm == 'AsLS':
                pipeline.append({'method': 'AsLS', 'params': {
                    'lam': trial.suggest_float('asls_lam', 1e4, 1e7, log=True),
                    'p': trial.suggest_float('asls_p', 0.001, 0.1, log=True),
                    'niter': trial.suggest_int('asls_niter', 5, 15),
                }})
            else:
                pipeline.append({'method': 'Polyfit', 'params': {
                    'order': trial.suggest_int('poly_order', 2, 4),
                    'niter': trial.suggest_int('poly_niter', 1, 10),
                }})
        if trial.suggest_categorical('use_smoothing', [True, False]):
            w = trial.suggest_int('sg_window', 5, 15, step=2)
            pipeline.append({'method': 'Savitzky-Golay', 'params': {
                'window': w,
                'poly': trial.suggest_int('sg_poly', 2, min(4, w - 1)),
            }})
        if trial.suggest_categorical('use_normalization', [True, False]):
            nm = trial.suggest_categorical('normalization_method', ['SNV', 'MSC', 'Area', 'Vector', 'Min-max', 'Pareto'])
            if nm == 'Min-max':
                pipeline.append({'method': 'Min-max', 'params': {
                    'minv': trial.suggest_float('minmax_min', 0.0, 0.1),
                    'maxv': trial.suggest_float('minmax_max', 0.9, 1.0),
                }})
            else:
                pipeline.append({'method': nm, 'params': {}})
        if trial.suggest_categorical('use_center', [True, False]):
            pipeline.append({
                'method': trial.suggest_categorical('center_method', ['Mean (spectrum)', 'Mean (wavelength)']),
                'params': {},
            })
        if trial.suggest_categorical('use_sg_derivative', [True, False]):
            w = trial.suggest_int('sgd_window', 5, 15, step=2)
            pipeline.append({'method': 'SG Derivative', 'params': {
                'window': w,
                'poly': trial.suggest_int('sgd_poly', 2, min(4, w - 1)),
                'order': trial.suggest_int('sgd_order', 1, 2),
            }})
        model_name = trial.suggest_categorical('model', ['PLS', 'RandomForest', 'XGBoost'])
        model_params = self._get_model_params(trial, model_name)
        return pipeline, model_name, model_params

    def _get_model_params(self, trial, model_name: str) -> Dict:
        if model_name == 'PLS':
            mc = min(25, self.X_train.shape[1] // 3, self.X_train.shape[0] - 1)
            return {'n_components': trial.suggest_int('pls_n_components', 1, max(1, mc))}
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('xgboost_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('xgboost_max_depth', 3, 8),
                'learning_rate': trial.suggest_float('xgboost_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('xgboost_subsample', 0.7, 1.0),
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('randomforest_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('randomforest_max_depth', 3, 12),
                'min_samples_leaf': trial.suggest_int('randomforest_min_samples_leaf', 1, 4),
            }
        return {}

    def _create_model(self, model_name: str, model_params: Dict):
        base = {}
        if model_name in ('RandomForest', 'XGBoost'):
            base['random_state'] = self.random_state
        if model_name == 'XGBoost':
            base['verbosity'] = 0
        params = {**base, **model_params}
        if model_name == 'PLS':
            mc = max(1, min(params.get('n_components', 5), self.X_train.shape[0] - 1, self.X_train.shape[1]))
            return PLSRegression(n_components=mc)
        elif model_name == 'XGBoost':
            return XGBRegressor(**params)
        elif model_name == 'RandomForest':
            return RandomForestRegressor(**params)
        raise ValueError(f"Unknown model: {model_name}")

    def _evaluate_pipeline(self, pipeline: List[Dict], model_name: str, model_params: Dict) -> float:
        try:
            X_proc = self._apply_preprocessing_pipeline(self.X_train, pipeline)
            if (np.any(np.isnan(X_proc)) or np.any(np.isinf(X_proc)) or
                    X_proc.shape[1] < 1 or np.sum(np.var(X_proc, axis=0) > 1e-10) < 1):
                return -1000
            model = self._create_model(model_name, model_params)
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X_proc, self.y_train, cv=kf, scoring='r2')
            s = float(np.mean(scores))
            return s if not np.isnan(s) else -1000
        except Exception:
            return -1000

    def _objective(self, trial) -> float:
        self.current_trial = trial.number
        pipeline, model_name, model_params = self._create_automated_pipeline(trial)
        score = self._evaluate_pipeline(pipeline, model_name, model_params)
        self.optimization_history.append({
            'trial': trial.number,
            'score': score,
            'pipeline': pipeline,
            'model_name': model_name,
            'model_params': model_params,
            'params': trial.params,
        })
        return score

    def optimize(self, progress_callback=None) -> Dict:
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(),
            )
            def _obj(trial):
                try:
                    if progress_callback:
                        pct = min(80, int(trial.number / self.n_trials * 80))
                        progress_callback(pct, f"MS Trial {trial.number+1}/{self.n_trials}")
                    return self._objective(trial)
                except Exception:
                    return -1000
            study.optimize(_obj, n_trials=self.n_trials)
            self.best_params = study.best_params
            self.best_score = study.best_value
            for info in self.optimization_history:
                if info['score'] == self.best_score:
                    self.best_preprocessing_steps = info['pipeline']
                    self.best_model = info['model_name']
                    break
            self._train_and_evaluate_final_model()
            return {
                'success': True,
                'cv_score': self.best_score,
                'all_model_results': self.all_model_results,
                'best_pipeline': self.best_preprocessing_steps,
                'best_model': self.best_model,
                'best_params': self.best_params,
                'optimization_study': study,
                'n_trials_completed': len(self.optimization_history),
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'summary': self._generate_summary(),
            }
        except Exception as exc:
            return {
                'success': False,
                'error': str(exc),
                'traceback': _tb.format_exc(),
                'cv_score': float(self.best_score),
                'all_model_results': None,
                'best_pipeline': self.best_preprocessing_steps,
                'best_params': self.best_params,
                'n_trials_completed': len(self.optimization_history),
                'summary': f"MS optimization failed: {exc}",
            }

    def _train_and_evaluate_final_model(self):
        try:
            X_tr = self._apply_preprocessing_pipeline(self.X_train, self.best_preprocessing_steps, fit_mode=True)
            X_te = self._apply_preprocessing_pipeline(self.X_test, self.best_preprocessing_steps, fit_mode=False)
            self.all_model_results = {}
            for model_name in self.models:
                try:
                    prefix = model_name.lower() + '_'
                    mp = {k[len(prefix):]: v for k, v in self.best_params.items() if k.startswith(prefix)}
                    model = self._create_model(model_name, mp)
                    model.fit(X_tr, self.y_train)
                    y_tr_pred = np.ravel(model.predict(X_tr))
                    y_te_pred = np.ravel(model.predict(X_te))
                    self.all_model_results[model_name] = {
                        'train_r2': r2_score(self.y_train, y_tr_pred),
                        'train_rmse': float(np.sqrt(mean_squared_error(self.y_train, y_tr_pred))),
                        'train_mae': float(mean_absolute_error(self.y_train, y_tr_pred)),
                        'test_r2': r2_score(self.y_test, y_te_pred),
                        'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, y_te_pred))),
                        'test_mae': float(mean_absolute_error(self.y_test, y_te_pred)),
                        'y_train_pred': y_tr_pred,
                        'y_test_pred': y_te_pred,
                        'train_residuals': self.y_train - y_tr_pred,
                        'test_residuals': self.y_test - y_te_pred,
                    }
                except Exception:
                    continue
        except Exception as exc:
            self.all_model_results = {'error': str(exc)}

    def _generate_summary(self) -> str:
        lines = [
            "MASS SPECTROMETRY PREPROCESSING OPTIMIZATION",
            "=" * 70,
            f"Best CV R²    : {self.best_score:.4f}",
            f"Best Model    : {self.best_model}",
            f"CV Folds      : {self.cv_folds}  |  Trials: {self.n_trials}",
            f"Train / Test  : {self.X_train.shape[0]} / {self.X_test.shape[0]}",
            f"m/z channels  : {self.X_train.shape[1]}",
            "=" * 70, "", "BEST PIPELINE:", "-" * 40,
        ]
        for i, s in enumerate(self.best_preprocessing_steps, 1):
            lines.append(f"  {i}. {s['method']}")
            for k, v in s.get('params', {}).items():
                lines.append(f"     {k}: {v:.4f}" if isinstance(v, float) else f"     {k}: {v}")
        lines += ["", "MODEL RESULTS:", "-" * 40]
        for mn, r in self.all_model_results.items():
            if isinstance(r, dict) and 'test_r2' in r:
                lines.append(f"  {mn}: Train R²={r['train_r2']:.4f}  Test R²={r['test_r2']:.4f}")
        return "\n".join(lines) + "\n"

    def apply_best_preprocessing(self, X_new=None, fit_mode=False):
        if not self.best_preprocessing_steps:
            raise ValueError("No optimization yet. Call optimize() first.")
        if X_new is None:
            raise ValueError("Must provide X_new.")
        return self._apply_preprocessing_pipeline(np.array(X_new, dtype=float), self.best_preprocessing_steps, fit_mode)

def optimize_mass_spec_preprocessing(X, y, n_trials=50, cv_folds=5, test_size=0.2, progress_callback=None) -> Dict:
    opt = MassSpectralPreprocessingOptimizer(X, y, cv_folds=cv_folds, n_trials=n_trials, test_size=test_size)
    return opt.optimize(progress_callback=progress_callback)