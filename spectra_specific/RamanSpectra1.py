import numpy as np
import pandas as pd
import optuna
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import logging
import contextlib
import io
from typing import Dict, List, Tuple, Optional, Any
import copy
from preprocess import SpectralData
import tempfile
import os

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def _suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

class RamanPreprocessingOptimizer:
    def __init__(self, X, y, cv_folds=5, n_trials=50, random_state=42):
        X = np.array(X)
        y = np.array(y).ravel()

        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.random_state = random_state

        self.X_train_full, self.X_test, self.y_train_full, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        nonzero_mask = self.y_train_full != 0
        self.X_train = self.X_train_full[nonzero_mask]
        self.y_train = self.y_train_full[nonzero_mask]

        self._validate_input_data()

        self.models = {'PLS': None}

        self.best_params = None
        self.best_score = -np.inf
        self.best_preprocessing_steps = []
        self.optimization_history = []
        self.all_model_results = {}
        self.current_trial = 0
        self.fitted_preprocessing_params = {}

    def _validate_input_data(self):
        if self.X_train.shape[0] != len(self.y_train):
            raise ValueError(
                f"X_train and y_train dimension mismatch: "
                f"X_train has {self.X_train.shape[0]} samples, y_train has {len(self.y_train)} samples"
            )
        if self.X_test.shape[0] != len(self.y_test):
            raise ValueError(
                f"X_test and y_test dimension mismatch: "
                f"X_test has {self.X_test.shape[0]} samples, y_test has {len(self.y_test)} samples"
            )
        for name, arr in [("X_train", self.X_train), ("y_train", self.y_train),
                          ("X_test", self.X_test), ("y_test", self.y_test)]:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                raise ValueError(f"{name} contains NaN or infinite values")

        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X_train has {self.X_train.shape[1]} features, "
                f"X_test has {self.X_test.shape[1]} features"
            )
        if self.X_train.shape[1] < 2:
            raise ValueError("X must have at least 2 features")
        if self.X_train.shape[0] < 10:
            raise ValueError("Training dataset too small. Need at least 10 samples.")

    def _safe_n_components(self, n_samples: int, n_features: int) -> int:
        return max(1, min(10, n_samples - 1, n_features))

    def _create_automated_pipeline(self, trial) -> List[Dict]:
        pipeline = []

        baseline_method = trial.suggest_categorical('baseline_method', ['AsLS', 'Polyfit', 'Pearson'])
        if baseline_method == 'AsLS':
            lam = trial.suggest_float('asls_lam', 1e4, 1e7, log=True)
            p = trial.suggest_float('asls_p', 0.001, 0.09, log=True)
            niter = trial.suggest_int('asls_niter', 5, 15)
            pipeline.append({'method': 'AsLS', 'params': {'lam': lam, 'p': p, 'niter': niter}})
        elif baseline_method == 'Polyfit':
            order = trial.suggest_int('polyfit_order', 1, 5)
            niter = trial.suggest_int('polyfit_niter', 3, 10)
            pipeline.append({'method': 'Polyfit', 'params': {'order': order, 'niter': niter}})
        elif baseline_method == 'Pearson':
            u = trial.suggest_int('pearson_u', 5, 20)
            v = trial.suggest_int('pearson_v', 5, 20)
            pipeline.append({'method': 'Pearson', 'params': {'u': u, 'v': v}})

        smooth_method = trial.suggest_categorical('smooth_method', ['Rolling', 'Savitzky-Golay'])
        if smooth_method == 'Rolling':
            window = trial.suggest_int('rolling_window', 3, 15, step=2)
            pipeline.append({'method': 'Rolling', 'params': {'window': window}})
        elif smooth_method == 'Savitzky-Golay':
            window = trial.suggest_int('sg_window', 5, 15, step=2)
            poly = trial.suggest_int('sg_poly', 2, min(4, window - 1))
            pipeline.append({'method': 'Savitzky-Golay', 'params': {'window': window, 'poly': poly}})

        if trial.suggest_categorical('use_normalization', [True, False]):
            norm_method = trial.suggest_categorical(
                'normalization_method',
                ['SNV', 'Detrend', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto']
            )

            if norm_method == 'Detrend':
                order = trial.suggest_int('detrend_order', 1, 3)
                pipeline.append({'method': 'Detrend', 'params': {'order': order}})
            elif norm_method == 'Peak Normalization':
                wave = trial.suggest_float('peak_wave', 0, float(self.X_train.shape[1] - 1))
                pipeline.append({'method': 'Peak Normalization', 'params': {'wave': wave}})
            elif norm_method == 'Min-max':
                minv = trial.suggest_float('minmax_min', -1.0, 0.0)
                maxv = trial.suggest_float('minmax_max', 1.0, 2.0)
                pipeline.append({'method': 'Min-max', 'params': {'minv': minv, 'maxv': maxv}})
            else:
                pipeline.append({'method': norm_method, 'params': {}})

        if trial.suggest_categorical('use_center', [True, False]):
            center_method = trial.suggest_categorical(
                'center_method', ['Mean (spectrum)', 'Mean (wavelength)', 'Last Point']
            )
            pipeline.append({'method': center_method, 'params': {}})

        if trial.suggest_categorical('use_sg_derivative', [True, False]):
            window = trial.suggest_int('sgd_window', 5, 15, step=2)
            poly = trial.suggest_int('sgd_poly', 2, min(4, window - 1))
            order = trial.suggest_int('sgd_order', 1, 2)
            pipeline.append({'method': 'SG Derivative', 'params': {'window': window, 'poly': poly, 'order': order}})

        return pipeline

    def _apply_preprocessing_pipeline(self, X_data: np.ndarray, pipeline: List[Dict], fit_mode: bool = True) -> np.ndarray:
        temp_file = None
        try:
            X_processed = X_data.copy()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                temp_file = tmp.name
                pd.DataFrame(X_processed).to_csv(temp_file, index=False)

            spectral_data = SpectralData(temp_file)

            if fit_mode:
                self.fitted_preprocessing_params = {}

            for step_idx, step in enumerate(pipeline):
                method = step['method']
                params = step.get('params', {})
                step_key = f"step_{step_idx}_{method}"

                try:
                    if method == 'AsLS':
                        with _suppress_stdout():
                            spectral_data.AsLS(lam=params['lam'], p=params['p'], niter=int(params['niter']))

                    elif method == 'Polyfit':
                        with _suppress_stdout():
                            spectral_data.polyfit(order=int(params['order']), niter=int(params['niter']))

                    elif method == 'Pearson':
                        with _suppress_stdout():
                            spectral_data.pearson(u=int(params['u']), v=int(params['v']))

                    elif method == 'Rolling':
                        spectral_data.rolling(window=int(params['window']))

                    elif method == 'Savitzky-Golay':
                        spectral_data.SGSmooth(window=int(params['window']), poly=int(params['poly']))

                    elif method == 'SNV':
                        current_data = spectral_data.spc.values
                        means = np.mean(current_data, axis=1, keepdims=True)
                        stds  = np.std(current_data,  axis=1, keepdims=True)
                        stds  = np.where(stds == 0, 1.0, stds)
                        spectral_data.spc = pd.DataFrame((current_data - means) / stds)

                    elif method == 'MSC':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            reference = np.mean(current_data, axis=0)
                            self.fitted_preprocessing_params[step_key] = {'reference': reference}
                            spectral_data.msc()
                        else:
                            reference = self.fitted_preprocessing_params.get(step_key, {}).get(
                                'reference', np.mean(current_data, axis=0)
                            )
                            corrected = np.zeros_like(current_data)
                            for i in range(current_data.shape[0]):
                                coeff = np.polyfit(reference, current_data[i], 1)
                                corrected[i] = (current_data[i] - coeff[1]) / (coeff[0] if coeff[0] != 0 else 1.0)
                            spectral_data.spc = pd.DataFrame(corrected)

                    elif method == 'Detrend':
                        spectral_data.detrend(order=int(params['order']))

                    elif method == 'Area':
                        spectral_data.area()

                    elif method == 'Peak Normalization':
                        spectral_data.peaknorm(wavenumber=params['wave'])

                    elif method == 'Vector':
                        spectral_data.vector()

                    elif method == 'Min-max':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            self.fitted_preprocessing_params[step_key] = {
                                'target_min': params['minv'], 'target_max': params['maxv']
                            }
                        t_min = self.fitted_preprocessing_params.get(step_key, {}).get('target_min', params.get('minv', 0))
                        t_max = self.fitted_preprocessing_params.get(step_key, {}).get('target_max', params.get('maxv', 1))
                        data_min = np.min(current_data, axis=1, keepdims=True)
                        data_max = np.max(current_data, axis=1, keepdims=True)
                        denom = np.where((data_max - data_min) == 0, 1.0, data_max - data_min)
                        scaled = (current_data - data_min) / denom
                        spectral_data.spc = pd.DataFrame(scaled * (t_max - t_min) + t_min)

                    elif method == 'Pareto':
                        current_data = spectral_data.spc.values
                        means     = np.mean(current_data, axis=1, keepdims=True)
                        stds      = np.std(current_data,  axis=1, keepdims=True)
                        sqrt_stds = np.sqrt(np.where(stds == 0, 1.0, stds))
                        spectral_data.spc = pd.DataFrame((current_data - means) / sqrt_stds)

                    elif method == 'Mean (spectrum)':
                        current_data   = spectral_data.spc.values
                        spectrum_means = np.mean(current_data, axis=1, keepdims=True)
                        spectral_data.spc = pd.DataFrame(current_data - spectrum_means)

                    elif method == 'Mean (wavelength)':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            wavelength_means = np.mean(current_data, axis=0, keepdims=True)
                            self.fitted_preprocessing_params[step_key] = {'wavelength_means': wavelength_means}
                        else:
                            wavelength_means = self.fitted_preprocessing_params.get(step_key, {}).get(
                                'wavelength_means', np.mean(current_data, axis=0, keepdims=True)
                            )
                        spectral_data.spc = pd.DataFrame(current_data - wavelength_means)

                    elif method == 'Last Point':
                        spectral_data.lastpoint()

                    elif method == 'Derivative_Subtract':
                        spectral_data.subtract(spectra=int(params['subtract_idx']))

                    elif method == 'Derivative_Reset':
                        spectral_data.reset()

                    elif method == 'SG Derivative':
                        spectral_data.SGDeriv(
                            window=int(params['window']),
                            poly=int(params['poly']),
                            order=int(params['order'])
                        )

                except Exception as e:
                    print(f"Warning: Error applying {method}: {e}")
                    continue

            return spectral_data.spc.values

        except Exception as e:
            print(f"Error in preprocessing pipeline: {e}")
            return X_data.copy()

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

    def _create_model(self, model_name: str, n_samples: int = None, n_features: int = None):
        if model_name == 'PLS':
            if n_samples is not None and n_features is not None:
                n_comp = self._safe_n_components(n_samples, n_features)
            else:
                n_comp = self._safe_n_components(self.X_train.shape[0], self.X_train.shape[1])
            return PLSRegression(n_components=n_comp)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _evaluate_pipeline(self, pipeline: List[Dict]) -> Tuple[float, Dict]:
        try:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            model_scores = {}

            for model_name in self.models.keys():
                fold_scores = []

                for train_idx, val_idx in kf.split(self.X_train):
                    X_train_fold = self.X_train[train_idx]
                    X_val_fold = self.X_train[val_idx]
                    y_train_fold = self.y_train[train_idx]
                    y_val_fold = self.y_train[val_idx]

                    X_train_fold_processed = self._apply_preprocessing_pipeline(
                        X_train_fold, pipeline, fit_mode=True
                    )

                    if (np.any(np.isnan(X_train_fold_processed))
                            or np.any(np.isinf(X_train_fold_processed))
                            or X_train_fold_processed.shape[1] < 1):
                        fold_scores.append(-1000.0)
                        continue

                    feature_vars = np.var(X_train_fold_processed, axis=0)
                    if np.sum(feature_vars > 1e-10) < 1:
                        fold_scores.append(-1000.0)
                        continue

                    X_val_fold_processed = self._apply_preprocessing_pipeline(
                        X_val_fold, pipeline, fit_mode=False
                    )

                    if np.any(np.isnan(X_val_fold_processed)) or np.any(np.isinf(X_val_fold_processed)):
                        fold_scores.append(-1000.0)
                        continue

                    try:
                        model_fold = self._create_model(
                            model_name,
                            n_samples=X_train_fold_processed.shape[0],
                            n_features=X_train_fold_processed.shape[1]
                        )
                        model_fold.fit(X_train_fold_processed, y_train_fold)
                        y_val_pred = model_fold.predict(X_val_fold_processed)
                        fold_score = r2_score(y_val_fold, y_val_pred)
                        fold_scores.append(fold_score if not np.isnan(fold_score) else -1000.0)
                    except Exception as e:
                        print(f"Warning: model fitting error in fold: {e}")
                        fold_scores.append(-1000.0)

                mean_score = float(np.mean(fold_scores))
                model_scores[model_name] = mean_score if not np.isnan(mean_score) else -1000.0

            best_score = max(model_scores.values()) if model_scores else -1000.0
            return best_score, model_scores

        except Exception as e:
            print(f"Warning: Error in pipeline evaluation: {e}")
            return -1000.0, {}

    def _objective(self, trial):
        self.current_trial = trial.number
        pipeline = self._create_automated_pipeline(trial)
        score, model_scores = self._evaluate_pipeline(pipeline)

        self.optimization_history.append({
            'trial': trial.number,
            'score': score,
            'model_scores': model_scores,
            'pipeline': pipeline,
            'params': trial.params
        })

        return score

    def optimize(self, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(0, "Initializing optimization...")

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )

            def progress_objective(trial):
                try:
                    if progress_callback:
                        progress = min(80, int((trial.number / self.n_trials) * 80))
                        progress_callback(progress, f"Trial {trial.number + 1}/{self.n_trials}")
                    return self._objective(trial)
                except Exception as e:
                    print(f"Error in trial {trial.number}: {e}")
                    return -1000.0

            study.optimize(progress_objective, n_trials=self.n_trials)

            if progress_callback:
                progress_callback(85, "Training final model on full training set...")

            self.best_params = study.best_params
            self.best_score = study.best_value

            for trial_info in self.optimization_history:
                if trial_info['score'] == self.best_score:
                    self.best_preprocessing_steps = trial_info['pipeline']
                    break

            if progress_callback:
                progress_callback(90, "Evaluating on held-out test set...")

            self._train_and_evaluate_all_models()

            if progress_callback:
                progress_callback(100, "Optimization complete!")

            return {
                'success': True,
                'cv_score': self.best_score,
                'all_model_results': self.all_model_results,
                'best_pipeline': self.best_preprocessing_steps,
                'best_params': self.best_params,
                'optimization_study': study,
                'n_trials_completed': len(self.optimization_history),
                'train_size_original': self.X_train_full.shape[0],
                'train_size_nonzero': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'summary': self._generate_summary()
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'cv_score': self.best_score,
                'all_model_results': None,
                'best_pipeline': self.best_preprocessing_steps,
                'best_params': self.best_params,
                'n_trials_completed': len(self.optimization_history),
                'summary': f"Optimization failed: {str(e)}"
            }

    def _train_and_evaluate_all_models(self):
        try:
            X_train_processed = self._apply_preprocessing_pipeline(
                self.X_train, self.best_preprocessing_steps, fit_mode=True
            )
            X_test_processed = self._apply_preprocessing_pipeline(
                self.X_test, self.best_preprocessing_steps, fit_mode=False
            )

            self.all_model_results = {}

            for model_name in self.models.keys():
                model = self._create_model(
                    model_name,
                    n_samples=X_train_processed.shape[0],
                    n_features=X_train_processed.shape[1]
                )
                model.fit(X_train_processed, self.y_train)

                y_train_pred = model.predict(X_train_processed)
                y_test_pred = model.predict(X_test_processed)

                self.all_model_results[model_name] = {
                    'train_r2': r2_score(self.y_train, y_train_pred),
                    'train_rmse': float(np.sqrt(mean_squared_error(self.y_train, y_train_pred))),
                    'train_mae': float(mean_absolute_error(self.y_train, y_train_pred)),
                    'test_r2': r2_score(self.y_test, y_test_pred),
                    'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, y_test_pred))),
                    'test_mae': float(mean_absolute_error(self.y_test, y_test_pred)),
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'train_residuals': self.y_train - y_train_pred,
                    'test_residuals': self.y_test - y_test_pred
                }

        except Exception as e:
            print(f"Error in final model training: {e}")
            self.all_model_results = {'error': {'error': str(e)}}

    def _generate_summary(self) -> str:
        summary = "AUTOMATED RAMAN PREPROCESSING OPTIMIZATION RESULTS\n"
        summary += "=" * 70 + "\n"
        summary += f"Best CV R² Score (Training): {self.best_score:.4f}\n"
        summary += f"CV Folds: {self.cv_folds} | Trials: {self.n_trials}\n"
        summary += (
            f"Original Train Size: {self.X_train_full.shape[0]} | "
            f"Non-zero Train Size: {self.X_train.shape[0]} | "
            f"Test Size: {self.X_test.shape[0]}\n"
        )
        summary += f"Features: {self.X_train.shape[1]}\n"
        summary += "=" * 70 + "\n\n"

        summary += "PREPROCESSING PIPELINE:\n"
        summary += "-" * 30 + "\n"
        if self.best_preprocessing_steps:
            for i, step in enumerate(self.best_preprocessing_steps, 1):
                summary += f"{i}. {step['method'].upper().replace('_', ' ')}\n"
                if step.get('params'):
                    for param, value in step['params'].items():
                        if isinstance(value, float):
                            summary += f"   • {param}: {value:.4f}\n"
                        else:
                            summary += f"   • {param}: {value}\n"
                summary += "\n"
        else:
            summary += "No preprocessing steps applied.\n\n"

        if self.all_model_results:
            summary += "ALL MODEL RESULTS WITH OPTIMISED PREPROCESSING:\n"
            summary += "-" * 50 + "\n"
            for model_name, results in self.all_model_results.items():
                if isinstance(results, dict) and 'test_r2' in results:
                    summary += f"{model_name}:\n"
                    summary += f"  Train R²: {results['train_r2']:.4f} | Test R²: {results['test_r2']:.4f}\n"
                    summary += f"  Train RMSE: {results['train_rmse']:.4f} | Test RMSE: {results['test_rmse']:.4f}\n"
                    summary += f"  Train MAE: {results['train_mae']:.4f} | Test MAE: {results['test_mae']:.4f}\n\n"

        return summary

    def apply_best_preprocessing(self, X_new: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        if not self.best_preprocessing_steps:
            raise ValueError("No optimization performed yet. Call optimize() first.")
        if X_new is None:
            raise ValueError("Must provide X_new data to preprocess.")
        return self._apply_preprocessing_pipeline(X_new, self.best_preprocessing_steps, fit_mode=fit_mode)

def optimize_raman_preprocessing(X, y, n_trials=50, cv_folds=5, progress_callback=None):
    optimizer = RamanPreprocessingOptimizer(X, y, cv_folds=cv_folds, n_trials=n_trials)
    return optimizer.optimize(progress_callback)