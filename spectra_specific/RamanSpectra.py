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
from typing import Dict, List, Tuple, Optional, Any
import copy
from preprocess import SpectralData
import tempfile
import os

warnings.filterwarnings('ignore')

class RamanPreprocessingOptimizer:

    def __init__(self, X_train, X_test, y_train, y_test, cv_folds=5, n_trials=50, random_state=42):
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.cv_folds = cv_folds
        self.n_trials = n_trials
        self.random_state = random_state
        
        self._validate_input_data()
        
        # Fixed: Define models properly
        self.models = {
            'PLS': PLSRegression(),
            'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        }
        
        self.best_params = None
        self.best_score = -np.inf 
        self.best_preprocessing_steps = []
        self.best_scaler = None
        self.optimization_history = []
        self.test_results = None
        self.all_model_results = {}
        
        self.current_trial = 0
        
        self.fitted_preprocessing_params = {}
        
    def _validate_input_data(self):
        if self.X_train.shape[0] != len(self.y_train):
            raise ValueError(f"X_train and y_train dimension mismatch: X_train has {self.X_train.shape[0]} samples, y_train has {len(self.y_train)} samples")
        
        if self.X_test.shape[0] != len(self.y_test):
            raise ValueError(f"X_test and y_test dimension mismatch: X_test has {self.X_test.shape[0]} samples, y_test has {len(self.y_test)} samples")
        
        if np.any(np.isnan(self.X_train)) or np.any(np.isinf(self.X_train)):
            raise ValueError("X_train contains NaN or infinite values")
        
        if np.any(np.isnan(self.y_train)) or np.any(np.isinf(self.y_train)):
            raise ValueError("y_train contains NaN or infinite values")
        
        if np.any(np.isnan(self.X_test)) or np.any(np.isinf(self.X_test)):
            raise ValueError("X_test contains NaN or infinite values")
        
        if np.any(np.isnan(self.y_test)) or np.any(np.isinf(self.y_test)):
            raise ValueError("y_test contains NaN or infinite values")
        
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError(f"Feature dimension mismatch: X_train has {self.X_train.shape[1]} features, X_test has {self.X_test.shape[1]} features")
        
        if self.X_train.shape[1] < 2:
            raise ValueError("X must have at least 2 features")
        
        if self.X_train.shape[0] < 10:
            raise ValueError("Training dataset too small. Need at least 10 samples.")

    def _create_automated_pipeline(self, trial) -> List[Dict]:
        pipeline = []
        steps_added = 0
        max_steps = 6  # Fixed: Increased max steps since you have many operations
        
        # Fixed: Made baseline correction optional
        if trial.suggest_categorical('use_baseline', [True, False]):
            baseline_method = trial.suggest_categorical('baseline_method', ['AsLS', 'Polyfit', 'Pearson'])
            
            if baseline_method == 'AsLS':
                lam = trial.suggest_float('asls_lam', 1e4, 1e7, log=True)
                p = trial.suggest_float('asls_p', 0.01, 0.09, log=True)
                niter = trial.suggest_int('asls_niter', 5, 15)
                pipeline.append({
                    'method': 'AsLS',
                    'params': {'lam': lam, 'p': p, 'niter': niter}
                })
            elif baseline_method == 'Polyfit':
                order = trial.suggest_int('poly_order', 2, 6)
                niter = trial.suggest_int('poly_niter', 1, 15)
                pipeline.append({
                    'method': 'Polyfit',
                    'params': {'order': order, 'niter': niter}
                })
            elif baseline_method == 'Pearson':
                u = trial.suggest_int('pearson_u', 2, 6)
                v = trial.suggest_int('pearson_v', 1, 5)
                pipeline.append({
                    'method': 'Pearson',
                    'params': {'u': u, 'v': v}
                })
            steps_added += 1

        
        if trial.suggest_categorical('use_smoothing', [True, False]):
            smooth_method = trial.suggest_categorical('smooth_method', ['Rolling', 'Savitzky-Golay'])
            
            if smooth_method == 'Rolling':
                window = trial.suggest_int('rolling_window', 3, 15, step=2)
                pipeline.append({
                    'method': 'Rolling',
                    'params': {'window': window}
                })
            elif smooth_method == 'Savitzky-Golay':
                window = trial.suggest_int('sg_window', 5, 15, step=2)
                poly = trial.suggest_int('sg_poly', 2, min(4, window-1))
                pipeline.append({
                    'method': 'Savitzky-Golay',
                    'params': {'window': window, 'poly': poly}
                })
            steps_added += 1
        
        if trial.suggest_categorical('use_normalization', [True, False]):
            norm_method = trial.suggest_categorical('normalization_method', 
                ['SNV', 'Detrend', 'MSC', 'Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto'])
            
            if norm_method == 'Detrend':
                order = trial.suggest_int('detrend_order', 1, 3)
                pipeline.append({
                    'method': 'Detrend',
                    'params': {'order': order}
                })
            elif norm_method == 'Peak Normalization':
                # Fixed: Ensure wave parameter is within valid range
                wave = trial.suggest_int('peak_wave', 0, min(100, self.X_train.shape[1]-1))
                pipeline.append({
                    'method': 'Peak Normalization',
                    'params': {'wave': wave}
                })
            elif norm_method == 'Min-max':
                minv = trial.suggest_float('minmax_min', -1.0, 0.0)
                maxv = trial.suggest_float('minmax_max', 1.0, 2.0)
                pipeline.append({
                    'method': 'Min-max',
                    'params': {'minv': minv, 'maxv': maxv}
                })
            else:
                pipeline.append({
                    'method': norm_method,
                    'params': {}
                })
            steps_added += 1

        if trial.suggest_categorical('use_center', [True, False]):
            center_method = trial.suggest_categorical('center_method', 
                ['Mean (spectrum)', 'Mean (wavelength)', 'Last Point'])
            pipeline.append({
                'method': center_method,
                'params': {}
            })
            steps_added += 1

        
        if trial.suggest_categorical('use_derivative', [True, False]):
            deriv_option = trial.suggest_categorical('derivative_option', ['Subtract', 'Reset'])
            
            if deriv_option == 'Subtract':
                # Fixed: Ensure subtract_idx is valid
                subtract_idx = trial.suggest_int('subtract_idx', 0, min(4, self.X_train.shape[0]-1))
                pipeline.append({
                    'method': 'Derivative_Subtract',
                    'params': {'subtract_idx': subtract_idx}
                })
            else:
                pipeline.append({
                    'method': 'Derivative_Reset',
                    'params': {}
                })
            
            steps_added += 1

        if trial.suggest_categorical('use_sg_derivative', [True, False]):
            window = trial.suggest_int('sgd_window', 5, 15, step=2)
            poly = trial.suggest_int('sgd_poly', 2, min(4, window-1))
            order = trial.suggest_int('sgd_order', 1, 2)
            pipeline.append({
                'method': 'SG Derivative',
                'params': {'window': window, 'poly': poly, 'order': order}
            })
            steps_added += 1
        
        return pipeline
    
    def _apply_preprocessing_pipeline(self, X_data, pipeline: List[Dict], fit_mode=True):
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
                        # Fixed: Store parameters correctly for fit/transform
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            means = np.mean(current_data, axis=1, keepdims=True)
                            stds = np.std(current_data, axis=1, keepdims=True, ddof=0)  # Fixed: Added ddof=0
                            stds = np.where(stds == 0, 1, stds)  # Fixed: Prevent division by zero
                            self.fitted_preprocessing_params[step_key] = {'means': means, 'stds': stds}
                            spectral_data.spc = pd.DataFrame((current_data - means) / stds)
                        else:
                            if step_key in self.fitted_preprocessing_params:
                                params_stored = self.fitted_preprocessing_params[step_key]
                                spectral_data.spc = pd.DataFrame((current_data - params_stored['means']) / params_stored['stds'])
                            else:
                                spectral_data.snv()  # Fallback to built-in method

                    elif method == 'MSC':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            reference = np.mean(current_data, axis=0)
                            self.fitted_preprocessing_params[step_key] = {'reference': reference}
                            spectral_data.msc()
                        else:
                            if step_key in self.fitted_preprocessing_params:
                                reference = self.fitted_preprocessing_params[step_key]['reference']
                                corrected = np.zeros_like(current_data)
                                for i in range(current_data.shape[0]):
                                    try:
                                        coeff = np.polyfit(reference, current_data[i], 1)
                                        corrected[i] = (current_data[i] - coeff[1]) / coeff[0]
                                    except:
                                        corrected[i] = current_data[i]  # Fallback
                                spectral_data.spc = pd.DataFrame(corrected)
                            else:
                                spectral_data.msc()  # Fallback to built-in method

                    elif method == 'Detrend':
                        spectral_data.detrend(order=params['order'])

                    elif method == 'Area':
                        spectral_data.area()

                    elif method == 'Peak Normalization':
                        spectral_data.peaknorm(wavenumber=params['wave'])

                    elif method == 'Vector':
                        spectral_data.vector()

                    elif method == 'Min-max':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            data_min = np.min(current_data, axis=1, keepdims=True)
                            data_max = np.max(current_data, axis=1, keepdims=True)
                            # Fixed: Prevent division by zero
                            data_range = data_max - data_min
                            data_range = np.where(data_range == 0, 1, data_range)
                            self.fitted_preprocessing_params[step_key] = {
                                'data_min': data_min, 'data_max': data_max, 'data_range': data_range,
                                'target_min': params['minv'], 'target_max': params['maxv']
                            }
                            scaled = (current_data - data_min) / data_range
                            spectral_data.spc = pd.DataFrame(scaled * (params['maxv'] - params['minv']) + params['minv'])
                        else:
                            if step_key in self.fitted_preprocessing_params:
                                stored = self.fitted_preprocessing_params[step_key]
                                scaled = (current_data - stored['data_min']) / stored['data_range']
                                spectral_data.spc = pd.DataFrame(scaled * (stored['target_max'] - stored['target_min']) + stored['target_min'])
                            else:
                                spectral_data.minmax(min_val=params['minv'], max_val=params['maxv'])

                    elif method == 'Pareto':
                        current_data = spectral_data.spc.values
                        if fit_mode:
                            means = np.mean(current_data, axis=1, keepdims=True)
                            stds = np.std(current_data, axis=1, keepdims=True, ddof=0)
                            sqrt_stds = np.sqrt(np.where(stds == 0, 1, stds))  # Fixed: Prevent sqrt of zero
                            self.fitted_preprocessing_params[step_key] = {'means': means, 'sqrt_stds': sqrt_stds}
                            spectral_data.spc = pd.DataFrame((current_data - means) / sqrt_stds)
                        else:
                            if step_key in self.fitted_preprocessing_params:
                                stored = self.fitted_preprocessing_params[step_key]
                                spectral_data.spc = pd.DataFrame((current_data - stored['means']) / stored['sqrt_stds'])
                            else:
                                spectral_data.pareto()

                    elif method == 'Mean (spectrum)':
                        # Fixed: This should use mean_center with option=False
                        spectral_data.mean_center(option=False)

                    elif method == 'Mean (wavelength)':
                        # Fixed: This should use mean_center with option=True
                        spectral_data.mean_center(option=True)

                    elif method == 'Last Point':
                        spectral_data.lastpoint()

                    elif method == 'Derivative_Subtract':
                        spectral_data.subtract(spectra=params['subtract_idx'])

                    elif method == 'Derivative_Reset':
                        spectral_data.reset()

                    elif method == 'SG Derivative':
                        spectral_data.SGDeriv(window=params['window'], poly=params['poly'], order=params['order'])

                except Exception as e:
                    print(f"Warning: Error applying {method}: {e}")
                    continue

            X_processed = spectral_data.spc.values
            return X_processed
            
        except Exception as e:
            print(f"Error in preprocessing pipeline: {e}")
            return X_data.copy()
            
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass 
    
    def _create_model(self, model_name: str):
        if model_name == 'PLS':
            return PLSRegression()
        elif model_name == 'XGBoost':  # Fixed: Added elif
            return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _evaluate_pipeline(self, pipeline: List[Dict]) -> Tuple[float, Dict]:
        try:
            model_scores = {}
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # Fixed: Only evaluate models that are defined
            for model_name in ['PLS', 'XGBoost']:
                model = self._create_model(model_name)
                fold_scores = []

                for train_idx, val_idx in kf.split(self.X_train):
                    X_train_fold = self.X_train[train_idx]
                    X_val_fold = self.X_train[val_idx]
                    y_train_fold = self.y_train[train_idx]
                    y_val_fold = self.y_train[val_idx]
                    
                    X_train_fold_processed = self._apply_preprocessing_pipeline(X_train_fold, pipeline, fit_mode=True)
                    
                    if np.any(np.isnan(X_train_fold_processed)) or np.any(np.isinf(X_train_fold_processed)):
                        fold_scores.append(-1000)
                        continue
                    
                    if X_train_fold_processed.shape[1] < 1:
                        fold_scores.append(-1000)
                        continue
                    
                    feature_vars = np.var(X_train_fold_processed, axis=0)
                    if np.sum(feature_vars > 1e-10) < 1:
                        fold_scores.append(-1000)
                        continue
                    
                    X_val_fold_processed = self._apply_preprocessing_pipeline(X_val_fold, pipeline, fit_mode=False)
                    
                    if np.any(np.isnan(X_val_fold_processed)) or np.any(np.isinf(X_val_fold_processed)):
                        fold_scores.append(-1000)
                        continue
                    
                    model_fold = self._create_model(model_name)
                    
                    # Fixed: Handle PLS component count properly
                    if model_name == 'PLS':
                        max_components = min(X_train_fold_processed.shape[0]-1, X_train_fold_processed.shape[1])
                        model_fold.n_components = min(10, max_components)
                    
                    model_fold.fit(X_train_fold_processed, y_train_fold)
                    y_val_pred = model_fold.predict(X_val_fold_processed)
                    fold_score = r2_score(y_val_fold, y_val_pred)
                    
                    fold_scores.append(fold_score if not np.isnan(fold_score) else -1000)
                
                mean_score = np.mean(fold_scores)
                model_scores[model_name] = mean_score if not np.isnan(mean_score) else -1000
            
            best_score = max(model_scores.values())
            return best_score, model_scores
            
        except Exception as e:
            print(f"Warning: Error in pipeline evaluation: {e}")
            return -1000, {}
    
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
                    return -1000
            
            study.optimize(progress_objective, n_trials=self.n_trials)
            
            if progress_callback:
                progress_callback(85, "Training all models on full training set...")
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            for trial_info in self.optimization_history:
                if trial_info['score'] == self.best_score:
                    self.best_preprocessing_steps = trial_info['pipeline']
                    break
            
            if progress_callback:
                progress_callback(90, "Evaluating all models on test set...")
            
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
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'summary': self._generate_summary()
            }
            
        except Exception as e:
            print(f"Error in optimization: {e}")
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
            X_train_processed = self._apply_preprocessing_pipeline(self.X_train, self.best_preprocessing_steps, fit_mode=True)
            
            X_test_processed = self._apply_preprocessing_pipeline(self.X_test, self.best_preprocessing_steps, fit_mode=False)
            
            self.all_model_results = {}
            
            for model_name in self.models.keys():
                model = self._create_model(model_name)
                
                # Fixed: Handle PLS components properly
                if model_name == 'PLS':
                    max_components = min(X_train_processed.shape[0]-1, X_train_processed.shape[1])
                    model.n_components = min(10, max_components)
                
                model.fit(X_train_processed, self.y_train)
                
                y_train_pred = model.predict(X_train_processed)
                y_test_pred = model.predict(X_test_processed)
            
                self.all_model_results[model_name] = {
                    'train_r2': r2_score(self.y_train, y_train_pred),
                    'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    'train_mae': mean_absolute_error(self.y_train, y_train_pred),
                    'test_r2': r2_score(self.y_test, y_test_pred),
                    'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                    'test_mae': mean_absolute_error(self.y_test, y_test_pred),
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'train_residuals': self.y_train - y_train_pred,
                    'test_residuals': self.y_test - y_test_pred
                }
            
        except Exception as e:
            print(f"Error in final model training: {e}")
            self.all_model_results = {'error': str(e)}
        
    def _generate_summary(self) -> str:
        if not self.best_preprocessing_steps:
            return "No optimization performed yet."
        
        summary = f"AUTOMATED RAMAN PREPROCESSING OPTIMIZATION RESULTS\n"
        summary += "=" * 70 + "\n"
        summary += f"Best CV R² Score (Training): {self.best_score:.4f}\n"
        summary += f"CV Folds: {self.cv_folds} | Trials: {self.n_trials}\n"
        summary += f"Train Size: {self.X_train.shape[0]} | Test Size: {self.X_test.shape[0]}\n"
        summary += f"Features: {self.X_train.shape[1]}\n"
        summary += "=" * 70 + "\n\n"
        
        summary += "PREPROCESSING PIPELINE:\n"
        summary += "-" * 30 + "\n"
        if self.best_preprocessing_steps:
            for i, step in enumerate(self.best_preprocessing_steps, 1):
                summary += f"{i}. {step['method'].upper().replace('_', ' ')}\n"
                if step['params']:
                    for param, value in step['params'].items():
                        if isinstance(value, float):
                            summary += f"   • {param}: {value:.4f}\n"
                        else:
                            summary += f"   • {param}: {value}\n"
                summary += "\n"
        else:
            summary += "No preprocessing steps applied.\n\n"
        
        if self.all_model_results:
            summary += "ALL MODEL RESULTS WITH OPTIMIZED PREPROCESSING:\n"
            summary += "-" * 50 + "\n"
            for model_name, results in self.all_model_results.items():
                if isinstance(results, dict) and 'test_r2' in results:
                    summary += f"{model_name}:\n"
                    summary += f"  Train R²: {results['train_r2']:.4f} | Test R²: {results['test_r2']:.4f}\n"
                    summary += f"  Train RMSE: {results['train_rmse']:.4f} | Test RMSE: {results['test_rmse']:.4f}\n"
                    summary += f"  Train MAE: {results['train_mae']:.4f} | Test MAE: {results['test_mae']:.4f}\n\n"
        
        return summary
    
    def apply_best_preprocessing(self, X_new=None, fit_mode=False):
        if not self.best_preprocessing_steps:
            raise ValueError("No optimization performed yet. Call optimize() first.")
        
        if X_new is None:
            raise ValueError("Must provide X_new data to preprocess.")
        
        X_processed = self._apply_preprocessing_pipeline(X_new, self.best_preprocessing_steps, fit_mode=fit_mode)
        return X_processed


def optimize_raman_preprocessing(X_train, X_test, y_train, y_test, n_trials=50, cv_folds=5, progress_callback=None):
    optimizer = RamanPreprocessingOptimizer(X_train, X_test, y_train, y_test, cv_folds, n_trials)
    return optimizer.optimize(progress_callback)