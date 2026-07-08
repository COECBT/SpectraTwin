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

class FTIRPreprocessingOptimizer:
    
    def __init__(self, X, y, wavenumbers=None, cv_folds=5, n_trials=50, test_size=0.2, random_state=42):
        self.X = np.array(X)
        self.y = np.array(y)
        self.wavenumbers = wavenumbers
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
            'SVR': SVR(),
            'RandomForest': RandomForestRegressor(random_state=self.random_state),
            'XGBoost': XGBRegressor(random_state=self.random_state, verbosity=0),
            'KNN': KNeighborsRegressor()
        }
        
        self.best_params = None
        self.best_score = -np.inf
        self.best_model = None
        self.best_model_instance = None
        self.best_preprocessing_steps = []
        self.best_scaler = None
        self.optimization_history = []
        self.test_results = None
        
        self.current_trial = 0
        
    def _validate_input_data(self):
        if self.X.shape[0] != len(self.y):
            raise ValueError(f"X and y dimension mismatch: X has {self.X.shape[0]} samples, y has {len(self.y)} samples")
        
        if np.any(np.isnan(self.X)) or np.any(np.isinf(self.X)):
            raise ValueError("X contains NaN or infinite values")
        
        if np.any(np.isnan(self.y)) or np.any(np.isinf(self.y)):
            raise ValueError("y contains NaN or infinite values")
        
        if self.X.shape[1] < 2:
            raise ValueError("X must have at least 2 features")
        
        if self.X.shape[0] < 10:
            raise ValueError("Dataset too small. Need at least 10 samples.")

    def _create_automated_pipeline(self, trial) -> Tuple[List[Dict], str, Dict]:
        pipeline = []
        
        baseline_method = trial.suggest_categorical('baseline_method', ['AsLS', 'Polyfit', 'Pearson'])
        
        if baseline_method == 'AsLS':
            lam = trial.suggest_float('asls_lam', 1e4, 1e9, log=True)
            p = trial.suggest_float('asls_p', 0.001, 0.1, log=True)
            niter = trial.suggest_int('asls_niter', 5, 30)
            pipeline.append({
                'method': 'AsLS',
                'params': {'lam': lam, 'p': p, 'niter': niter}
            })
        elif baseline_method == 'Polyfit':
            order = trial.suggest_int('poly_order', 1, 5)
            niter = trial.suggest_int('poly_niter', 1, 15)
            pipeline.append({
                'method': 'Polyfit',
                'params': {'order': order, 'niter': niter}
            })
        elif baseline_method == 'Pearson':
            u = trial.suggest_int('pearson_u', 2, 8)
            v = trial.suggest_int('pearson_v', 1, 6)
            pipeline.append({
                'method': 'Pearson',
                'params': {'u': u, 'v': v}
            })

        norm_method = trial.suggest_categorical('normalization_method', 
            ['Area', 'Peak Normalization', 'Vector', 'Min-max', 'Pareto'])
        
        if norm_method == 'Peak Normalization':
            wave = trial.suggest_float('peak_wave', 0, self.X_train.shape[1]-1)
            pipeline.append({
                'method': 'Peak Normalization',
                'params': {'wave': wave}
            })
        elif norm_method == 'Min-max':
            minv = trial.suggest_float('minmax_min', 0.0, 0.1)
            maxv = trial.suggest_float('minmax_max', 0.9, 1.0)
            pipeline.append({
                'method': 'Min-max',
                'params': {'minv': minv, 'maxv': maxv}
            })
        else:
            pipeline.append({
                'method': norm_method,
                'params': {}
            })
        
        if trial.suggest_categorical('use_smoothing', [True, False]):
            smooth_method = trial.suggest_categorical('smooth_method', ['Savitzky-Golay', 'Rolling'])
            
            if smooth_method == 'Savitzky-Golay':
                window = trial.suggest_int('sg_window', 5, 15, step=2)
                poly = trial.suggest_int('sg_poly', 2, min(4, window-1))
                pipeline.append({
                    'method': 'Savitzky-Golay',
                    'params': {'window': window, 'poly': poly}
                })
            elif smooth_method == 'Rolling':
                window = trial.suggest_int('rolling_window', 3, 11, step=2)
                pipeline.append({
                    'method': 'Rolling',
                    'params': {'window': window}
                })
        
        if trial.suggest_categorical('use_center', [True, False]):
            center_method = trial.suggest_categorical('center_method', 
                ['Mean (spectrum)', 'Mean (wavelength)', 'Last Point'])
            pipeline.append({
                'method': center_method,
                'params': {}
            })
        
        if trial.suggest_categorical('use_derivative', [True, False]):
            deriv_option = trial.suggest_categorical('derivative_option', ['Subtract', 'Reset'])
            
            if deriv_option == 'Subtract':
                subtract_idx = trial.suggest_int('subtract_idx', 1, min(10, self.X_train.shape[0]))
                pipeline.append({
                    'method': 'Derivative_Subtract',
                    'params': {'subtract_idx': subtract_idx}
                })
            else:
                pipeline.append({
                    'method': 'Derivative_Reset',
                    'params': {}
                })
        
        if trial.suggest_categorical('use_sg_derivative', [True, False]):
            window = trial.suggest_int('sgd_window', 7, 17, step=2)
            poly = trial.suggest_int('sgd_poly', 2, min(4, window-1))
            order = trial.suggest_int('sgd_order', 1, 2)
            pipeline.append({
                'method': 'SG Derivative',
                'params': {'window': window, 'poly': poly, 'order': order}
            })
        
        model_name = trial.suggest_categorical('model', ['PLS', 'SVR', 'RandomForest', 'XGBoost', 'KNN'])
        model_params = self._get_model_params(trial, model_name)
        
        return pipeline, model_name, model_params
    
    def _get_model_params(self, trial, model_name: str) -> Dict:
        if model_name == 'PLS':
            max_components = min(25, self.X_train.shape[1]//3, self.X_train.shape[0]-1)
            return {
                'n_components': trial.suggest_int('pls_n_components', 1, max_components)
            }
        
        elif model_name == 'SVR':
            gamma_type = trial.suggest_categorical('svr_gamma_type', ['fixed', 'float'])
            if gamma_type == 'fixed':
                gamma = trial.suggest_categorical('svr_gamma', ['scale', 'auto'])
            else:
                gamma = trial.suggest_float('svr_gamma_val', 1e-4, 1e-1, log=True)
                
            return {
                'C': trial.suggest_float('svr_C', 0.1, 100.0, log=True),
                'gamma': gamma,
                'epsilon': trial.suggest_float('svr_epsilon', 0.01, 1.0),
                'kernel': trial.suggest_categorical('svr_kernel', ['rbf', 'linear', 'poly'])
            }
        
        elif model_name == 'XGBoost':
            return {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 1.0, log=True)
            }
        
        elif model_name == 'RandomForest':
            return {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 15),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 8),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 4),
                'max_features': trial.suggest_categorical('rf_max_features', ['auto', 'sqrt', 'log2'])
            }
        
        elif model_name == 'KNN':
            return {
                'n_neighbors': trial.suggest_int('knn_n_neighbors', 3, min(10, self.X_train.shape[0]-1)),
                'weights': trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('knn_algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                'p': trial.suggest_int('knn_p', 1, 2)
            }
        
        return {}
    
    def _apply_preprocessing_pipeline(self, X, pipeline: List[Dict]):
        temp_file = None
        try:
            X_processed = X.copy()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                temp_file = tmp.name
                pd.DataFrame(X_processed).to_csv(temp_file, index=False)
            
            spectral_data = SpectralData(temp_file)
            
            for step in pipeline:
                method = step['method']
                params = step.get('params', {})

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

                    elif method == 'Area':
                        spectral_data.area()

                    elif method == 'Peak Normalization':
                        spectral_data.peaknorm(wavenumber=params['wave'])

                    elif method == 'Vector':
                        spectral_data.vector()

                    elif method == 'Min-max':
                        spectral_data.minmax(min_val=params['minv'], max_val=params['maxv'])

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

                    else:
                        continue

                except Exception:
                    continue

            X_processed = spectral_data.spc.values
            
            return X_processed
            
        except Exception:
            return X.copy()
            
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def _create_model(self, model_name: str, model_params: Dict):
        base_params = {}
        if model_name in ['XGBoost', 'RandomForest']:
            base_params['random_state'] = self.random_state
        if model_name == 'XGBoost':
            base_params['verbosity'] = 0
        
        all_params = {**base_params, **model_params}
        
        if model_name == 'PLS':
            return PLSRegression(**all_params)
        elif model_name == 'SVR':
            return SVR(**all_params)
        elif model_name == 'XGBoost':
            return XGBRegressor(**all_params)
        elif model_name == 'RandomForest':
            return RandomForestRegressor(**all_params)
        elif model_name == 'KNN':
            return KNeighborsRegressor(**all_params)
        
        raise ValueError(f"Unknown model: {model_name}")
    
    def _evaluate_pipeline(self, pipeline: List[Dict], model_name: str, model_params: Dict) -> float:
        try:
            X_processed = self._apply_preprocessing_pipeline(self.X_train, pipeline)
            
            if np.any(np.isnan(X_processed)) or np.any(np.isinf(X_processed)):
                return -1000
            
            if X_processed.shape[1] < 1:
                return -1000
            
            feature_vars = np.var(X_processed, axis=0)
            if np.sum(feature_vars > 1e-10) < 1:
                return -1000
            
            model = self._create_model(model_name, model_params)
            
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            if model_name in ['SVR', 'KNN']:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_processed)
                cv_scores = cross_val_score(model, X_scaled, self.y_train, cv=kf, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_processed, self.y_train, cv=kf, scoring='r2')
            
            mean_score = np.mean(cv_scores)
            return mean_score if not np.isnan(mean_score) else -1000
            
        except Exception:
            return -1000
    
    def _objective(self, trial):
        self.current_trial = trial.number
        pipeline, model_name, model_params = self._create_automated_pipeline(trial)
        score = self._evaluate_pipeline(pipeline, model_name, model_params)
        
        self.optimization_history.append({
            'trial': trial.number,
            'score': score,
            'pipeline': pipeline,
            'model_name': model_name,
            'model_params': model_params,
            'params': trial.params
        })
        
        return score
    
    def optimize(self, progress_callback=None):
        
        try:
            if progress_callback:
                progress_callback(0, "Initializing FTIR optimization...")
            
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )
            
            def progress_objective(trial):
                try:
                    if progress_callback:
                        progress = min(80, int((trial.number / self.n_trials) * 80))
                        progress_callback(progress, f"FTIR Trial {trial.number + 1}/{self.n_trials}")
                    return self._objective(trial)
                except Exception:
                    return -1000
            
            study.optimize(progress_objective, n_trials=self.n_trials)
            
            if progress_callback:
                progress_callback(85, "Training best FTIR model on full training set...")
            
            self.best_params = study.best_params
            self.best_score = study.best_value
            
            for trial_info in self.optimization_history:
                if trial_info['score'] == self.best_score:
                    self.best_preprocessing_steps = trial_info['pipeline']
                    self.best_model = trial_info['model_name']
                    break
            
            if progress_callback:
                progress_callback(90, "Evaluating FTIR model on test set...")
            
            self._train_and_evaluate_final_model()
            
            if progress_callback:
                progress_callback(100, "FTIR optimization complete!")
            
            return {
                'success': True,
                'cv_score': self.best_score,
                'test_results': self.test_results,
                'best_model': self.best_model,
                'best_pipeline': self.best_preprocessing_steps,
                'best_params': self.best_params,
                'optimization_study': study,
                'n_trials_completed': len(self.optimization_history),
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'summary': self._generate_summary()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'cv_score': self.best_score,
                'test_results': None,
                'best_model': self.best_model,
                'best_pipeline': self.best_preprocessing_steps,
                'best_params': self.best_params,
                'n_trials_completed': len(self.optimization_history),
                'summary': f"FTIR optimization failed: {str(e)}"
            }
    
    def _train_and_evaluate_final_model(self):
        try:
            X_train_processed = self._apply_preprocessing_pipeline(self.X_train, self.best_preprocessing_steps)
            X_test_processed = self._apply_preprocessing_pipeline(self.X_test, self.best_preprocessing_steps)
            
            model_params = {}
            prefix = self.best_model.lower() + '_'
            for key, value in self.best_params.items():
                if key.startswith(prefix):
                    param_name = key[len(prefix):]
                    model_params[param_name] = value
            
            model = self._create_model(self.best_model, model_params)
            
            if self.best_model in ['SVR', 'KNN']:
                self.best_scaler = StandardScaler()
                X_train_scaled = self.best_scaler.fit_transform(X_train_processed)
                X_test_scaled = self.best_scaler.transform(X_test_processed)
                
                model.fit(X_train_scaled, self.y_train)
                y_train_pred = model.predict(X_train_scaled)
                y_test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_processed, self.y_train)
                y_train_pred = model.predict(X_train_processed)
                y_test_pred = model.predict(X_test_processed)
            
            self.best_model_instance = model
            
            self.test_results = {
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
            self.test_results = {
                'error': str(e),
                'train_r2': None,
                'test_r2': None
            }
    
    def _generate_summary(self) -> str:
        if not self.best_preprocessing_steps:
            return "No FTIR optimization performed yet."
        
        summary = f"AUTOMATED FTIR PREPROCESSING AND MODEL OPTIMIZATION RESULTS\n"
        summary += "=" * 70 + "\n"
        summary += f"Best CV R² Score (Training): {self.best_score:.4f}\n"
        
        if self.test_results and 'test_r2' in self.test_results and self.test_results['test_r2'] is not None:
            summary += f"Final Test R² Score: {self.test_results['test_r2']:.4f}\n"
            summary += f"Final Test RMSE: {self.test_results['test_rmse']:.4f}\n"
            summary += f"Final Test MAE: {self.test_results['test_mae']:.4f}\n"
        
        summary += f"Best Model: {self.best_model}\n"
        summary += f"CV Folds: {self.cv_folds} | Trials: {self.n_trials}\n"
        summary += f"Train Size: {self.X_train.shape[0]} | Test Size: {self.X_test.shape[0]}\n"
        summary += f"Features: {self.X_train.shape[1]}\n"
        summary += "=" * 70 + "\n\n"
        
        summary += "FTIR PREPROCESSING PIPELINE:\n"
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
        
        summary += f"{self.best_model.upper()} MODEL PARAMETERS:\n"
        summary += "-" * 30 + "\n"
        model_params = {}
        prefix = self.best_model.lower() + '_'
        for key, value in self.best_params.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                model_params[param_name] = value
        
        if model_params:
            for param, value in model_params.items():
                if isinstance(value, float):
                    summary += f"• {param}: {value:.4f}\n"
                else:
                    summary += f"• {param}: {value}\n"
        
        return summary
    
    def apply_best_preprocessing(self, X_new=None):
        if not self.best_preprocessing_steps:
            raise ValueError("No optimization performed yet. Call optimize() first.")
        
        if X_new is None:
            X_new = self.X
        
        X_processed = self._apply_preprocessing_pipeline(X_new, self.best_preprocessing_steps)
        
        if self.best_scaler is not None:
            X_processed = self.best_scaler.transform(X_processed)
        
        return X_processed
    
    def predict(self, X_new):
        if self.best_model_instance is None:
            raise ValueError("No trained model available. Call optimize() first.")
        
        X_processed = self.apply_best_preprocessing(X_new)
        return self.best_model_instance.predict(X_processed)
    
    def get_best_model(self):
        if self.best_model_instance is None:
            raise ValueError("No trained model available. Call optimize() first.")
        
        return self.best_model_instance

def optimize_ftir_preprocessing(X, y, wavenumbers=None, n_trials=50, cv_folds=5, test_size=0.2, progress_callback=None):
    optimizer = FTIRPreprocessingOptimizer(X, y, wavenumbers, cv_folds, n_trials, test_size)
    return optimizer.optimize(progress_callback)