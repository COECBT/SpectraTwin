import numpy as np
import optuna
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional
import warnings
import traceback as _tb

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _snv(X: np.ndarray) -> np.ndarray:
    means = np.mean(X, axis=1, keepdims=True)
    stds  = np.std(X,  axis=1, keepdims=True)
    stds  = np.where(stds < 1e-12, 1.0, stds)
    return (X - means) / stds

def _msc(X: np.ndarray, reference: Optional[np.ndarray] = None):
    if reference is None:
        reference = np.mean(X, axis=0)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        try:
            c = np.polyfit(reference, X[i], 1)
            out[i] = (X[i] - c[1]) / (c[0] if abs(c[0]) > 1e-12 else 1.0)
        except Exception:
            out[i] = X[i]
    return out, reference

def _emsc_correction(X: np.ndarray, reference: Optional[np.ndarray] = None, poly_order: int = 2):
    if reference is None:
        reference = np.mean(X, axis=0)
    n_f = X.shape[1]
    x   = np.linspace(-1, 1, n_f)
    cols = [reference] + [x**p for p in range(poly_order + 1)]
    A    = np.column_stack(cols)
    out  = np.zeros_like(X)
    for i in range(X.shape[0]):
        try:
            coeff, *_ = np.linalg.lstsq(A, X[i], rcond=None)
            background = A[:, 1:] @ coeff[1:]
            scale      = coeff[0] if abs(coeff[0]) > 1e-12 else 1.0
            out[i]     = (X[i] - background) / scale
        except Exception:
            out[i] = X[i]
    return out, reference

def _sg_smooth(X: np.ndarray, window: int, poly: int) -> np.ndarray:
    window = window + (1 - window % 2)
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    window = min(window, X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2)
    if window <= poly:
        return X
    try:
        return savgol_filter(X, window_length=window, polyorder=poly, axis=1)
    except Exception:
        return X

def _sg_deriv(X: np.ndarray, window: int, poly: int, order: int) -> np.ndarray:
    window = window + (1 - window % 2)
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    window = min(window, X.shape[1] - 1 if (X.shape[1] - 1) % 2 == 1 else X.shape[1] - 2)
    if window <= poly:
        return X
    try:
        return savgol_filter(X, window_length=window, polyorder=poly, deriv=order, axis=1)
    except Exception:
        return X

def _rolling_smooth(X: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.apply_along_axis(lambda r: np.convolve(r, kernel, mode='same'), axis=1, arr=X)

def _asls_baseline(X: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    n_f = X.shape[1]
    D   = lam * (np.diff(np.eye(n_f), n=2, axis=0).T @ np.diff(np.eye(n_f), n=2, axis=0))
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        y = X[i]; w = np.ones(n_f)
        for _ in range(niter):
            Z = np.linalg.solve(np.diag(w) + D, w * y)
            w = np.where(y > Z, p, 1 - p)
        out[i] = y - Z
    return out

def _polyfit_baseline(X: np.ndarray, order: int, niter: int) -> np.ndarray:
    n_f = X.shape[1]; x = np.arange(n_f)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        y = X[i].copy()
        for _ in range(niter):
            try:
                bl = np.polyval(np.polyfit(x, y, order), x)
                y  = np.minimum(y, bl)
            except Exception:
                break
        try:
            out[i] = X[i] - np.polyval(np.polyfit(x, y, order), x)
        except Exception:
            out[i] = X[i]
    return out

def _detrend(X: np.ndarray, order: int) -> np.ndarray:
    n_f = X.shape[1]; x = np.arange(n_f)
    out = np.zeros_like(X)
    for i in range(X.shape[0]):
        try:
            out[i] = X[i] - np.polyval(np.polyfit(x, X[i], order), x)
        except Exception:
            out[i] = X[i]
    return out

def _vector_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n < 1e-12, 1.0, n)

def _area_norm(X: np.ndarray) -> np.ndarray:
    a = np.trapz(np.abs(X), axis=1, keepdims=True)
    return X / np.where(a < 1e-12, 1.0, a)

def _minmax_norm(X: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    xmin = np.min(X, axis=1, keepdims=True)
    xmax = np.max(X, axis=1, keepdims=True)
    rng  = np.where((xmax - xmin) < 1e-12, 1.0, xmax - xmin)
    return (X - xmin) / rng * (hi - lo) + lo

def _pareto(X: np.ndarray) -> np.ndarray:
    means = np.mean(X, axis=1, keepdims=True)
    stds  = np.std(X,  axis=1, keepdims=True)
    stds  = np.where(stds < 1e-12, 1.0, stds)
    return (X - means) / np.sqrt(stds)

def _mean_center_spectrum(X: np.ndarray) -> np.ndarray:
    return X - np.mean(X, axis=1, keepdims=True)

def _water_band_mask(X: np.ndarray, n_features: int, mask_ranges: List[Tuple[int, int]]) -> np.ndarray:
    X_out = X.copy()
    for s, e in mask_ranges:
        s = max(0, min(s, n_features - 1))
        e = max(0, min(e, n_features - 1))
        if s <= e:
            X_out[:, s: e + 1] = 0.0
    return X_out

class NIRPreprocessingOptimizer:

    def __init__(self, X_train, X_test, y_train, y_test, cv_folds: int = 5, n_trials: int = 50, random_state: int = 42):
        self.X_train = np.array(X_train, dtype=float)
        self.X_test  = np.array(X_test,  dtype=float)
        self.y_train = np.array(y_train, dtype=float).ravel()
        self.y_test  = np.array(y_test,  dtype=float).ravel()

        self.cv_folds     = cv_folds
        self.n_trials     = n_trials
        self.random_state = random_state

        self._validate_input_data()

        self.models = {
            'PLS':          PLSRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1),
        }

        self.best_params              = None
        self.best_score               = -np.inf
        self.best_preprocessing_steps = []
        self.optimization_history     = []
        self.all_model_results        = {}
        self.current_trial            = 0
        self.fitted_preprocessing_params = {}
        self.auto_trim_params         = None
        self.plot_callback            = None

    def _validate_input_data(self):
        for name, X, y in [('train', self.X_train, self.y_train), ('test',  self.X_test,  self.y_test)]:
            if X.shape[0] != len(y):
                raise ValueError(f"X_{name}/y_{name} sample count mismatch")
            for arr, lbl in [(X, f'X_{name}'), (y, f'y_{name}')]:
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(f"{lbl} contains NaN or Inf")
        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("Feature dimension mismatch")
        if self.X_train.shape[1] < 4:
            raise ValueError("Need at least 4 wavelength channels")
        if self.X_train.shape[0] < 10:
            raise ValueError("Need at least 10 training samples")

    def _apply_preprocessing_pipeline(self, X_data: np.ndarray, pipeline: List[Dict], fit_mode: bool = True) -> np.ndarray:
        X = X_data.copy().astype(float)
        if fit_mode:
            self.fitted_preprocessing_params = {}

        for step_idx, step in enumerate(pipeline):
            method   = step['method']
            params   = step.get('params', {})
            step_key = f"step_{step_idx}_{method}"
            X_before = X.copy()

            try:
                if method == 'Trim':
                    X = X[:, params['start_idx']: params['end_idx'] + 1]

                elif method == 'Water_Band_Mask':
                    nf = X.shape[1]
                    r1s = int(params.get('r1_start', 0.35) * nf)
                    r1e = int(params.get('r1_end',   0.45) * nf)
                    r2s = int(params.get('r2_start', 0.70) * nf)
                    r2e = int(params.get('r2_end',   0.80) * nf)
                    X = _water_band_mask(X, nf, [(r1s, r1e), (r2s, r2e)])

                elif method == 'AsLS':
                    X = _asls_baseline(X, lam=params['lam'], p=params['p'], niter=params['niter'])

                elif method == 'Polyfit':
                    X = _polyfit_baseline(X, order=params['order'], niter=params['niter'])

                elif method == 'Savitzky-Golay':
                    X = _sg_smooth(X, window=params['window'], poly=params['poly'])

                elif method == 'Rolling':
                    X = _rolling_smooth(X, window=params['window'])

                elif method == 'SNV':
                    X = _snv(X)

                elif method == 'MSC':
                    if fit_mode:
                        X, ref = _msc(X)
                        self.fitted_preprocessing_params[step_key] = {'reference': ref}
                    else:
                        ref = self.fitted_preprocessing_params.get(step_key, {}).get('reference', None)
                        X, _ = _msc(X, reference=ref)

                elif method == 'EMSC':
                    order = params.get('poly_order', 2)
                    if fit_mode:
                        X, ref = _emsc_correction(X, poly_order=order)
                        self.fitted_preprocessing_params[step_key] = {'reference': ref, 'poly_order': order}
                    else:
                        ref = self.fitted_preprocessing_params.get(step_key, {}).get('reference', None)
                        X, _ = _emsc_correction(X, reference=ref, poly_order=order)

                elif method == 'Detrend':
                    X = _detrend(X, order=params['order'])

                elif method == 'Area':
                    X = _area_norm(X)

                elif method == 'Vector':
                    X = _vector_norm(X)

                elif method == 'Min-max':
                    X = _minmax_norm(X, lo=params.get('minv', 0.0), hi=params.get('maxv', 1.0))

                elif method == 'Pareto':
                    X = _pareto(X)

                elif method == 'Mean (spectrum)':
                    X = _mean_center_spectrum(X)

                elif method == 'Mean (wavelength)':
                    if fit_mode:
                        wm = np.mean(X, axis=0, keepdims=True)
                        self.fitted_preprocessing_params[step_key] = {'wl_mean': wm}
                    else:
                        wm = self.fitted_preprocessing_params.get(step_key, {}).get('wl_mean', 0.0)
                    X = X - wm

                elif method == 'SG Derivative':
                    X = _sg_deriv(X, window=params['window'], poly=params['poly'], order=params['order'])

                if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                    X = X_before

            except Exception:
                X = X_before

        return X

    def _create_automated_pipeline(self, trial) -> List[Dict]:
        pipeline: List[Dict] = []

        if self.auto_trim_params:
            if trial.suggest_categorical('use_auto_trim', [True, False]):
                pipeline.append({'method': 'Trim', 'params': self.auto_trim_params})

        if trial.suggest_categorical('use_water_mask', [True, False]):
            pipeline.append({'method': 'Water_Band_Mask', 'params': {
                'r1_start': 0.35, 'r1_end': 0.45,
                'r2_start': 0.70, 'r2_end': 0.80,
            }})

        if trial.suggest_categorical('use_baseline', [True, False]):
            bm = trial.suggest_categorical('baseline_method', ['AsLS', 'Polyfit'])
            if bm == 'AsLS':
                pipeline.append({'method': 'AsLS', 'params': {
                    'lam':   trial.suggest_float('asls_lam',   1e4, 1e7, log=True),
                    'p':     trial.suggest_float('asls_p',     0.01, 0.09, log=True),
                    'niter': trial.suggest_int('asls_niter',   5, 20),
                }})
            else:
                pipeline.append({'method': 'Polyfit', 'params': {
                    'order': trial.suggest_int('poly_order', 2, 6),
                    'niter': trial.suggest_int('poly_niter', 1, 15),
                }})

        if trial.suggest_categorical('use_smoothing', [True, False]):
            sm = trial.suggest_categorical('smooth_method', ['Savitzky-Golay', 'Rolling'])
            if sm == 'Savitzky-Golay':
                w = trial.suggest_int('sg_window', 5, 21, step=2)
                pipeline.append({'method': 'Savitzky-Golay', 'params': {
                    'window': w,
                    'poly':   trial.suggest_int('sg_poly', 2, min(4, w - 1)),
                }})
            else:
                pipeline.append({'method': 'Rolling', 'params': {
                    'window': trial.suggest_int('rolling_window', 3, 11, step=2),
                }})

        nm = trial.suggest_categorical('normalization_method', ['SNV', 'MSC', 'EMSC', 'Detrend', 'Area', 'Vector', 'Min-max', 'Pareto'])
        if nm == 'EMSC':
            pipeline.append({'method': 'EMSC', 'params': {'poly_order': trial.suggest_int('emsc_poly', 1, 4)}})
        elif nm == 'Detrend':
            pipeline.append({'method': 'Detrend', 'params': {'order': trial.suggest_int('detrend_order', 1, 3)}})
        elif nm == 'Min-max':
            pipeline.append({'method': 'Min-max', 'params': {
                'minv': trial.suggest_float('minmax_min', 0.0, 0.1),
                'maxv': trial.suggest_float('minmax_max', 0.9, 1.0),
            }})
        else:
            pipeline.append({'method': nm, 'params': {}})

        if trial.suggest_categorical('use_centering', [True, False]):
            pipeline.append({
                'method': trial.suggest_categorical('center_method', ['Mean (spectrum)', 'Mean (wavelength)']),
                'params': {},
            })

        w = trial.suggest_int('sgd_window', 5, 21, step=2)
        pipeline.append({'method': 'SG Derivative', 'params': {
            'window': w,
            'poly':   trial.suggest_int('sgd_poly',  2, min(4, w - 1)),
            'order':  trial.suggest_int('sgd_order', 1, 2),
        }})

        return pipeline

    def _create_model(self, model_name: str):
        if model_name == 'PLS':
            return PLSRegression()
        if model_name == 'RandomForest':
            return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        raise ValueError(f"Unknown model: {model_name}")

    def _evaluate_pipeline(self, pipeline: List[Dict]) -> Tuple[float, float, Dict]:
        try:
            model_scores: Dict = {}
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            for model_name in self.models:
                fold_scores, train_fold_scores = [], []

                for tr_idx, va_idx in kf.split(self.X_train):
                    X_tr_p = self._apply_preprocessing_pipeline(self.X_train[tr_idx], pipeline, fit_mode=True)
                    if (np.any(np.isnan(X_tr_p)) or np.any(np.isinf(X_tr_p)) or X_tr_p.shape[1] < 1 or np.sum(np.var(X_tr_p, axis=0) > 1e-10) < 1):
                        fold_scores.append(-1000); train_fold_scores.append(-1000); continue

                    X_va_p = self._apply_preprocessing_pipeline(self.X_train[va_idx], pipeline, fit_mode=False)
                    if np.any(np.isnan(X_va_p)) or np.any(np.isinf(X_va_p)):
                        fold_scores.append(-1000); train_fold_scores.append(-1000); continue

                    model = self._create_model(model_name)
                    if model_name == 'PLS':
                        mc = min(X_tr_p.shape[0] - 1, X_tr_p.shape[1])
                        model.n_components = max(1, min(10, mc))

                    model.fit(X_tr_p, self.y_train[tr_idx])
                    y_va_pred = np.ravel(model.predict(X_va_p))
                    y_tr_pred = np.ravel(model.predict(X_tr_p))

                    fs = r2_score(self.y_train[va_idx], y_va_pred)
                    ft = r2_score(self.y_train[tr_idx], y_tr_pred)
                    fold_scores.append(float(fs) if not np.isnan(fs) else -1000)
                    train_fold_scores.append(float(ft) if not np.isnan(ft) else -1000)

                mv = float(np.mean(fold_scores))
                mt = float(np.mean(train_fold_scores))
                model_scores[model_name] = {'test': mv if not np.isnan(mv) else -1000, 'train': mt if not np.isnan(mt) else -1000}

            best = max(model_scores, key=lambda k: model_scores[k]['test'])
            return model_scores[best]['test'], model_scores[best]['train'], model_scores

        except Exception:
            return -1000.0, -1000.0, {}

    def _objective(self, trial) -> float:
        self.current_trial = trial.number
        pipeline = self._create_automated_pipeline(trial)
        score, train_score, model_scores = self._evaluate_pipeline(pipeline)

        if self.plot_callback and trial.number % max(1, self.n_trials // 15) == 0:
            try:
                Xp = self._apply_preprocessing_pipeline(self.X_train[:min(5, len(self.X_train))], pipeline, True)
                self.plot_callback(Xp, pipeline, score, train_score)
            except Exception:
                pass

        self.optimization_history.append({
            'trial': trial.number, 'score': score,
            'train_score': train_score, 'model_scores': model_scores,
            'pipeline': pipeline, 'params': trial.params,
        })
        return score

    def optimize(self, progress_callback=None, plot_callback=None) -> Dict:
        self.plot_callback = plot_callback
        try:
            fv  = np.var(self.X_train, axis=0)
            act = np.where(fv > np.max(fv) * 1e-4)[0]
            if len(act):
                s, e = int(act[0]), int(act[-1])
                self.auto_trim_params = {'type': 'Trim', 'start_idx': s, 'end_idx': e} if s > 0 or e < len(fv) - 1 else None
            else:
                self.auto_trim_params = None

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(),
            )

            def _obj(trial):
                try:
                    if progress_callback:
                        pct = min(80, int(trial.number / self.n_trials * 80))
                        progress_callback(pct, f"Trial {trial.number+1}/{self.n_trials}")
                    return self._objective(trial)
                except Exception:
                    return -1000.0

            study.optimize(_obj, n_trials=self.n_trials)

            self.best_params = study.best_params
            self.best_score  = study.best_value
            for info in self.optimization_history:
                if info['score'] == self.best_score:
                    self.best_preprocessing_steps = info['pipeline']
                    break

            self._train_and_evaluate_all_models()

            return {
                'success':             True,
                'cv_score':            self.best_score,
                'all_model_results':   self.all_model_results,
                'best_pipeline':       self.best_preprocessing_steps,
                'best_params':         self.best_params,
                'optimization_study':  study,
                'n_trials_completed':  len(self.optimization_history),
                'train_size':          self.X_train.shape[0],
                'test_size':           self.X_test.shape[0],
                'summary':             self._generate_summary(),
            }

        except Exception as exc:
            return {
                'success':            False,
                'error':              str(exc),
                'traceback':          _tb.format_exc(),
                'cv_score':           float(self.best_score),
                'all_model_results':  None,
                'best_pipeline':      self.best_preprocessing_steps,
                'best_params':        self.best_params,
                'n_trials_completed': len(self.optimization_history),
                'summary':            f"Optimization failed: {exc}",
            }

    def _train_and_evaluate_all_models(self):
        try:
            X_tr = self._apply_preprocessing_pipeline(self.X_train, self.best_preprocessing_steps, fit_mode=True)
            X_te = self._apply_preprocessing_pipeline(self.X_test,  self.best_preprocessing_steps, fit_mode=False)
            self.all_model_results = {}
            for model_name in self.models:
                model = self._create_model(model_name)
                if model_name == 'PLS':
                    mc = min(X_tr.shape[0] - 1, X_tr.shape[1])
                    model.n_components = max(1, min(10, mc))
                model.fit(X_tr, self.y_train)
                y_tr_pred = np.ravel(model.predict(X_tr))
                y_te_pred = np.ravel(model.predict(X_te))
                self.all_model_results[model_name] = {
                    'train_r2':        r2_score(self.y_train, y_tr_pred),
                    'train_rmse':      float(np.sqrt(mean_squared_error(self.y_train, y_tr_pred))),
                    'train_mae':       float(mean_absolute_error(self.y_train, y_tr_pred)),
                    'test_r2':         r2_score(self.y_test,  y_te_pred),
                    'test_rmse':       float(np.sqrt(mean_squared_error(self.y_test,  y_te_pred))),
                    'test_mae':        float(mean_absolute_error(self.y_test,  y_te_pred)),
                    'y_train_pred':    y_tr_pred,
                    'y_test_pred':     y_te_pred,
                    'train_residuals': self.y_train - y_tr_pred,
                    'test_residuals':  self.y_test  - y_te_pred,
                }
        except Exception as exc:
            self.all_model_results = {'error': str(exc)}

    def _generate_summary(self) -> str:
        if self.best_score == -np.inf:
            return "No optimization performed yet."
        lines = [
            "NIR SPECTROSCOPY PREPROCESSING OPTIMIZATION",
            "=" * 70,
            f"Best CV R²         : {self.best_score:.4f}",
            f"CV Folds           : {self.cv_folds}  |  Trials: {self.n_trials}",
            f"Train / Test       : {self.X_train.shape[0]} / {self.X_test.shape[0]}",
            f"Wavelength channels: {self.X_train.shape[1]}",
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

    def apply_best_preprocessing(self, X_new: np.ndarray, fit_mode: bool = False) -> np.ndarray:
        if self.best_score == -np.inf:
            raise ValueError("No optimization yet. Call optimize() first.")
        return self._apply_preprocessing_pipeline(np.array(X_new, dtype=float), self.best_preprocessing_steps, fit_mode=fit_mode)


def optimize_nir_preprocessing(X_train, X_test, y_train, y_test, n_trials=50, cv_folds=5, progress_callback=None) -> Dict:
    opt = NIRPreprocessingOptimizer(X_train, X_test, y_train, y_test, cv_folds=cv_folds, n_trials=n_trials)
    return opt.optimize(progress_callback=progress_callback)