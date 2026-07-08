"""
Shared helpers for handling single- and multi-target (y) data consistently
across the SpectraTwin pages.

The previous code called ``y.values.ravel()`` unconditionally, which silently
corrupted multi-target data: a ``(n_samples, n_targets)`` matrix was flattened
into a 1-D vector of length ``n_samples * n_targets``, so the number of targets
no longer matched the number of feature rows ("Data inconsistency detected").

``prepare_targets`` flattens to 1-D only when there is a single target column,
and otherwise preserves the 2-D ``(n_samples, n_targets)`` shape so that the
multi-output model paths (MultiOutputRegressor / MultiOutputClassifier) work.
"""

import numpy as np
import pandas as pd


def n_targets(y):
    """Return the number of target columns in ``y``."""
    if y is None:
        return 0
    if hasattr(y, "shape") and len(getattr(y, "shape")) > 1:
        return y.shape[1]
    if isinstance(y, pd.Series):
        return 1
    arr = np.asarray(y)
    return 1 if arr.ndim == 1 else arr.shape[1]


def is_multi_target(y):
    """True when ``y`` holds more than one target column."""
    return n_targets(y) > 1


def select_target(y, column):
    """Return ``y`` reduced to a single target column.

    ``column`` may be a column name (for DataFrames) or an integer index.
    """
    if isinstance(y, pd.DataFrame):
        if column in y.columns:
            return y[[column]]
        return y.iloc[:, [int(column)]]
    arr = np.asarray(y)
    if arr.ndim == 1:
        return arr
    return arr[:, [int(column)]]


def prepare_targets(y):
    """Convert target(s) into a model-ready array without corrupting them.

    * Single target  -> 1-D array of shape ``(n_samples,)``
    * Multiple targets -> 2-D array of shape ``(n_samples, n_targets)``
    """
    if y is None:
        return None

    arr = y.values if hasattr(y, "values") else np.asarray(y)

    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr.ravel()
    return arr
