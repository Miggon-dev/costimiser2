"""
Data cleaning module: outlier detection, physical-constraint filtering,
and derived feature engineering for the paper mill dataset.
"""

from typing import Iterable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


# =============================================================================
# Collection helpers
# =============================================================================

def unique_in_order(values: Iterable[T]) -> list[T]:
    """Remove duplicates while preserving first-occurrence order."""
    return list(dict.fromkeys(values))


def ordered_difference(values: Iterable[T], excluded: Iterable[T]) -> list[T]:
    """Remove excluded items while preserving order."""
    excluded_set = set(excluded)
    return [v for v in values if v not in excluded_set]


def ordered_intersection(values: Iterable[T], allowed: Iterable[T]) -> list[T]:
    """Keep only allowed items while preserving order."""
    allowed_set = set(allowed)
    return [v for v in values if v in allowed_set]


# =============================================================================
# Outlier detection
# =============================================================================

def outlier(y: pd.Series, option: str = "IQR") -> pd.Series:
    """
    Return a boolean mask where True indicates an outlier.

    Supported options: IQR, MAD, Hampel, PctClip, StrPctClip.
    """
    if option == "IQR":
        q1, q3 = y.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)

    elif option == "MAD":
        med = y.median()
        mad = (np.abs(y - med)).median()
        modz = 0.6745 * (y - med) / mad
        return modz.abs() > 3.5

    elif option == "Hampel":
        window, n = 24, 3.0
        med = y.rolling(window, center=True).median()
        mad = (np.abs(y - med)).rolling(window, center=True).median()
        return (np.abs(y - med) / (1.4826 * mad)) > n

    elif option == "PctClip":
        low, high = y.quantile([0.01, 0.99])
        return (y < low) | (y > high)

    elif option == "StrPctClip":
        low, high = y.quantile([0.001, 0.999])
        return (y < low) | (y > high)

    else:
        raise ValueError(f"Unknown outlier option: {option}")


# =============================================================================
# Design matrix construction
# =============================================================================

def make_design(
    df: pd.DataFrame,
    ycol: str,
    ux: list[str],
    u_lags: dict | None,
    y_lags=range(1, 6),
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a regression design matrix with optional lagged features.

    Parameters
    ----------
    df : DataFrame with time-series index
    ycol : target column name
    ux : exogenous variable names
    u_lags : dict mapping variable -> lag offset, or None for no lagging
    y_lags : range of autoregressive lags for the target

    Returns
    -------
    (X, y) with NaN rows dropped
    """
    cols = {}

    for u in ux:
        if u_lags is None:
            cols[u] = df[u]
        else:
            L0 = u_lags.get(u, 0)
            for L in range(L0, L0 + len(y_lags)):
                cols[f"{u}_L{L}"] = df[u].shift(L)

    for L in y_lags:
        cols[f"y_L{L}"] = df[ycol].shift(L)

    X = pd.DataFrame(cols, index=df.index)
    Y = df[ycol]

    Z = X.join(Y).dropna()
    return Z.drop(columns=[ycol]), Z[ycol]
