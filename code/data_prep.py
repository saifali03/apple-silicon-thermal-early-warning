"""
data_prep.py
============
Phase 5 & 6: Chronological Session Split + Preprocessing

Preprocessing Strategy
----------------------
The ML-ready feature matrix contains three distinct column families that need
different treatment. This module detects them automatically at fit-time:

FAMILY 1 — MULTIMODAL (explicitly listed with fixed EDA thresholds)
  Raw columns with operating-regime splits defined manually from EDA.
  → Fixed threshold → binary flag column added
  → Original column retained and scaled with RobustScaler
  Children (lag_*, roll_*, interaction cols whose name starts with
  a multimodal parent) inherit the RobustScaler treatment ONLY — the
  flag is only meaningful on the base column.

FAMILY 2 — BOX-COX (strictly positive, non-constant, non-binary, non-diff/accel)
  All remaining columns that are verified strictly positive (min > 0)
  AND are not diff/accel derivatives (which may be negative).
  → Box-Cox via sklearn PowerTransformer(method="box-cox", standardize=True)

FAMILY 3 — YEO-JOHNSON (fallback for anything not in Family 1 or 2)
  Catches diff_*, accel columns, zero-containing columns,
  and any column the positivity check fails for.
  → Yeo-Johnson via PowerTransformer(method="yeo-johnson", standardize=True)

DROPPED before all of this:
  - META_COLS and TARGET_COL
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — edit these for your project
# ---------------------------------------------------------------------------

META_COLS = [
    "session_id",
    "timestamp_utc",
    "os_family",
    "thermal_pressure_code",
    "thermal_pressure_level",
]
TARGET_COL = "Y"

# Manual thresholds from EDA eyeballing
MULTIMODAL_THRESHOLDS = {
    "cpu_die_temp_c": 65.0,
    "cpu_total_active_pct": 60.0,
    "cpu_pcluster_active_pct": 80.0,
    "cpu_pcluster_freq_mhz": 4600.0,
    "cpu_power_mw": 15000.0,
    "combined_power_mw": 15000.0,
    "loadavg_5m": 2.5,
    "ram_used_gb": 7.0,
}

MULTIMODAL_BASE_COLS = list(MULTIMODAL_THRESHOLDS.keys())

# Regex patterns identifying diff / accel engineered features.
_DIFF_ACCEL_RE = re.compile(r"(_diff_\d+|_accel)$")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_diff_or_accel(col: str) -> bool:
    """True if this column is a rate-of-change or acceleration feature."""
    return bool(_DIFF_ACCEL_RE.search(col))

def _multimodal_parent(col: str) -> Optional[str]:
    """
    Return the multimodal base column that this column was derived from,
    or None if it has no multimodal ancestor.
    """
    for base in sorted(MULTIMODAL_BASE_COLS, key=len, reverse=True):
        if col == base or col.startswith(base + "_"):
            return base
    return None

def _col_is_strictly_positive(series: pd.Series) -> bool:
    """True if every non-NaN value is strictly > 0."""
    s = series.dropna()
    if s.empty:
        return False
    return bool((s > 0).all())

# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def chronological_session_split(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by WHOLE SESSIONS ordered by session start time.
    Prevents any lag/rolling feature context from leaking across splits.
    """
    logger.info(
        "Chronological session split — train %.0f%% / val %.0f%% / test %.0f%%.",
        train_pct * 100, val_pct * 100, (1 - train_pct - val_pct) * 100,
    )

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    session_starts = df.groupby("session_id")["timestamp_utc"].min().sort_values()
    ordered = session_starts.index.tolist()
    n = len(ordered)

    i_train = int(n * train_pct)
    i_val = int(n * (train_pct + val_pct))

    train_s = ordered[:i_train]
    val_s = ordered[i_train:i_val]
    test_s = ordered[i_val:]

    logger.info(
        "Sessions assigned — train: %d, val: %d, test: %d.",
        len(train_s), len(val_s), len(test_s),
    )

    return (
        df[df["session_id"].isin(train_s)].copy(),
        df[df["session_id"].isin(val_s)].copy(),
        df[df["session_id"].isin(test_s)].copy(),
    )

def _separate(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Peel metadata and target from the feature matrix."""
    if df is None or df.empty:
        return None, None, None

    drop_cols = [c for c in META_COLS + [target_col] if c in df.columns]
    meta = df[[c for c in META_COLS if c in df.columns]].copy()
    y = df[target_col].copy()
    X = df.drop(columns=drop_cols)
    return X, y, meta

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit_preprocessors(
    X_train: pd.DataFrame,
    artifact_dir: str = "../artifacts",
) -> dict:
    """
    Learn ALL transformation parameters EXCLUSIVELY from X_train.

    Saves a joblib bundle to `artifact_dir/preprocessors.joblib`. The default
    ("../artifacts") assumes the caller's cwd is code/ (true for playground.ipynb),
    landing the bundle at the repo-root artifacts/ directory, which .gitignore
    excludes from version control.
    Returns the bundle dict for immediate use.

    Column routing (decided in order, first match wins):
      1. Already binary (nunique <= 2)         → pass through untouched
      2. Is a multimodal base column           → fixed-threshold flag + RobustScaler
      3. Is a child of a multimodal base col   → RobustScaler only
      4. Is a diff/accel feature               → Yeo-Johnson
      5. Strictly positive, non-constant       → Box-Cox
      6. Fallback                              → Yeo-Johnson
    """
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    passthrough_cols = []
    multimodal_base_present = []
    multimodal_child_cols = []
    diff_accel_cols = []
    box_cox_cols = []
    yj_fallback_cols = []

    multimodal_thresholds: dict[str, float] = {}

    for col in numeric_cols:
        series = X_train[col].dropna()

        # Rule 1: binary or near-constant → pass through
        if series.nunique() <= 2:
            passthrough_cols.append(col)
            continue

        # Rule 2: multimodal base column with fixed threshold
        if col in MULTIMODAL_BASE_COLS:
            multimodal_thresholds[col] = MULTIMODAL_THRESHOLDS[col]
            multimodal_base_present.append(col)
            continue

        # Rule 3: child of a multimodal base (lag, roll, interaction but NOT diff/accel)
        parent = _multimodal_parent(col)
        if parent is not None and not _is_diff_or_accel(col):
            multimodal_child_cols.append(col)
            continue

        # Rule 4: diff / accel
        if _is_diff_or_accel(col):
            diff_accel_cols.append(col)
            continue

        # Rule 5: strictly positive → Box-Cox
        if _col_is_strictly_positive(series):
            box_cox_cols.append(col)
            continue

        # Rule 6: fallback → Yeo-Johnson
        yj_fallback_cols.append(col)

    logger.info(
        "Column routing — passthrough: %d, multimodal_base: %d, multimodal_children: %d, "
        "diff/accel (YJ): %d, box-cox: %d, yj-fallback: %d",
        len(passthrough_cols), len(multimodal_base_present), len(multimodal_child_cols),
        len(diff_accel_cols), len(box_cox_cols), len(yj_fallback_cols),
    )

    # ---- Fit RobustScaler for multimodal columns (base + children together) ----
    all_multimodal_scale_cols = multimodal_base_present + multimodal_child_cols
    multimodal_scaler = None
    if all_multimodal_scale_cols:
        multimodal_scaler = RobustScaler()
        multimodal_scaler.fit(X_train[all_multimodal_scale_cols])

    # ---- Fit Box-Cox transformer ----
    bc_transformer = None
    if box_cox_cols:
        bc_transformer = PowerTransformer(method="box-cox", standardize=True)
        bc_transformer.fit(X_train[box_cox_cols])
        logger.info("Box-Cox fitted on %d columns.", len(box_cox_cols))

    # ---- Fit Yeo-Johnson for diff/accel + other fallbacks ----
    all_yj_cols = diff_accel_cols + yj_fallback_cols
    yj_transformer = None
    if all_yj_cols:
        yj_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        yj_transformer.fit(X_train[all_yj_cols])
        logger.info(
            "Yeo-Johnson fitted on %d columns (%d diff/accel + %d other fallback).",
            len(all_yj_cols), len(diff_accel_cols), len(yj_fallback_cols),
        )

    bundle = {
        "passthrough_cols": passthrough_cols,
        "multimodal_base_cols": multimodal_base_present,
        "multimodal_thresholds": multimodal_thresholds,
        "multimodal_child_cols": multimodal_child_cols,
        "all_multimodal_scale_cols": all_multimodal_scale_cols,
        "box_cox_cols": box_cox_cols,
        "all_yj_cols": all_yj_cols,
        "multimodal_scaler": multimodal_scaler,
        "bc_transformer": bc_transformer,
        "yj_transformer": yj_transformer,
    }

    joblib.dump(bundle, Path(artifact_dir) / "preprocessors.joblib")
    logger.info("Preprocessing bundle saved to %s/preprocessors.joblib.", artifact_dir)
    return bundle

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform_with_preprocessors(
    X: pd.DataFrame,
    bundle: dict,
) -> pd.DataFrame:
    """
    Apply the bundle learned from training data to any matrix.
    Returns a new DataFrame with the same index and updated columns.
    """
    X_out = X.copy()

    # 1. Multimodal base columns: add flag, then scale
    for col in bundle["multimodal_base_cols"]:
        if col not in X_out.columns:
            continue
        threshold = bundle["multimodal_thresholds"][col]
        flag_col = f"{col}_is_high"
        X_out[flag_col] = (X_out[col] > threshold).astype(int)

    # 2. Scale all multimodal columns (base + children)
    scale_cols_present = [
        c for c in bundle["all_multimodal_scale_cols"] if c in X_out.columns
    ]
    if scale_cols_present and bundle["multimodal_scaler"] is not None:
        X_out[scale_cols_present] = bundle["multimodal_scaler"].transform(
            X_out[scale_cols_present]
        )

    # 3. Box-Cox
    bc_present = [c for c in bundle["box_cox_cols"] if c in X_out.columns]
    if bc_present and bundle["bc_transformer"] is not None:
        X_out[bc_present] = bundle["bc_transformer"].transform(X_out[bc_present])

    # 4. Yeo-Johnson
    yj_present = [c for c in bundle["all_yj_cols"] if c in X_out.columns]
    if yj_present and bundle["yj_transformer"] is not None:
        X_out[yj_present] = bundle["yj_transformer"].transform(X_out[yj_present])

    return X_out

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_ml_splits(
    df: pd.DataFrame,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    artifact_dir: str = "../artifacts",
    target_col: str = TARGET_COL,
) -> dict:
    """
    One-call entry point.

    Returns
    -------
    dict with keys:
        "train"  : (X_train_processed, y_train, meta_train)
        "val"    : (X_val_processed, y_val, meta_val)
        "test"   : (X_test_processed, y_test, meta_test)
        "bundle" : the fitted preprocessing bundle dict
    """
    train_df, val_df, test_df = chronological_session_split(df, train_pct, val_pct)

    X_train, y_train, meta_train = _separate(train_df, target_col)
    X_val, y_val, meta_val = _separate(val_df, target_col)
    X_test, y_test, meta_test = _separate(test_df, target_col)

    bundle = fit_preprocessors(X_train, artifact_dir=artifact_dir)

    def safe_transform(X, y, meta):
        if X is None:
            return None, None, None
        return transform_with_preprocessors(X, bundle), y, meta

    logger.info("Transforming X_train...")
    train_out = safe_transform(X_train, y_train, meta_train)

    logger.info("Transforming X_val...")
    val_out = safe_transform(X_val, y_val, meta_val)

    logger.info("Transforming X_test...")
    test_out = safe_transform(X_test, y_test, meta_test)

    return {
        "train": train_out,
        "val": val_out,
        "test": test_out,
        "bundle": bundle,
    }

def apply_preprocessing_live(
    X_live: pd.DataFrame,
    artifact_dir: str = "../artifacts",
) -> pd.DataFrame:
    """
    Load the saved bundle and apply it to live inference data.
    """
    bundle = joblib.load(Path(artifact_dir) / "preprocessors.joblib")
    return transform_with_preprocessors(X_live, bundle)

# ---------------------------------------------------------------------------
# Raw split helpers for unscaled feature-importance analysis
# ---------------------------------------------------------------------------

# Same split logic as chronological_session_split — kept as a distinct name
# since it's the conventional entry point for the raw (unscaled) analysis path.
chronological_session_split_raw = chronological_session_split


def separate_raw(df):
    """Like _separate, but also restricts X to numeric columns (for raw feature-importance analysis)."""
    X, y, meta = _separate(df)
    if X is not None:
        X = X.select_dtypes(include=[np.number]).copy()
    return X, y, meta