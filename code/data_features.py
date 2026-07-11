import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# The 5 core telemetry signals judged most predictive from EDA (see data_eda.py's
# feature-importance diagnostics): both lag and rolling-window features are built
# from this same set.
CORE_VARS = [
    "cpu_die_temp_c",
    "cpu_power_mw",
    "combined_power_mw",
    "cpu_pcluster_active_pct",
    "cpu_pcluster_freq_mhz",
]

def engineer_features(df: pd.DataFrame, interval_seconds: int = 5) -> pd.DataFrame:
    """
    Phase 4: Feature Engineering
    Optimized for Pandas memory efficiency to prevent highly fragmented DataFrames.
    """
    if df.empty:
        logger.warning("engineer_features received an empty DataFrame.")
        return df.copy()

    df = df.copy()
    df = df.sort_values(["session_id", "timestamp_utc"]).reset_index(drop=True)

    logger.info("Engineering features grouped by session_id.")

    eps = 1e-6
    steps_30s = int(30 / interval_seconds)
    steps_1m  = int(60 / interval_seconds)
    steps_5m  = int(300 / interval_seconds)
    steps_15m = int(900 / interval_seconds)

    lag_steps = [1, 2, 3, 4, 5, 10]

    windows = [
        ("30s", steps_30s),
        ("1m", steps_1m),
        ("5m", steps_5m),
    ]

    # Initialize a dictionary to hold all new feature columns
    new_features = {}

    # 1. Lag features
    for col in CORE_VARS:
        if col in df.columns:
            g = df.groupby("session_id")[col]
            for lag in lag_steps:
                new_features[f"{col}_lag_{lag}"] = g.shift(lag)

    # 2. Rolling statistics
    for col in CORE_VARS:
        if col in df.columns:
            g = df.groupby("session_id")[col]

            for label, window in windows:
                roll = g.rolling(window=window, min_periods=max(3, window // 2))

                # Extract series directly to the dictionary
                new_features[f"{col}_roll_mean_{label}"] = roll.mean().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_std_{label}"]  = roll.std().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_max_{label}"]  = roll.max().reset_index(level=0, drop=True)
                new_features[f"{col}_roll_min_{label}"]  = roll.min().reset_index(level=0, drop=True)
                
                # Compute range from the dict entries
                new_features[f"{col}_roll_range_{label}"] = (
                    new_features[f"{col}_roll_max_{label}"] - new_features[f"{col}_roll_min_{label}"]
                )

    # Selective 15m heat-soak features
    for col in ["cpu_power_mw", "cpu_die_temp_c", "cpu_pcluster_freq_mhz"]:
        if col in df.columns:
            new_features[f"{col}_roll_mean_15m"] = (
                df.groupby("session_id")[col]
                  .rolling(window=steps_15m, min_periods=10)
                  .mean()
                  .reset_index(level=0, drop=True)
            )

    # 3. Rate of change / acceleration
    diff_vars = ["cpu_die_temp_c", "cpu_power_mw", "cpu_pcluster_freq_mhz"]

    for col in diff_vars:
        if col in df.columns:
            g = df.groupby("session_id")[col]
            new_features[f"{col}_diff_1"] = g.diff(1)
            new_features[f"{col}_diff_6"] = g.diff(steps_30s)
            
            # Group the newly created diff_1 by session to get acceleration
            new_features[f"{col}_accel"] = (
                new_features[f"{col}_diff_1"].groupby(df["session_id"]).diff(1)
            )

    # 4. Interaction / domain features
    if {"cpu_die_temp_c", "cpu_pcluster_freq_mhz"}.issubset(df.columns):
        new_features["thermal_distress_proxy"] = df["cpu_die_temp_c"] / (df["cpu_pcluster_freq_mhz"] + eps)

    if {"cpu_power_mw", "cpu_pcluster_active_pct"}.issubset(df.columns):
        new_features["pcluster_power_pressure"] = df["cpu_power_mw"] * df["cpu_pcluster_active_pct"]

    if {"ram_used_gb", "ram_total_gb"}.issubset(df.columns):
        new_features["ram_pressure_ratio"] = df["ram_used_gb"] / (df["ram_total_gb"] + eps)

    # Concatenate all new features in one shot (assigning columns one-by-one
    # would fragment the DataFrame and tank performance on wide feature sets).
    new_features_df = pd.DataFrame(new_features)
    df = pd.concat([df, new_features_df], axis=1)

    # Force a defragmentation copy just to be perfectly safe
    df = df.copy()

    logger.info("Feature engineering complete.")
    return df

def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop boundary rows created by lag/rolling operations.

    We anchor validity on:
    - the largest short-memory rolling feature used in core modelling
    - the largest lag used
    - the target label Y
    """
    if df.empty:
        return df.copy()

    # Now perfectly aligns with the generated features
    required_cols = [col for col in [
        "combined_power_mw_roll_mean_5m",
        "combined_power_mw_lag_10", 
        "Y"
    ] if col in df.columns]

    initial_rows = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    dropped = initial_rows - len(df)

    logger.info("Dropped %d boundary rows with invalid feature history.", dropped)

    return df

# ---------------------------------------------------------------------------
# Feature-family importance aggregation (used by playground.ipynb)
# ---------------------------------------------------------------------------

# Recognised transform suffixes produced by engineer_features(), longest first so
# multi-word transforms (e.g. "roll_mean") match before their prefixes ("roll").
_TRANSFORM_PATTERNS = [
    "roll_mean", "roll_std", "roll_max", "roll_min", "roll_range",
    "diff", "accel", "lag",
]

# Base telemetry signals features are derived from: CORE_VARS plus the extra
# parents used in interaction/domain features.
_BASE_SIGNALS = CORE_VARS + [
    "cpu_pcluster_active_pct",
    "ram_used_gb",
    "ram_total_gb",
]


def _signal_family(feature_name: str) -> str:
    """Return the base telemetry signal a feature is derived from."""
    for sig in sorted(_BASE_SIGNALS, key=len, reverse=True):
        if feature_name == sig or feature_name.startswith(sig + "_"):
            return sig
    return "other"


def _transform_family(feature_name: str) -> str:
    """Return the transform type of a feature (lag, roll_mean, diff, ...)."""
    for patt in _TRANSFORM_PATTERNS:
        if f"_{patt}_" in feature_name or feature_name.endswith(f"_{patt}"):
            return patt
    return "raw_or_interaction"


def aggregate_family_importance(
    importance: pd.Series,
    by: str = "signal",
) -> pd.DataFrame:
    """
    Aggregate per-feature importance scores into feature families.

    Parameters
    ----------
    importance : pd.Series
        Importance indexed by feature name (e.g. permutation importance).
    by : {"signal", "transform"}
        Group by base telemetry signal (default) or by transform type.

    Returns
    -------
    pd.DataFrame indexed by family, with total/mean importance and feature count,
    sorted by total importance descending.
    """
    if importance is None or len(importance) == 0:
        return pd.DataFrame(columns=["total_importance", "mean_importance", "n_features"])

    classifier = _signal_family if by == "signal" else _transform_family
    families = importance.index.to_series().map(classifier)

    grouped = importance.groupby(families)
    out = pd.DataFrame({
        "total_importance": grouped.sum(),
        "mean_importance": grouped.mean(),
        "n_features": grouped.size(),
    }).sort_values("total_importance", ascending=False)

    return out
