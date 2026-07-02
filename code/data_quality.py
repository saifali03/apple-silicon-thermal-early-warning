import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def trim_and_handle_gaps(
    df: pd.DataFrame, 
    expected_interval_s: int = 5, 
    tolerance_s: int = 2,
    max_fill_gap_s: int = 15
) -> pd.DataFrame:
    """
    Removes the first 5 and last 3 rows of each session to eliminate startup/shutdown noise.
    Identifies timestamp gaps within individual sessions.
    Flags large gaps and forward-fills minor micro-gaps.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    
    # Guarantee chronological order for cumulative and diff operations
    df = df.sort_values(by=["session_id", "timestamp_utc"]).reset_index(drop=True)
    
    # --- 1. Trim the Tails ---
    df["_row_num"] = df.groupby("session_id").cumcount()
    df["_row_num_rev"] = df.groupby("session_id").cumcount(ascending=False)
    
    df = df[(df["_row_num"] >= 5) & (df["_row_num_rev"] >= 3)]
    df = df.drop(columns=["_row_num", "_row_num_rev"]).reset_index(drop=True)
    
    logger.info("Trimmed the first 5 and last 3 rows from all sessions.")
    
    if df.empty:
        logger.warning("All rows were trimmed because sessions were too short.")
        return df
    
    # --- 2. Calculate Temporal Gaps ---
    df["delta_s"] = df.groupby("session_id")["timestamp_utc"].diff().dt.total_seconds()
    df["delta_s"] = df["delta_s"].fillna(expected_interval_s)
    
    # --- 3. Flag Significant Gaps ---
    gap_threshold = expected_interval_s + tolerance_s
    df["is_gap"] = (df["delta_s"] > gap_threshold).astype(int)
    
    total_gaps = df["is_gap"].sum()
    logger.info(f"Detected {total_gaps} temporal gaps exceeding {gap_threshold} seconds after trimming.")
    
    # --- 4. Forward-Fill Minor Gaps ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ["delta_s", "is_gap"]
    cols_to_fill = [c for c in numeric_cols if c not in cols_to_exclude]
    
    small_gap_mask = (df["delta_s"] > gap_threshold) & (df["delta_s"] <= max_fill_gap_s)
    
    num_small_gaps = small_gap_mask.sum()
    if num_small_gaps > 0:
        logger.info(f"Forward-filling missing numeric values for {num_small_gaps} minor gaps.")
        
        # Calculate full forward fill across sessions first
        ffilled_df = df.groupby("session_id")[cols_to_fill].ffill()
        
        # Conditionally apply it only to rows matching the mask
        df.loc[small_gap_mask, cols_to_fill] = ffilled_df.loc[small_gap_mask, cols_to_fill]
    # --- 5. Clean Up Temporary Columns ---
    # Drop the columns before returning so the final DataFrame is clean
    df = df.drop(columns=["delta_s", "is_gap"], errors="ignore")
    
    return df