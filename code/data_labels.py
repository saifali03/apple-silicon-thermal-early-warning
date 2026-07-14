import pandas as pd
import logging

logger = logging.getLogger(__name__)


def construct_labels(
    df: pd.DataFrame,
    lookahead_minutes: float = 2.0,
    interval_seconds: int = 5,
    temp_threshold_c: float = 75.0,
) -> pd.DataFrame:
    """
    Construct binary look-ahead label Y for thermal event prediction.

    Y_t = 1 if within the window [t, t + delta_t]:
      - cpu_die_temp_c  >= temp_threshold_c  (where non-NaN)

    Rows where the full look-ahead window is unavailable (session tails,
    very short sessions) are dropped. Sleeping rows (code == -1) are
    removed before labelling.

    Parameters
    ----------
    df                   : must have columns session_id, timestamp_utc,
                           cpu_die_temp_c, thermal_pressure_code
    lookahead_minutes    : prediction horizon in minutes
    interval_seconds     : logging cadence in seconds
    temp_threshold_c     : CPU temperature threshold for class 1
    """
    if df.empty:
        logger.warning("construct_labels received an empty DataFrame.")
        return df.copy()

    df = df.copy()

    #Dtype safety
    df["thermal_pressure_code"] = pd.to_numeric(df["thermal_pressure_code"], errors="coerce")
    df["cpu_die_temp_c"]        = pd.to_numeric(df["cpu_die_temp_c"],        errors="coerce")

    #Remove sleeping rows (code == -1): no thermal signal
    n_sleeping = (df["thermal_pressure_code"] == -1).sum()
    if n_sleeping > 0:
        logger.info("Removing %d sleeping rows (thermal_pressure_code == -1).", n_sleeping)
        df = df[df["thermal_pressure_code"] != -1].copy()

    if df.empty:
        logger.warning("DataFrame is empty after removing sleeping rows.")
        return df

    #Chronological order
    df = df.sort_values(["session_id", "timestamp_utc"]).reset_index(drop=True)

    #Horizon arithmetic
    horizon_steps = int((lookahead_minutes * 60) / interval_seconds)
    window_size   = horizon_steps + 1   # includes current row t

    logger.info(
        "Constructing labels. Horizon: %.1f min = %d steps (window_size=%d).",
        lookahead_minutes, horizon_steps, window_size,
    )

    #Forward-looking maximum
    # min_periods=window_size: emit NaN instead of a partial-window value
    # for tail rows. Those NaN rows are then dropped cleanly below.
    df["_future_max_temp"] = df.groupby("session_id")["cpu_die_temp_c"].transform(
        lambda x: x[::-1].rolling(window=window_size, min_periods=window_size).max()[::-1]
    )

    #Drop tail rows (incomplete look-ahead window → NaN in future max temp)
    initial_rows = len(df)
    df = df.dropna(subset=["_future_max_temp"]).copy()
    rows_dropped = initial_rows - len(df)
    logger.info("Dropped %d tail rows with incomplete look-ahead windows.", rows_dropped)

    if df.empty:
        logger.warning("No rows remained after dropping incomplete look-ahead windows.")
        return df

    #Binary label
    df["Y"] = (df["_future_max_temp"] >= temp_threshold_c).astype(int)

    df = df.drop(columns=["_future_max_temp"])

    positive_pct = df["Y"].mean() * 100
    logger.info(
        "Label construction complete. Class 1: %.2f%% (%d / %d rows).",
        positive_pct, int(df["Y"].sum()), len(df),
    )

    return df
