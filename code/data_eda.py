import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

def plot_univariate_distributions(df: pd.DataFrame, target_col: str = "Y"):
    """
    Plots KDE distributions for key thermal and power metrics.
    Overlays the target class to check for feature separability.
    """
    features = ["ram_used_gb", "combined_power_mw", "loadavg_1m", "cpu_total_active_pct"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if feature in df.columns:
            # Overlapping distributions based on whether the future temp crosses 80C
            sns.kdeplot(data=df, x=feature, hue=target_col, fill=True, common_norm=False, ax=axes[i], alpha=0.5)
            axes[i].set_title(f"Distribution of {feature} by Future Critical Target")
            axes[i].set_xlabel(feature)
            
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Generates a correlation heatmap.
    Flags feature pairs with > 0.85 correlation to warn about multicollinearity.
    """
    # Select only numeric, non-boolean columns
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=["is_gap"], errors="ignore")
    
    # Drop columns that are entirely NaN or constant
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    # remove the is outlier cols
    is_outlier_cols = [col for col in df.columns if col.endswith("_is_outlier")]
    numeric_df = numeric_df.drop(columns=is_outlier_cols, errors="ignore")
    corr = numeric_df.corr()
    
    # Mask the upper triangle for readability
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Telemetry Feature Correlation Heatmap")
    plt.show()
    # Print warnings for highly correlated features
    print("\n--- Multicollinearity Warnings (r > 0.85) ---")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [column for column in upper.columns if any(upper[column].abs() > 0.85)]
    for col in high_corr:
        correlated_with = upper.index[upper[col].abs() > 0.85].tolist()
        print(f"Feature '{col}' is highly correlated with {correlated_with}. Consider dropping for linear models.")
    # Also print, in descending order, the top 10 most correlated features to target_cpu_critical
    if "Y" in corr.columns:
        target_corr = corr["Y"].drop(["Y", "cpu_die_temp_c"]).sort_values(ascending=False)
        print("\n--- Top 10 Positively Correlated Features with Target ---")
        print(target_corr.head(10))

def test_heat_soak_assumption(df: pd.DataFrame):
    """
    Tests if the relationship between power and temperature changes over the length of a session.
    """
    df = df.copy()
    
    # Calculate minutes since the start of each session
    df["session_start_time"] = df.groupby("session_id")["timestamp_utc"].transform("min")
    df["minutes_since_start"] = (df["timestamp_utc"] - df["session_start_time"]).dt.total_seconds() / 60.0
    
    plt.figure(figsize=(10, 6))
    # Scatter plot of Power vs Temp, colored by how long the session has been running
    scatter = plt.scatter(df["combined_power_mw"], df["cpu_die_temp_c"], 
                          c=df["minutes_since_start"], cmap="viridis", alpha=0.3, s=10)
    
    plt.colorbar(scatter, label="Minutes Since Session Start")
    plt.title("Power vs. Temperature: The Heat Soak Effect")
    plt.xlabel("Combined Power (mW)")
    plt.ylabel("CPU Die Temp (°C)")
    plt.show()

def plot_power_temp_crosscorr_eda(df: pd.DataFrame, power_col: str = "combined_power_mw", temp_col: str = "cpu_die_temp_c", lag_limit: int = 60):
    """
    Uses Cross-Correlation to find the physical delay between a power spike and a temperature spike.
    Helps empirically justify the chosen Delta t.
    """
    # Pick the longest continuous session for this test
    longest_session = df["session_id"].value_counts().idxmax()
    session_data = df[df["session_id"] == longest_session].copy().dropna(subset=[power_col, temp_col])
    
    # Calculate cross correlation
    cross_corr = ccf(session_data[power_col], session_data[temp_col], adjusted=False)[:lag_limit]
    
    plt.figure(figsize=(10, 4))
    plt.stem(range(lag_limit), cross_corr)
    plt.title(f"Cross-Correlation: How long before a Power spike hits Temperature?\n(Session: {longest_session})")
    plt.xlabel("Lag (Number of 5-second intervals)")
    plt.ylabel("Correlation Coefficient")
    plt.axhline(0, color="black", linestyle="--")
    plt.show()
    
    max_lag_idx = np.argmax(cross_corr)
    seconds_delay = max_lag_idx * 5  # assuming 5 second interval
    print(f"\nMaximum correlation occurs at a lag of {max_lag_idx} intervals ({seconds_delay} seconds).")


def plot_autocorrelation(df: pd.DataFrame, target_col: str = "cpu_die_temp_c", lags: int = 30):
    """
    Plots ACF and PACF for a single representative session.
    Used to determine how many lag features to engineer.
    """
    # Isolate a single long session so we don't calculate ACF across session boundaries
    longest_session = df["session_id"].value_counts().idxmax()
    session_data = df[df["session_id"] == longest_session][target_col].dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ACF: Overall correlation with past (includes indirect effects)
    plot_acf(session_data, lags=lags, ax=axes[0], title=f"Autocorrelation (ACF)\n(Session: {longest_session})")
    
    # PACF: Direct correlation with past (removes intermediate effects)
    plot_pacf(session_data, lags=lags, ax=axes[1], title=f"Partial Autocorrelation (PACF)\nDirect Lag Signal")
    
    plt.tight_layout()
    plt.show()
    
    print("\n💡 INTERPRETATION:")
    print("Look at the PACF plot (right). The point where the stems fall inside the blue shaded area")
    print("is where historical data stops providing *new* direct information. Use this to set your max lag features.")

def plot_annotated_timeline(df: pd.DataFrame, session_idx: int = 0, target_col: str = "Y"):
    """
    Plots Temperature and Power over time for a specific session, 
    highlighting the regions where the future target crosses the critical threshold.
    """
    # Pick a session (default is the first one, or pass an index)
    session_id = df["session_id"].unique()[session_idx]
    session_data = df[df["session_id"] == session_id].copy()
    
    # Create a time axis in minutes for readability
    session_data["minutes"] = (session_data["timestamp_utc"] - session_data["timestamp_utc"].min()).dt.total_seconds() / 60
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot Temperature
    color1 = 'tab:red'
    ax1.set_xlabel('Time (Minutes)')
    ax1.set_ylabel('CPU Die Temp (°C)', color=color1)
    ax1.plot(session_data["minutes"], session_data["cpu_die_temp_c"], color=color1, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(80, color='darkred', linestyle='--', alpha=0.5, label="Critical Threshold (80°C)")
    
    # Plot Power on secondary Y axis
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Combined Power (mW)', color=color2)
    ax2.plot(session_data["minutes"], session_data["combined_power_mw"], color=color2, alpha=0.4)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Highlight the target zones (where future temp is critical)
    if target_col in session_data.columns:
        critical_zones = session_data[session_data[target_col] == 1]
        ax1.scatter(critical_zones["minutes"], critical_zones["cpu_die_temp_c"],
                    color='black', s=10, zorder=5, label="Target Window Active")
    
    fig.tight_layout()
    ax1.legend(loc="upper left")
    plt.title(f"Telemetry Timeline with Target Annotations (Session: {session_id})")
    plt.show()

def plot_throttling_regime(df: pd.DataFrame):
    """
    Examines hardware self-preservation. Plots CPU Temperature vs P-Cluster Frequency.
    """
    if "cpu_pcluster_freq_mhz" not in df.columns or "cpu_die_temp_c" not in df.columns:
        logger.warning("Missing columns for throttling regime plot.")
        return
        
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="cpu_die_temp_c", y="cpu_pcluster_freq_mhz", 
                    hue="thermal_pressure_code", palette="coolwarm", alpha=0.6, s=15)
    
    plt.title("Hardware Throttling Regime: Temperature vs. P-Cluster Frequency")
    plt.xlabel("CPU Die Temp (°C)")
    plt.ylabel("P-Cluster Frequency (MHz)")
    plt.axvline(80, color='black', linestyle='--', alpha=0.5, label="80°C Boundary")
    plt.legend(title="Thermal Pressure Code")
    plt.show()

def calculate_baseline_target_importance(df: pd.DataFrame, target_col: str = "Y"):
    """
    Calculates a quick Mutual Information score to see which raw features 
    already contain strong signals for the target before we engineer lags.
    """
    print("\n--- Baseline Target Signal (Mutual Information) ---")
    
    # Drop NaNs and metadata columns
    clean_df = df.dropna().copy()
    features = clean_df.select_dtypes(include=[np.number]).drop(columns=[target_col, "is_gap", "cpu_die_temp_c"], errors="ignore")
    
    # Remove constant columns
    features = features.loc[:, features.nunique() > 1]
    
    # Calculate MI scores
    mi_scores = mutual_info_classif(features, clean_df[target_col], random_state=42)
    mi_series = pd.Series(mi_scores, index=features.columns).sort_values(ascending=False)
    
    # Display top 10
    for feat, score in mi_series.head(10).items():
        print(f"{feat:<30} {score:.4f}")
    
    print("\n💡 Note: Features with high MI scores are your strongest baseline predictors.")


def plot_numeric_kdes(df: pd.DataFrame, plots_per_row: int = 3) -> None:
    """
    Identifies numeric, non-binary columns in a DataFrame and plots their 
    KDE distributions in a grid with a fixed number of plots per row.
    """
    if df.empty:
        logger.warning("The provided DataFrame is empty. Skipping plots.")
        return

    # --- 1. Filter for Numeric, Non-Binary Columns ---
    target_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check number of unique non-NaN values to exclude binary/constant columns
            unique_count = df[col].nunique(dropna=True)
            if unique_count > 2:
                target_cols.append(col)
            else:
                logger.debug(f"Skipping column '{col}': Only {unique_count} unique value(s).")

    total_plots = len(target_cols)
    
    if total_plots == 0:
        logger.warning("No numeric, non-binary columns found in the DataFrame.")
        return

    # --- 2. Calculate Grid Layout ---
    n_cols = plots_per_row
    n_rows = math.ceil(total_plots / n_cols)
    
    # Adjust overall figure size dynamically based on row count
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # Flatten axes array for straightforward 1D indexing, regardless of grid size
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    logger.info(f"Generating KDE plots for {total_plots} variables across a {n_rows}x{n_cols} grid.")

    # --- 3. Plot Each Variable ---
    for idx, col in enumerate(target_cols):
        ax = axes[idx]
        
        # Plot KDE. dropna() ensures seaborn handles missing values cleanly.
        sns.kdeplot(data=df[col].dropna(), ax=ax, fill=True, color="skyblue", alpha=0.6)
        
        ax.set_title(f"KDE of {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # --- 4. Clean up Unused Subplots ---
    # If total_plots doesn't cleanly divide by plots_per_row, hide the empty remainder axes
    for idx in range(total_plots, len(axes)):
        fig.delaxes(axes[idx])

    # Tight layout prevents axis titles and labels from overlapping
    plt.tight_layout()
    plt.show()

def plot_variable_pairplot(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """
    Generates a Seaborn pairplot for specified columns or a core set of default columns.
    Safely ignores requested columns that do not exist in the provided DataFrame.
    """
    if df.empty:
        logger.warning("The provided DataFrame is empty. Skipping pairplot.")
        return

    # --- 1. Define Core Defaults ---
    default_cols = [
        "mem_compressed_gb",
        "cpu_pcluster_freq_mhz",
        "cpu_power_mw",
        "loadavg_1m",
        "cpu_pcluster_active_pct",
        "cpu_total_active_pct",
        "combined_power_mw",
        "cpu_power_mw_is_outlier",
        "cpu_ecluster_freq_mhz"
    ]

    # --- 2. Handle Fallback Logic & Notification ---
    if columns is None:
        print(
            "Notice: No specific columns provided. The pairplot defaults to:\n"
            "  - mem_compressed_gb\n"
            "  - cpu_pcluster_freq_mhz\n"
            "  - cpu_power_mw\n"
            "  - loadavg_1m\n"
            "  - cpu_pcluster_active_pct\n"
            "  - cpu_total_active_pct\n"
            "  - combined_power_mw\n"
            "  - cpu_power_mw_is_outlier\n"
            "  - cpu_ecluster_freq_mhz\n"
        )
        target_cols = default_cols
    else:
        target_cols = columns

    # --- 3. Validate Column Existence ---
    existing_cols = [c for c in target_cols if c in df.columns]
    missing_cols = [c for c in target_cols if c not in df.columns]

    if missing_cols:
        logger.warning(f"The following requested columns were missing from the DataFrame and will be skipped: {missing_cols}")

    if not existing_cols:
        logger.error("None of the requested or default columns exist in the provided DataFrame. Aborting plot.")
        return

    # --- 4. Render Pairplot ---
    logger.info(f"Generating pairplot for {len(existing_cols)} features...")
    
    sns.pairplot(
        df[existing_cols],
        diag_kind="kde",
    )
    
    plt.show()

    # add a function to print descriptive statistics for the numeric non binary columns
def print_numeric_descriptive_stats(df: pd.DataFrame) -> None:
    """
    Prints descriptive statistics for numeric, non-binary columns in the DataFrame.
    Excludes columns with only 1 or 2 unique values (binary/constant).
    """
    if df.empty:
        logger.warning("The provided DataFrame is empty. No statistics to display.")
        return

    # Filter for numeric, non-binary columns
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if df[col].nunique(dropna=True) > 2]

    if not numeric_cols:
        logger.warning("No numeric, non-binary columns found in the DataFrame.")
        return

    stats_df = df[numeric_cols].describe().T  # Transpose for better readability
    stats_df["missing_count"] = df[numeric_cols].isna().sum()
    stats_df["missing_pct"] = (stats_df["missing_count"] / len(df)) * 100

    print("\n--- Descriptive Statistics for Numeric Non-Binary Columns ---")
    display(stats_df[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_count", "missing_pct"]])


# ---------------------------------------------------------------------------
# Feature-importance diagnostics (used to decide which engineered features
# from data_features.py are actually worth keeping)
# ---------------------------------------------------------------------------

def fit_tree_importance(X_train, y_train, feature_names):
    """Fits a RandomForest and a GradientBoosting classifier, returning both
    models plus their built-in feature importances for a quick side-by-side."""
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    gb_imp = pd.Series(gb.feature_importances_, index=feature_names).sort_values(ascending=False)

    return rf, gb, rf_imp, gb_imp


def get_permutation_importance(model, X_val, y_val, feature_names, scoring="f1"):
    """More trustworthy than built-in importances: measures the actual score
    drop from shuffling each feature, on held-out data."""
    result = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=20,
        random_state=42,
        scoring=scoring,
        n_jobs=-1
    )
    imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
    std = pd.Series(result.importances_std, index=feature_names).loc[imp.index]
    return imp, std


def feature_family(name):
    """Buckets an engineered feature name by which data_features.py step produced it."""
    if "_lag_" in name:
        return "lag"
    if "_roll_" in name:
        return "rolling"
    if name.endswith("_diff_1") or name.endswith("_diff_6"):
        return "diff"
    if name.endswith("_accel"):
        return "accel"
    if name in ["pcluster_power_pressure", "thermal_distress_proxy"]:
        return "interaction"
    if name in [
        "cpu_die_temp_c", "cpu_power_mw", "combined_power_mw",
        "cpu_pcluster_active_pct", "cpu_pcluster_freq_mhz"
    ]:
        return "raw_core"
    return "other"


def aggregate_family_importance(importance_series):
    """Rolls up a per-feature importance Series (e.g. from get_permutation_importance)
    into total importance per feature_family, to see which engineering step paid off."""
    df = importance_series.reset_index()
    df.columns = ["feature", "importance"]
    df["family"] = df["feature"].apply(feature_family)
    return df.groupby("family")["importance"].sum().sort_values(ascending=False)