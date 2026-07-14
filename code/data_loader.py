import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_and_merge_sessions(raw_data_dir: str | Path) -> pd.DataFrame:
    """
    Recursively loads all CSVs from the raw data directory, assigns a unique 
    session_id based on the filename, and merges them into a single chronological DataFrame.
    """
    data_path = Path(raw_data_dir)
    csv_files = sorted(data_path.rglob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path.resolve()}")
        
    frames = []
    
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            
            # The filename format is {machine_id}_{date}_{run_number}.csv
            # e.g., mac_m5_20260625_001.csv
            filename_stem = filepath.stem
            
            # Use the filename as the definitive session_id
            df["session_id"] = filename_stem
            
            # Convert timestamp immediately to ensure proper sorting later
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            
            frames.append(df)
            logger.info(f"Loaded {filename_stem} with {len(df)} rows.")
            
        except Exception as e:
            logger.error(f"Failed to load {filepath.name}: {e}")
            
    # Concatenate all sessions
    combined_df = pd.concat(frames, ignore_index=True)
    
    # Sort by session, then strictly by time to ensure chronological integrity
    combined_df = combined_df.sort_values(by=["session_id", "timestamp_utc"]).reset_index(drop=True)
    
    logger.info(f"Successfully merged {len(frames)} sessions. Total rows: {len(combined_df)}")
    return combined_df