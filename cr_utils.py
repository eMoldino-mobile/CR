import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Constants ---
PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77'
}

# --- Utility Functions ---
def format_duration(seconds):
    if pd.isna(seconds) or seconds < 0: return "N/A"
    total_minutes = int(seconds / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours}h {minutes}m"

@st.cache_data
def load_data(file):
    """Loads data, similar to the Run Rate loader but looks for CR specific cols."""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Column Normalization
        col_map = {col.strip().upper(): col for col in df.columns}
        def get_col(target): return col_map.get(target)

        # Mapping specific to Capacity Risk
        mappings = {
            "SHOT TIME": get_col("SHOT TIME") or get_col("TIMESTAMP") or get_col("DATE"),
            "ACTUAL CT": get_col("ACTUAL CT") or get_col("CYCLE TIME"),
            "APPROVED CT": get_col("APPROVED CT") or get_col("STD CT") or get_col("OPTIMAL CT"),
            "WORKING CAVITIES": get_col("WORKING CAVITIES") or get_col("CAVITIES"),
            "TOOL_ID": get_col("TOOLING ID") or get_col("EQUIPMENT CODE") or get_col("TOOL_ID")
        }
        
        # Rename found columns
        df = df.rename(columns={v: k for k, v in mappings.items() if v})
        
        # Data Type Conversion
        if "SHOT TIME" in df.columns:
            df["SHOT TIME"] = pd.to_datetime(df["SHOT TIME"], errors='coerce')
        
        # Ensure numeric
        for col in ["ACTUAL CT", "APPROVED CT", "WORKING CAVITIES"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return pd.DataFrame()

# ==============================================================================
# --- CORE LOGIC: RUN RATE CALCULATOR (Identical to Run Rate App) ---
# ==============================================================================
class RunRateCalculator:
    """
    The foundational logic. Identifies Runs, Stops, and Total Runtime.
    """
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, analysis_mode='aggregate'):
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.analysis_mode = analysis_mode
        self.results = self._calculate_all_metrics()

    def _calculate_all_metrics(self) -> dict:
        df = self.df_raw.copy()
        if "SHOT TIME" not in df.columns or df.empty: return {}
        
        df = df.dropna(subset=["SHOT TIME"]).sort_values("SHOT TIME").reset_index(drop=True)
        
        # Ensure columns exist (default to 0 if missing, though CR usually needs them)
        if "ACTUAL CT" not in df.columns: df["ACTUAL CT"] = 0
        else: df["ACTUAL CT"] = df["ACTUAL CT"].fillna(0)

        df["time_diff_sec"] = df["SHOT TIME"].diff().dt.total_seconds().fillna(0)
        
        # 1. Mode CT Calculation
        if not df.empty:
             mode_ct = df[df["ACTUAL CT"] < 999.9]['ACTUAL CT'].mode().max() if not df[df["ACTUAL CT"] < 999.9].empty else 0
        else: mode_ct = 0
        
        lower_limit = mode_ct * (1 - self.tolerance)
        upper_limit = mode_ct * (1 + self.tolerance)

        # 2. Stop Logic
        df['next_shot_time_diff'] = df['time_diff_sec'].shift(-1).fillna(0)
        is_hard_stop = df["ACTUAL CT"] >= 999.9
        is_abnormal = ((df["ACTUAL CT"] < lower_limit) | (df["ACTUAL CT"] > upper_limit)) & ~is_hard_stop
        is_gap = df["next_shot_time_diff"] > (df["ACTUAL CT"] + self.downtime_gap_tolerance)
        
        df["stop_flag"] = np.where(is_abnormal | is_gap | is_hard_stop, 1, 0)
        df["stop_event"] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # 3. Time Attribution (The "Downtime Plug")
        df['adj_ct_sec'] = df['ACTUAL CT']
        df.loc[is_gap, 'adj_ct_sec'] = df['next_shot_time_diff']
        
        # Basic sums (refined later by Run Splitter)
        production_time_sec = df.loc[df['stop_flag'] == 0, 'ACTUAL CT'].sum()
        
        return {
            "processed_df": df, 
            "production_time_sec": production_time_sec,
            "mode_ct": mode_ct
        }

# ==============================================================================
# --- EXTENSION: CAPACITY CALCULATOR (Builds on top of Run Rate) ---
# ==============================================================================

def calculate_capacity_metrics(df_tool, tolerance, gap, run_interval_hours, default_cavities=1):
    """
    1. Runs the RunRateCalculator to identify Stops.
    2. Splits data into Runs based on Interval.
    3. Calculates Capacity Loss (Parts) based on Approved CT.
    """
    # 1. Base Run Rate Logic
    base_calc = RunRateCalculator(df_tool, tolerance, gap)
    df = base_calc.results.get('processed_df')
    
    if df is None or df.empty: return {}

    # Fill missing cols if needed
    if "APPROVED CT" not in df.columns: df["APPROVED CT"] = df['ACTUAL CT'].mode().max() # Fallback
    if "WORKING CAVITIES" not in df.columns: df["WORKING CAVITIES"] = default_cavities

    # 2. Run Identification (The "Run-Based" Philosophy)
    RUN_INTERVAL_SEC = run_interval_hours * 3600
    is_new_run = df['time_diff_sec'] > RUN_INTERVAL_SEC
    df['run_id'] = is_new_run.cumsum()
    
    # 3. Iterate Runs and Aggregate
    # We do this to get accurate "Total Runtime" (excluding 8h+ gaps)
    
    total_metrics = {
        "Total Run Duration (sec)": 0,
        "Production Time (sec)": 0,
        "Downtime (sec)": 0,
        "Optimal Output (parts)": 0,
        "Actual Output (parts)": 0,
        "Loss - Downtime (parts)": 0,
        "Loss - Speed (parts)": 0,
        "Gain - Speed (parts)": 0,
        "Total Shots": len(df)
    }
    
    for _, df_run in df.groupby('run_id'):
        # A. Time Calculations (Per Run)
        first_shot = df_run['SHOT TIME'].min()
        last_shot = df_run['SHOT TIME'].max()
        last_ct = df_run.iloc[-1]['ACTUAL CT']
        
        run_duration = (last_shot - first_shot).total_seconds() + last_ct
        
        # Production time = Sum of Actual CT of "Good" shots (stop_flag=0)
        # Note: RunRateCalculator already flagged these.
        prod_time = df_run[df_run['stop_flag'] == 0]['ACTUAL CT'].sum()
        downtime = run_duration - prod_time
        
        # B. Capacity Calculations (Per Run)
        # Using weighted average Approved CT for the run in case it changes
        avg_approved_ct = df_run['APPROVED CT'].mean()
        if avg_approved_ct == 0: avg_approved_ct = 1
        
        max_cavities = df_run['WORKING CAVITIES'].max() # Assuming constant per run
        
        # Optimal Output = If we ran the whole duration at Approved CT
        optimal_parts = (run_duration / avg_approved_ct) * max_cavities
        
        # Actual Output = Sum of cavities per shot
        actual_parts = df_run['WORKING CAVITIES'].sum()
        
        # C. Loss Attribution
        # 1. How many parts did we lose because the machine stopped?
        #    (Downtime Duration / Approved CT)
        loss_downtime_parts = (downtime / avg_approved_ct) * max_cavities
        
        # 2. Speed Loss/Gain
        #    Calculated as a plug or shot-by-shot. Let's do shot-by-shot for precision.
        #    For every production shot: (Actual CT - Approved CT) is lost time.
        prod_shots = df_run[df_run['stop_flag'] == 0].copy()
        prod_shots['time_delta'] = prod_shots['ACTUAL CT'] - prod_shots['APPROVED CT']
        
        # If Time Delta > 0 (Slower than approved) -> Loss
        # If Time Delta < 0 (Faster than approved) -> Gain
        
        loss_speed_sec = prod_shots[prod_shots['time_delta'] > 0]['time_delta'].sum()
        gain_speed_sec = abs(prod_shots[prod_shots['time_delta'] < 0]['time_delta'].sum())
        
        loss_speed_parts = (loss_speed_sec / avg_approved_ct) * max_cavities
        gain_speed_parts = (gain_speed_sec / avg_approved_ct) * max_cavities
        
        # Aggregate
        total_metrics["Total Run Duration (sec)"] += run_duration
        total_metrics["Production Time (sec)"] += prod_time
        total_metrics["Downtime (sec)"] += downtime
        total_metrics["Optimal Output (parts)"] += optimal_parts
        total_metrics["Actual Output (parts)"] += actual_parts
        total_metrics["Loss - Downtime (parts)"] += loss_downtime_parts
        total_metrics["Loss - Speed (parts)"] += loss_speed_parts
        total_metrics["Gain - Speed (parts)"] += gain_speed_parts

    return total_metrics