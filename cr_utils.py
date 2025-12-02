import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from io import BytesIO

# ==============================================================================
# --- CONSTANTS & SHARED FUNCTIONS ---
# ==============================================================================

PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77',
    'blue': '#3498DB',
    'grey': '#808080',
    'target_line': 'deepskyblue',
    'optimal_line': 'darkblue',
    'purple': '#8A2BE2'
}

def format_seconds_to_dhm(total_seconds):
    """Converts total seconds into a 'Xd Yh Zm' or 'Xs' string."""
    if pd.isna(total_seconds) or total_seconds < 0: return "N/A"
    
    if total_seconds < 60:
         return f"{total_seconds:.1f}s"

    total_minutes = int(total_seconds / 60)
    days = total_minutes // (60 * 24)
    remaining_minutes = total_minutes % (60 * 24)
    hours = remaining_minutes // 60
    minutes = remaining_minutes % 60
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if minutes > 0 or not parts: parts.append(f"{minutes}m")
    return " ".join(parts) if parts else "0m"

def load_all_data_cr(files):
    """Loads and standardizes data."""
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file) if file.name.endswith(('.xls', '.xlsx')) else pd.read_csv(file)
            col_map = {col.strip().upper(): col for col in df.columns}
            def get_col(target_list):
                for t in target_list:
                    if t in col_map: return col_map[t]
                return None

            tool_col = get_col(["TOOLING ID", "EQUIPMENT CODE", "TOOL_ID", "TOOL"])
            if tool_col: df.rename(columns={tool_col: "tool_id"}, inplace=True)

            time_col = get_col(["SHOT TIME", "TIMESTAMP", "DATE", "TIME"])
            if time_col: df.rename(columns={time_col: "shot_time"}, inplace=True)

            act_ct_col = get_col(["ACTUAL CT", "ACTUAL_CT", "CYCLE TIME"])
            if act_ct_col: df.rename(columns={act_ct_col: "actual_ct"}, inplace=True)

            app_ct_col = get_col(["APPROVED CT", "APPROVED_CT", "STD CT"])
            if app_ct_col: df.rename(columns={app_ct_col: "approved_ct"}, inplace=True)

            cav_col = get_col(["WORKING CAVITIES", "CAVITIES"])
            if cav_col: df.rename(columns={cav_col: "working_cavities"}, inplace=True)
            
            area_col = get_col(["PLANT AREA", "AREA"])
            if area_col: df.rename(columns={area_col: "plant_area"}, inplace=True)

            if "shot_time" in df.columns and "actual_ct" in df.columns:
                df["shot_time"] = pd.to_datetime(df["shot_time"], errors="coerce")
                df["actual_ct"] = pd.to_numeric(df["actual_ct"], errors="coerce")
                df.dropna(subset=["shot_time", "actual_ct"], inplace=True)
                df_list.append(df)
        except Exception: continue
    
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# ==============================================================================
# --- CORE CALCULATION ENGINE ---
# ==============================================================================

class CapacityRiskCalculator:
    def __init__(self, df: pd.DataFrame, tolerance: float, downtime_gap_tolerance: float, 
                 run_interval_hours: float, target_output_perc: float = 100.0, 
                 default_cavities: int = 1, remove_maintenance: bool = False):
        
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.run_interval_hours = run_interval_hours
        self.target_output_perc = target_output_perc
        self.default_cavities = default_cavities
        self.remove_maintenance = remove_maintenance
        self.results = self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        df = self.df_raw.copy()
        if df.empty: return {}

        if self.remove_maintenance and 'plant_area' in df.columns:
            df = df[~df['plant_area'].astype(str).str.lower().isin(['maintenance', 'warehouse'])].copy()
            if df.empty: return {}

        if 'approved_ct' not in df.columns: df['approved_ct'] = df['actual_ct'].median() 
        if 'working_cavities' not in df.columns: df['working_cavities'] = self.default_cavities
        
        df['approved_ct'] = pd.to_numeric(df['approved_ct'], errors='coerce').fillna(1)
        df['working_cavities'] = pd.to_numeric(df['working_cavities'], errors='coerce').fillna(self.default_cavities)
        
        # Ensure positive Approved CT
        df.loc[df['approved_ct'] <= 0, 'approved_ct'] = 1
        
        df = df.sort_values("shot_time").reset_index(drop=True)

        # 1. Run Identification (Run Rate Logic)
        df['time_diff_sec'] = df['shot_time'].diff().dt.total_seconds().fillna(0)
        if len(df) > 0: df.loc[0, 'time_diff_sec'] = df.loc[0, 'actual_ct']

        is_new_run = df['time_diff_sec'] > (self.run_interval_hours * 3600)
        df['run_id'] = is_new_run.cumsum()

        # 2. Mode CT & Limits (Per Run)
        run_modes = df[df['actual_ct'] < 1000].groupby('run_id')['actual_ct'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()
        )
        df['mode_ct'] = df['run_id'].map(run_modes)
        lower_limit = df['mode_ct'] * (1 - self.tolerance)
        upper_limit = df['mode_ct'] * (1 + self.tolerance)
        
        df['mode_lower'] = lower_limit
        df['mode_upper'] = upper_limit

        # 3. Approved CT (Per Run)
        run_approved_cts = df.groupby('run_id')['approved_ct'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else 1
        )
        df['approved_ct_for_run'] = df['run_id'].map(run_approved_cts)
        
        # 4. Stop Detection (MATCHING RUN RATE LOGIC)
        df['next_shot_time_diff'] = df['time_diff_sec'].shift(-1).fillna(0)
        
        is_time_gap = df['next_shot_time_diff'] > (df['actual_ct'] + self.downtime_gap_tolerance)
        is_abnormal = ((df['actual_ct'] < lower_limit) | (df['actual_ct'] > upper_limit))
        is_hard_stop = df['actual_ct'] >= 999.9

        df['stop_flag'] = np.where(is_time_gap | is_abnormal | is_hard_stop, 1, 0)
        
        if not df.empty:
            df.loc[0, 'stop_flag'] = 0
        
        df['stop_event'] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        df['adj_ct_sec'] = df['actual_ct']
        df.loc[is_time_gap, 'adj_ct_sec'] = df['next_shot_time_diff']
        
        # --- Metrics Calculation ---
        
        # Total Run Duration (Wall Clock)
        run_durations = []
        for _, run_df in df.groupby('run_id'):
            if not run_df.empty:
                start = run_df['shot_time'].min()
                end = run_df['shot_time'].max()
                last_ct = run_df.iloc[-1]['actual_ct']
                duration = (end - start).total_seconds() + last_ct
                run_durations.append(duration)
        
        total_runtime_sec = sum(run_durations)

        # Production Time
        prod_df = df[df['stop_flag'] == 0].copy()
        production_time_sec = prod_df['actual_ct'].sum()
        
        # Downtime (Plug)
        downtime_sec = total_runtime_sec - production_time_sec
        if downtime_sec < 0: downtime_sec = 0

        stops = df['stop_event'].sum()
        mttr_min = (downtime_sec / 60 / stops) if stops > 0 else 0
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0

        # --- Capacity Logic ---
        
        avg_approved_ct = df['approved_ct_for_run'].mean()
        max_cavities = df['working_cavities'].max()
        
        optimal_output_parts = (total_runtime_sec / avg_approved_ct) * max_cavities
        actual_output_parts = prod_df['working_cavities'].sum()
        target_output_parts = optimal_output_parts * (self.target_output_perc / 100.0)

        true_loss_parts = optimal_output_parts - actual_output_parts
        
        # Inefficiency Calculation
        prod_df['parts_delta'] = ((prod_df['approved_ct_for_run'] - prod_df['actual_ct']) / prod_df['approved_ct_for_run']) * prod_df['working_cavities']
        
        capacity_gain_fast_parts = prod_df.loc[prod_df['parts_delta'] > 0, 'parts_delta'].sum()
        capacity_loss_slow_parts = abs(prod_df.loc[prod_df['parts_delta'] < 0, 'parts_delta'].sum())
        
        # Inefficiency Time Calculation
        prod_df['time_delta'] = prod_df['approved_ct_for_run'] - prod_df['actual_ct']
        capacity_gain_fast_sec = prod_df.loc[prod_df['time_delta'] > 0, 'time_delta'].sum()
        capacity_loss_slow_sec = abs(prod_df.loc[prod_df['time_delta'] < 0, 'time_delta'].sum())

        net_cycle_loss_parts = capacity_loss_slow_parts - capacity_gain_fast_parts
        capacity_loss_downtime_parts = true_loss_parts - net_cycle_loss_parts
        
        net_cycle_loss_sec = capacity_loss_slow_sec - capacity_gain_fast_sec
        total_capacity_loss_sec = downtime_sec + net_cycle_loss_sec
        
        gap_to_target_parts = actual_output_parts - target_output_parts
        capacity_loss_vs_target_parts = max(0, -gap_to_target_parts)
        
        # --- NEW: Shot Counts for Run Rate Dashboard ---
        total_shots = len(df)
        stop_count_shots = df['stop_flag'].sum()
        normal_shots = total_shots - stop_count_shots
        
        # Run Rate Efficiency (Shot based)
        run_rate_efficiency = (normal_shots / total_shots) if total_shots > 0 else 0
        
        # Capacity Efficiency (Part based)
        capacity_efficiency = (actual_output_parts / optimal_output_parts) if optimal_output_parts > 0 else 0

        # Shot Typing
        epsilon = 0.001
        conditions = [
            df['stop_flag'] == 1,
            df['actual_ct'] > (df['approved_ct_for_run'] + epsilon), 
            df['actual_ct'] < (df['approved_ct_for_run'] - epsilon)
        ]
        choices = ['Downtime (Stop)', 'Slow Cycle', 'Fast Cycle']
        df['shot_type'] = np.select(conditions, choices, default='On Target')
        df.loc[is_new_run, 'shot_type'] = 'Run Break (Excluded)'

        return {
            "processed_df": df,
            "total_runtime_sec": total_runtime_sec,
            "production_time_sec": production_time_sec,
            "downtime_sec": downtime_sec,
            "mttr_min": mttr_min,
            "stability_index": stability_index,
            "stops": stops,
            "optimal_output_parts": optimal_output_parts,
            "actual_output_parts": actual_output_parts,
            "target_output_parts": target_output_parts,
            "capacity_loss_downtime_parts": capacity_loss_downtime_parts,
            "capacity_loss_slow_parts": capacity_loss_slow_parts,
            "capacity_gain_fast_parts": capacity_gain_fast_parts,
            "total_capacity_loss_parts": true_loss_parts,
            "total_capacity_loss_sec": total_capacity_loss_sec,
            "gap_to_target_parts": gap_to_target_parts,
            "capacity_loss_vs_target_parts": capacity_loss_vs_target_parts,
            # Updated Keys for Dashboard
            "efficiency_rate": run_rate_efficiency, # Mapped to Run Rate Efficiency
            "capacity_efficiency": capacity_efficiency,
            "total_shots": total_shots,
            "normal_shots": normal_shots
        }

def calculate_run_summaries(df_period, config):
    """Calculates per-run metrics for the breakdown table."""
    summary_list = []
    
    # Group by run_id
    if 'run_id' not in df_period.columns:
        return pd.DataFrame()
        
    for r_id, df_run in df_period.groupby('run_id'):
        if df_run.empty: continue
        
        # Calculate metrics for this specific run
        calc = CapacityRiskCalculator(df_run, **config)
        res = calc.results
        
        summary_list.append({
            'run_id': r_id,
            'start_time': df_run['shot_time'].min(),
            'end_time': df_run['shot_time'].max(),
            'total_shots': res['total_shots'],
            'normal_shots': res['normal_shots'],
            'stop_events': res['stops'],
            'stopped_shots': res['total_shots'] - res['normal_shots'],
            'mode_ct': df_run['mode_ct'].iloc[0] if 'mode_ct' in df_run else 0,
            'lower_limit': df_run['mode_lower'].iloc[0] if 'mode_lower' in df_run else 0,
            'upper_limit': df_run['mode_upper'].iloc[0] if 'mode_upper' in df_run else 0,
            'total_runtime_sec': res['total_runtime_sec'],
            'production_time_sec': res['production_time_sec'],
            'downtime_sec': res['downtime_sec']
        })
        
    if not summary_list:
        return pd.DataFrame()
        
    return pd.DataFrame(summary_list).sort_values('start_time')

# ==============================================================================
# --- AGGREGATION, PREDICTION & RISK LOGIC ---
# ==============================================================================