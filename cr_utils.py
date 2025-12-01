import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ==============================================================================
# --- CONSTANTS & SHARED FUNCTIONS ---
# ==============================================================================

PASTEL_COLORS = {
    'red': '#ff6961',
    'orange': '#ffb347',
    'green': '#77dd77',
    'blue': '#3498DB',
    'grey': '#808080'
}

def format_seconds_to_dhm(total_seconds):
    """Converts total seconds into a 'Xd Yh Zm' string."""
    if pd.isna(total_seconds) or total_seconds < 0: return "N/A"
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
    """
    Loads data for Capacity Risk. 
    Standardizes: tool_id, shot_time, actual_ct, approved_ct, working_cavities.
    """
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file) if file.name.endswith(('.xls', '.xlsx')) else pd.read_csv(file)
            
            # Normalize Columns
            col_map = {col.strip().upper(): col for col in df.columns}
            def get_col(target_list):
                for t in target_list:
                    if t in col_map: return col_map[t]
                return None

            # 1. Tool ID
            tool_col = get_col(["TOOLING ID", "EQUIPMENT CODE", "TOOL_ID", "TOOL"])
            if tool_col: df.rename(columns={tool_col: "tool_id"}, inplace=True)

            # 2. Shot Time
            time_col = get_col(["SHOT TIME", "TIMESTAMP", "DATE", "TIME", "DATETIME"])
            if time_col: df.rename(columns={time_col: "shot_time"}, inplace=True)

            # 3. Actual CT
            act_ct_col = get_col(["ACTUAL CT", "ACTUAL_CT", "CYCLE TIME", "CT"])
            if act_ct_col: df.rename(columns={act_ct_col: "actual_ct"}, inplace=True)

            # 4. Approved CT (Specific to CR)
            app_ct_col = get_col(["APPROVED CT", "APPROVED_CT", "STD CT", "STANDARD CT", "REFERENCE CT"])
            if app_ct_col: df.rename(columns={app_ct_col: "approved_ct"}, inplace=True)

            # 5. Working Cavities (Specific to CR)
            cav_col = get_col(["WORKING CAVITIES", "CAVITIES", "ACTUAL CAVITIES", "CAVITY"])
            if cav_col: df.rename(columns={cav_col: "working_cavities"}, inplace=True)

            # Essential Validation
            if "shot_time" in df.columns and "actual_ct" in df.columns:
                df["shot_time"] = pd.to_datetime(df["shot_time"], errors="coerce")
                df["actual_ct"] = pd.to_numeric(df["actual_ct"], errors="coerce")
                df.dropna(subset=["shot_time", "actual_ct"], inplace=True)
                df_list.append(df)
                
        except Exception as e:
            continue
    
    if not df_list:
        return pd.DataFrame()
        
    return pd.concat(df_list, ignore_index=True)


# ==============================================================================
# --- CORE CALCULATION ENGINE (EXTENDED FOR CAPACITY) ---
# ==============================================================================

class CapacityRiskCalculator:
    """
    Based on the RunRateCalculator, but extended to calculate Capacity Losses/Gains.
    """
    def __init__(self, df: pd.DataFrame, 
                 tolerance: float, 
                 downtime_gap_tolerance: float, 
                 run_interval_hours: float,
                 target_output_perc: float = 100.0,
                 default_cavities: int = 1):
        
        self.df_raw = df.copy()
        self.tolerance = tolerance
        self.downtime_gap_tolerance = downtime_gap_tolerance
        self.run_interval_hours = run_interval_hours
        self.target_output_perc = target_output_perc
        self.default_cavities = default_cavities
        
        self.results = self._calculate_metrics()

    def _calculate_metrics(self) -> dict:
        df = self.df_raw.copy()
        if df.empty: return {}

        # --- 1. Standardization & Defaults ---
        if 'approved_ct' not in df.columns: df['approved_ct'] = df['actual_ct'].median() # Fallback
        if 'working_cavities' not in df.columns: df['working_cavities'] = self.default_cavities
        
        df['approved_ct'] = pd.to_numeric(df['approved_ct'], errors='coerce').fillna(1)
        df['working_cavities'] = pd.to_numeric(df['working_cavities'], errors='coerce').fillna(self.default_cavities)
        
        df = df.sort_values("shot_time").reset_index(drop=True)

        # --- 2. Run Identification (The RR Backbone) ---
        # Calculate time diff to find "Breaks" (Weekends, Shift changes > interval)
        df['time_diff_sec'] = df['shot_time'].diff().dt.total_seconds().fillna(0)
        # Fix first shot
        if len(df) > 0: df.loc[0, 'time_diff_sec'] = df.loc[0, 'actual_ct']

        # Determine Runs based on interval threshold
        is_new_run = df['time_diff_sec'] > (self.run_interval_hours * 3600)
        df['run_id'] = is_new_run.cumsum()

        # --- 3. Mode CT Calculation (Per Run for accuracy) ---
        run_modes = df[df['actual_ct'] < 1000].groupby('run_id')['actual_ct'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()
        )
        df['mode_ct'] = df['run_id'].map(run_modes)
        
        # --- 4. Stop Detection (RR Logic) ---
        # A stop is either an abnormal cycle (outside mode tolerance) or a time gap
        lower_limit = df['mode_ct'] * (1 - self.tolerance)
        upper_limit = df['mode_ct'] * (1 + self.tolerance)
        
        # Time Gap Logic (exclude the Run Breaks)
        is_time_gap = (df['time_diff_sec'] > (df['actual_ct'] + self.downtime_gap_tolerance)) & (~is_new_run)
        
        # Abnormal Cycle Logic
        is_abnormal = ((df['actual_ct'] < lower_limit) | (df['actual_ct'] > upper_limit))
        
        df['stop_flag'] = np.where(is_time_gap | is_abnormal, 1, 0)
        # Force first shot of a *new run* to be valid (it's the start)
        df.loc[is_new_run, 'stop_flag'] = 0

        # --- 5. Adjusted Time (Accounting for gaps) ---
        # If it's a time gap stop, the "duration" of that shot is the gap
        df['adj_ct_sec'] = df['actual_ct']
        df.loc[is_time_gap, 'adj_ct_sec'] = df['time_diff_sec']
        # If it's a new run, the "duration" is just the cycle time (gap excluded)
        df.loc[is_new_run, 'adj_ct_sec'] = df['actual_ct']

        # --- 6. Capacity Calculations (The CR Layer) ---
        
        # Totals for the dataframe
        # Run Time = Sum of all adjusted cycle times (includes downtime gaps, excludes run breaks)
        total_run_time_sec = df['adj_ct_sec'].sum()
        
        # Production Time = Sum of actual cycle times
        total_actual_ct_sec = df['actual_ct'].sum()
        
        # Downtime = Difference
        capacity_loss_downtime_sec = total_run_time_sec - total_actual_ct_sec
        
        # Averages for calc
        avg_approved_ct = df['approved_ct'].mean()
        max_cavities = df['working_cavities'].max()

        # 6a. Optimal Output (The Ceiling)
        # Theoretical max if machine ran 100% of the *Run Time* at *Approved CT*
        optimal_output_parts = (total_run_time_sec / avg_approved_ct) * max_cavities
        
        # 6b. Actual Output
        actual_output_parts = df['working_cavities'].sum()
        
        # 6c. Target Output
        target_output_parts = optimal_output_parts * (self.target_output_perc / 100.0)

        # 6d. Loss Segmentation (Parts)
        
        # 1. Downtime Loss: Time lost * Rate
        # We calculate "Parts lost due to downtime"
        capacity_loss_downtime_parts = (capacity_loss_downtime_sec / avg_approved_ct) * max_cavities
        
        # 2. Inefficiency (Slow Cycles vs Fast Cycles)
        # Loss = (Actual - Approved) / Approved * Cavities
        df['parts_delta'] = ((df['approved_ct'] - df['actual_ct']) / df['approved_ct']) * df['working_cavities']
        
        capacity_gain_fast_parts = df.loc[df['parts_delta'] > 0, 'parts_delta'].sum()
        capacity_loss_slow_parts = abs(df.loc[df['parts_delta'] < 0, 'parts_delta'].sum())

        # Reconciliation check
        total_capacity_loss_parts = capacity_loss_downtime_parts + capacity_loss_slow_parts - capacity_gain_fast_parts
        
        # Gap to Target
        gap_to_target_parts = actual_output_parts - target_output_parts
        capacity_loss_vs_target_parts = max(0, -gap_to_target_parts)

        # --- 7. Shot Classification for Charts ---
        conditions = [
            df['stop_flag'] == 1,
            df['actual_ct'] > df['approved_ct'] * 1.02, # Slow
            df['actual_ct'] < df['approved_ct'] * 0.98  # Fast
        ]
        choices = ['Downtime (Stop)', 'Slow Cycle', 'Fast Cycle']
        df['shot_type'] = np.select(conditions, choices, default='On Target')

        return {
            "processed_df": df,
            "total_run_time_sec": total_run_time_sec,
            "total_actual_ct_sec": total_actual_ct_sec,
            "capacity_loss_downtime_sec": capacity_loss_downtime_sec,
            "optimal_output_parts": optimal_output_parts,
            "actual_output_parts": actual_output_parts,
            "target_output_parts": target_output_parts,
            "capacity_loss_downtime_parts": capacity_loss_downtime_parts,
            "capacity_loss_slow_parts": capacity_loss_slow_parts,
            "capacity_gain_fast_parts": capacity_gain_fast_parts,
            "total_capacity_loss_parts": total_capacity_loss_parts,
            "gap_to_target_parts": gap_to_target_parts,
            "capacity_loss_vs_target_parts": capacity_loss_vs_target_parts,
            "efficiency_rate": (actual_output_parts / optimal_output_parts) if optimal_output_parts > 0 else 0
        }

def aggregate_by_period(df, period_col, tolerance, dt_gap, run_int, target_perc, def_cav):
    """
    Groups data by a period column (e.g., 'date', 'week') and runs the calculator for each.
    Returns a DataFrame of aggregated results.
    """
    results_list = []
    
    for period, df_subset in df.groupby(period_col):
        calc = CapacityRiskCalculator(df_subset, tolerance, dt_gap, run_int, target_perc, def_cav)
        res = calc.results
        
        row = {
            period_col: period,
            'Optimal Output': res['optimal_output_parts'],
            'Target Output': res['target_output_parts'],
            'Actual Output': res['actual_output_parts'],
            'Total Loss': res['total_capacity_loss_parts'],
            'Downtime Loss': res['capacity_loss_downtime_parts'],
            'Slow Cycle Loss': res['capacity_loss_slow_parts'],
            'Fast Cycle Gain': res['capacity_gain_fast_parts'],
            'Run Time (hrs)': res['total_run_time_sec'] / 3600,
            'Gap to Target': res['gap_to_target_parts']
        }
        results_list.append(row)
        
    return pd.DataFrame(results_list).sort_values(period_col)

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_waterfall(metrics_dict, benchmark_mode="Optimal"):
    """
    Creates the Waterfall chart showing how capacity is lost.
    """
    
    total_opt = metrics_dict['optimal_output_parts']
    total_target = metrics_dict['target_output_parts']
    actual = metrics_dict['actual_output_parts']
    
    loss_dt = -metrics_dict['capacity_loss_downtime_parts']
    loss_slow = -metrics_dict['capacity_loss_slow_parts']
    gain_fast = metrics_dict['capacity_gain_fast_parts']
    
    # Starting Point
    if benchmark_mode == "Target":
        start_val = total_target
        start_label = "Target Output"
        # If calculating vs Target, we need to treat the gap differently, 
        # but for visual simplicity in this refactor, we stick to the Optimal breakdown
        # and show Target as a line.
    
    measure = ["absolute", "relative", "relative", "relative", "total"]
    x_label = ["Optimal Capacity", "Downtime Loss", "Slow Cycle Loss", "Fast Cycle Gain", "Actual Output"]
    y_val = [total_opt, loss_dt, loss_slow, gain_fast, actual]
    text_val = [f"{total_opt:,.0f}", f"{loss_dt:,.0f}", f"{loss_slow:,.0f}", f"+{gain_fast:,.0f}", f"{actual:,.0f}"]
    
    fig = go.Figure(go.Waterfall(
        name="Capacity Bridge", orientation="v",
        measure=measure,
        x=x_label,
        y=y_val,
        text=text_val,
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": PASTEL_COLORS['red']}},
        increasing={"marker": {"color": PASTEL_COLORS['green']}},
        totals={"marker": {"color": PASTEL_COLORS['blue']}}
    ))
    
    if benchmark_mode == "Target":
        fig.add_shape(type="line", x0=-0.5, x1=4.5, y0=total_target, y1=total_target,
                      line=dict(color="orange", width=2, dash="dash"))
        fig.add_annotation(x=0, y=total_target, text="Target", showarrow=False, yshift=10)

    fig.update_layout(title="Capacity Loss Waterfall", showlegend=False, height=450)
    return fig

def plot_capacity_trend(trend_df, x_col):
    """
    Stacked bar chart of Actual vs Losses over time.
    """
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Bar(name='Actual Output', x=trend_df[x_col], y=trend_df['Actual Output'], marker_color=PASTEL_COLORS['blue']))
    
    # Losses
    fig.add_trace(go.Bar(name='Downtime Loss', x=trend_df[x_col], y=trend_df['Downtime Loss'], marker_color=PASTEL_COLORS['grey']))
    fig.add_trace(go.Bar(name='Slow Cycle Loss', x=trend_df[x_col], y=trend_df['Slow Cycle Loss'], marker_color=PASTEL_COLORS['red']))
    
    # Lines
    fig.add_trace(go.Scatter(name='Optimal Output', x=trend_df[x_col], y=trend_df['Optimal Output'], mode='lines', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(name='Target Output', x=trend_df[x_col], y=trend_df['Target Output'], mode='lines', line=dict(color='orange', dash='dash')))

    fig.update_layout(barmode='stack', title="Capacity Trend", xaxis_title=x_col, yaxis_title="Parts")
    return fig