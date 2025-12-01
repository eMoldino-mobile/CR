import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import timedelta
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
    'optimal_line': 'darkblue'
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
        df.loc[df['approved_ct'] <= 0, 'approved_ct'] = 1
        
        df = df.sort_values("shot_time").reset_index(drop=True)

        # Run Identification
        df['time_diff_sec'] = df['shot_time'].diff().dt.total_seconds().fillna(0)
        if len(df) > 0: df.loc[0, 'time_diff_sec'] = df.loc[0, 'actual_ct']

        is_new_run = df['time_diff_sec'] > (self.run_interval_hours * 3600)
        df['run_id'] = is_new_run.cumsum()

        # Mode CT & Limits
        run_modes = df[df['actual_ct'] < 1000].groupby('run_id')['actual_ct'].apply(
            lambda x: x.mode().iloc[0] if not x.mode().empty else x.mean()
        )
        df['mode_ct'] = df['run_id'].map(run_modes)
        lower_limit = df['mode_ct'] * (1 - self.tolerance)
        upper_limit = df['mode_ct'] * (1 + self.tolerance)
        
        df['mode_lower'] = lower_limit
        df['mode_upper'] = upper_limit

        # Stop Detection
        is_time_gap = (df['time_diff_sec'] > (df['actual_ct'] + self.downtime_gap_tolerance)) & (~is_new_run)
        is_abnormal = ((df['actual_ct'] < lower_limit) | (df['actual_ct'] > upper_limit))
        is_hard_stop = df['actual_ct'] >= 999.9

        df['stop_flag'] = np.where(is_time_gap | is_abnormal | is_hard_stop, 1, 0)
        df.loc[is_new_run, 'stop_flag'] = 0
        if not df.empty: df.loc[0, 'stop_flag'] = 0
        
        df['stop_event'] = (df["stop_flag"] == 1) & (df["stop_flag"].shift(1, fill_value=0) == 0)

        # Adjusted Time
        df['adj_ct_sec'] = df['actual_ct']
        df.loc[is_time_gap, 'adj_ct_sec'] = df['time_diff_sec']
        df.loc[is_new_run, 'adj_ct_sec'] = df['actual_ct']

        # Metrics
        total_runtime_sec = df['adj_ct_sec'].sum()
        prod_df = df[df['stop_flag'] == 0].copy()
        production_time_sec = prod_df['actual_ct'].sum()
        downtime_sec = total_runtime_sec - production_time_sec
        if downtime_sec < 0: downtime_sec = 0

        stops = df['stop_event'].sum()
        mttr_min = (downtime_sec / 60 / stops) if stops > 0 else 0
        stability_index = (production_time_sec / total_runtime_sec * 100) if total_runtime_sec > 0 else 100.0

        # Capacity Logic
        avg_approved_ct = df['approved_ct'].mean()
        max_cavities = df['working_cavities'].max()
        
        optimal_output_parts = (total_runtime_sec / avg_approved_ct) * max_cavities
        actual_output_parts = prod_df['working_cavities'].sum()
        target_output_parts = optimal_output_parts * (self.target_output_perc / 100.0)

        true_loss_parts = optimal_output_parts - actual_output_parts
        
        prod_df['parts_delta'] = ((prod_df['approved_ct'] - prod_df['actual_ct']) / prod_df['approved_ct']) * prod_df['working_cavities']
        
        capacity_gain_fast_parts = prod_df.loc[prod_df['parts_delta'] > 0, 'parts_delta'].sum()
        capacity_loss_slow_parts = abs(prod_df.loc[prod_df['parts_delta'] < 0, 'parts_delta'].sum())

        net_cycle_loss_parts = capacity_loss_slow_parts - capacity_gain_fast_parts
        capacity_loss_downtime_parts = true_loss_parts - net_cycle_loss_parts # The Plug
        
        gap_to_target_parts = actual_output_parts - target_output_parts
        capacity_loss_vs_target_parts = max(0, -gap_to_target_parts)

        # Classification
        conditions = [
            df['stop_flag'] == 1,
            df['actual_ct'] > df['approved_ct'] * 1.02, 
            df['actual_ct'] < df['approved_ct'] * 0.98 
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
            "gap_to_target_parts": gap_to_target_parts,
            "efficiency_rate": (actual_output_parts / optimal_output_parts) if optimal_output_parts > 0 else 0
        }

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================

def plot_waterfall(metrics, benchmark_mode="Optimal"):
    total_opt = metrics['optimal_output_parts']
    total_target = metrics['target_output_parts']
    actual = metrics['actual_output_parts']
    
    loss_dt = -metrics['capacity_loss_downtime_parts']
    loss_slow = -metrics['capacity_loss_slow_parts']
    gain_fast = metrics['capacity_gain_fast_parts']
    
    measure = ["absolute", "relative", "relative", "relative", "total"]
    x_label = ["Optimal Output", "Downtime Loss", "Slow Cycle Loss", "Fast Cycle Gain", "Actual Output"]
    y_val = [total_opt, loss_dt, loss_slow, gain_fast, actual]
    text_val = [f"{total_opt:,.0f}", f"{loss_dt:,.0f}", f"{loss_slow:,.0f}", f"+{gain_fast:,.0f}", f"{actual:,.0f}"]
    
    fig = go.Figure(go.Waterfall(
        name="Breakdown", orientation="v", measure=measure, x=x_label, y=y_val, text=text_val,
        textposition="outside", connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": PASTEL_COLORS['red']}},
        increasing={"marker": {"color": PASTEL_COLORS['green']}},
        totals={"marker": {"color": PASTEL_COLORS['blue']}}
    ))
    
    if benchmark_mode == "Target Output":
        fig.add_shape(type="line", x0=-0.5, x1=4.5, y0=total_target, y1=total_target,
                      line=dict(color=PASTEL_COLORS['target_line'], width=2, dash="dash"))
        fig.add_annotation(x=0, y=total_target, text=f"Target: {total_target:,.0f}", showarrow=False, yshift=10)

    fig.update_layout(title="Capacity Loss Breakdown", showlegend=False, height=450)
    return fig

def plot_performance_breakdown(df_agg, x_col, benchmark_mode):
    """
    Stacked Bar Chart: Actual + Losses vs Optimal/Target Lines
    """
    fig = go.Figure()
    
    # 1. Actual Output
    fig.add_trace(go.Bar(
        x=df_agg[x_col], y=df_agg['Actual Output'], name='Actual Output',
        marker_color=PASTEL_COLORS['blue']
    ))
    
    # 2. Net Cycle Loss (Combined Slow - Fast)
    # Ensure we don't plot negative bars if Gain > Loss
    cycle_loss_net = df_agg['Slow Loss'] - df_agg['Fast Gain']
    cycle_loss_plot = cycle_loss_net.clip(lower=0)
    
    fig.add_trace(go.Bar(
        x=df_agg[x_col], y=cycle_loss_plot, name='Net Cycle Loss',
        marker_color=PASTEL_COLORS['orange']
    ))
    
    # 3. Downtime Loss
    fig.add_trace(go.Bar(
        x=df_agg[x_col], y=df_agg['Downtime Loss'], name='Downtime Loss',
        marker_color=PASTEL_COLORS['grey']
    ))
    
    # Lines
    fig.add_trace(go.Scatter(
        x=df_agg[x_col], y=df_agg['Optimal Output'], name='Optimal Output',
        mode='lines', line=dict(color=PASTEL_COLORS['optimal_line'], dash='dot')
    ))
    
    if benchmark_mode == "Target Output":
        fig.add_trace(go.Scatter(
            x=df_agg[x_col], y=df_agg['Target Output'], name='Target Output',
            mode='lines', line=dict(color=PASTEL_COLORS['target_line'], dash='dash')
        ))

    fig.update_layout(barmode='stack', title="Performance Breakdown", hovermode="x unified", height=450)
    return fig

def plot_shot_analysis(df_shots, zoom_y=None):
    """
    Detailed shot-by-shot bar chart with Mode Bands.
    """
    if df_shots.empty: return go.Figure()
    
    fig = go.Figure()
    
    # Color Map
    color_map = {
        'Slow Cycle': PASTEL_COLORS['red'],
        'Fast Cycle': PASTEL_COLORS['orange'],
        'On Target': PASTEL_COLORS['blue'],
        'Downtime (Stop)': PASTEL_COLORS['grey'],
        'Run Break (Excluded)': '#d3d3d3'
    }
    
    for shot_type, color in color_map.items():
        subset = df_shots[df_shots['shot_type'] == shot_type]
        if not subset.empty:
            fig.add_trace(go.Bar(
                x=subset['shot_time'], y=subset['actual_ct'],
                name=shot_type, marker_color=color,
                hovertemplate='Time: %{x}<br>CT: %{y:.2f}s<extra></extra>'
            ))
            
    # Add Mode Bands (Grey rectangles)
    # Simplify by doing one rect per run to avoid 1000s of shapes
    for r_id, run_df in df_shots.groupby('run_id'):
        lower = run_df['mode_lower'].iloc[0]
        upper = run_df['mode_upper'].iloc[0]
        start = run_df['shot_time'].min()
        end = run_df['shot_time'].max()
        
        fig.add_shape(type="rect", x0=start, x1=end, y0=lower, y1=upper,
                      fillcolor="grey", opacity=0.2, line_width=0)

    # Reference Line (Average Approved CT)
    avg_ref = df_shots['approved_ct'].mean()
    fig.add_hline(y=avg_ref, line_dash="dash", line_color="green", annotation_text=f"Avg Approved CT: {avg_ref:.2f}s")

    layout_args = dict(title="Shot-by-Shot Analysis", yaxis_title="Cycle Time (sec)", hovermode="closest")
    if zoom_y: layout_args['yaxis_range'] = [0, zoom_y]
    
    fig.update_layout(**layout_args)
    return fig

# ==============================================================================
# --- AGGREGATION & RISK ---
# ==============================================================================

def get_aggregated_data(df, freq_mode, config):
    """
    Generates the aggregated dataframe for the Stacked Bar Chart & Detailed Tables.
    freq_mode: 'Daily', 'Weekly', 'Monthly', 'by Run'
    """
    rows = []
    
    # Define grouper
    if freq_mode == 'Daily': grouper = df.groupby(df['shot_time'].dt.date)
    elif freq_mode == 'Weekly': grouper = df.groupby(df['shot_time'].dt.to_period('W').astype(str))
    elif freq_mode == 'Monthly': grouper = df.groupby(df['shot_time'].dt.to_period('M').astype(str))
    elif freq_mode == 'by Run': 
        # Must run calc first to get run_ids
        temp_calc = CapacityRiskCalculator(df, **config)
        grouper = temp_calc.results['processed_df'].groupby('run_id')
    else: return pd.DataFrame()

    for group_name, df_subset in grouper:
        calc = CapacityRiskCalculator(df_subset, **config)
        res = calc.results
        if not res: continue
        
        rows.append({
            'Period': group_name,
            'Actual Output': res['actual_output_parts'],
            'Optimal Output': res['optimal_output_parts'],
            'Target Output': res['target_output_parts'],
            'Downtime Loss': res['capacity_loss_downtime_parts'],
            'Slow Loss': res['capacity_loss_slow_parts'],
            'Fast Gain': res['capacity_gain_fast_parts'],
            'Net Cycle Loss': res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts'],
            'Total Loss': res['total_capacity_loss_parts'],
            'Gap to Target': res['gap_to_target_parts'],
            'Run Time': format_seconds_to_dhm(res['total_runtime_sec']),
            'Prod Time': format_seconds_to_dhm(res['production_time_sec']),
            'Downtime': format_seconds_to_dhm(res['downtime_sec'])
        })
        
    return pd.DataFrame(rows)

def calculate_capacity_risk_scores(df_all, config):
    """Risk Tower Logic"""
    risk_data = []
    for tool_id, df_tool in df_all.groupby('tool_id'):
        max_date = df_tool['shot_time'].max()
        cutoff_date = max_date - timedelta(weeks=4)
        df_period = df_tool[df_tool['shot_time'] >= cutoff_date].copy()
        
        if df_period.empty: continue
        
        calc = CapacityRiskCalculator(df_period, **config)
        res = calc.results
        if res['target_output_parts'] == 0: continue
        
        ach_perc = (res['actual_output_parts'] / res['target_output_parts']) * 100
        
        # Trend
        midpoint = cutoff_date + (max_date - cutoff_date) / 2
        df_late = df_period[df_period['shot_time'] >= midpoint]
        df_early = df_period[df_period['shot_time'] < midpoint]
        
        trend = "Stable"
        if not df_early.empty and not df_late.empty:
            c_early = CapacityRiskCalculator(df_early, **config).results
            c_late = CapacityRiskCalculator(df_late, **config).results
            early_rate = c_early['actual_output_parts'] / (c_early['total_runtime_sec']/3600) if c_early['total_runtime_sec'] > 0 else 0
            late_rate = c_late['actual_output_parts'] / (c_late['total_runtime_sec']/3600) if c_late['total_runtime_sec'] > 0 else 0
            
            if late_rate < early_rate * 0.95: trend = "Declining"
            elif late_rate > early_rate * 1.05: trend = "Improving"

        base_score = min(ach_perc, 100)
        if trend == "Declining": base_score -= 20
        
        risk_data.append({
            'Tool ID': tool_id,
            'Risk Score': max(0, base_score),
            'Achievement %': ach_perc,
            'Trend': trend,
            'Gap': res['gap_to_target_parts']
        })
    return pd.DataFrame(risk_data).sort_values('Risk Score')