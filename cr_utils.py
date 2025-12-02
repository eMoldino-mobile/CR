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
            'mode_lower': df_run['mode_lower'].iloc[0] if 'mode_lower' in df_run else 0,
            'mode_upper': df_run['mode_upper'].iloc[0] if 'mode_upper' in df_run else 0,
            'lower_limit': df_run['mode_lower'].iloc[0] if 'mode_lower' in df_run else 0, # Alias for consistency
            'upper_limit': df_run['mode_upper'].iloc[0] if 'mode_upper' in df_run else 0, # Alias for consistency
            
            # Time Metrics
            'total_runtime_sec': res['total_runtime_sec'],
            'production_time_sec': res['production_time_sec'],
            'downtime_sec': res['downtime_sec'],
            'total_capacity_loss_sec': res['total_capacity_loss_sec'],
            
            # Capacity Metrics (Added for Aggregation)
            'optimal_output_parts': res['optimal_output_parts'],
            'target_output_parts': res['target_output_parts'], # Added for Trend Analysis
            'actual_output_parts': res['actual_output_parts'],
            'capacity_loss_downtime_parts': res['capacity_loss_downtime_parts'],
            'capacity_loss_slow_parts': res['capacity_loss_slow_parts'],
            'capacity_gain_fast_parts': res['capacity_gain_fast_parts'],
            'total_capacity_loss_parts': res['total_capacity_loss_parts']
        })
        
    if not summary_list:
        return pd.DataFrame()
        
    return pd.DataFrame(summary_list).sort_values('start_time')

# ==============================================================================
# --- AGGREGATION, PREDICTION & RISK LOGIC ---
# ==============================================================================

def get_aggregated_data(df, freq_mode, config):
    """Generates aggregated dataframe for tables/charts."""
    rows = []
    
    if freq_mode == 'Daily': grouper = df.groupby(df['shot_time'].dt.date)
    elif freq_mode == 'Weekly': grouper = df.groupby(df['shot_time'].dt.to_period('W').astype(str))
    elif freq_mode == 'Monthly': grouper = df.groupby(df['shot_time'].dt.to_period('M').astype(str))
    elif freq_mode == 'Hourly': grouper = df.groupby(df['shot_time'].dt.floor('H'))
    elif freq_mode == 'by Run': 
        temp_calc = CapacityRiskCalculator(df, **config)
        grouper = temp_calc.results['processed_df'].groupby('run_id')
    else: return pd.DataFrame()

    for group_name, df_subset in grouper:
        calc = CapacityRiskCalculator(df_subset, **config)
        res = calc.results
        if not res: continue
        
        # New logic for period label consistency
        period_label = group_name
        if freq_mode == 'by Run':
            try:
                period_label = f"Run {int(group_name) + 1}"
            except (ValueError, TypeError):
                pass

        rows.append({
            'Period': period_label,
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
            'Downtime': format_seconds_to_dhm(res['downtime_sec']),
            
            # --- Added for detailed Tables ---
            'Run Time Sec': res['total_runtime_sec'],
            'Production Time Sec': res['production_time_sec'],
            'Downtime Sec': res['downtime_sec'],
            'Total Shots': res['total_shots'],
            'Normal Shots': res['normal_shots'],
            'Downtime Shots': res['total_shots'] - res['normal_shots']
        })
        
    return pd.DataFrame(rows)

def calculate_theoretical_capacity(df_daily_agg):
    if df_daily_agg.empty: return 0, 0, 0, 0
    
    min_date = pd.to_datetime(df_daily_agg['Period'].min())
    max_date = pd.to_datetime(df_daily_agg['Period'].max())
    total_span_days = (max_date - min_date).days + 1
    if total_span_days <= 0: return 0, 0, 0, 0
    
    operating_days = len(df_daily_agg)
    total_span_weeks = total_span_days / 7.0
    avg_prod_days_per_week = operating_days / total_span_weeks if total_span_weeks > 0 else 0
    
    monthly_factor = avg_prod_days_per_week * 4.33
    peak_daily = df_daily_agg['Actual Output'].quantile(0.90)
    theoretical_monthly = peak_daily * monthly_factor
    
    return theoretical_monthly, monthly_factor, avg_prod_days_per_week, peak_daily

def generate_prediction_data(df_daily_agg, start_date, target_date, demand_target):
    df = df_daily_agg.copy()
    df['Period'] = pd.to_datetime(df['Period'])
    df = df.sort_values('Period').set_index('Period')
    df['Cum Actual'] = df['Actual Output'].cumsum()
    
    start_ts = pd.Timestamp(start_date)
    past_data = df[df.index <= start_ts]
    start_val = past_data['Cum Actual'].iloc[-1] if not past_data.empty else 0
    
    days_in_data = len(df)
    current_daily_rate = df['Actual Output'].sum() / days_in_data if days_in_data > 0 else 0
    peak_daily = df['Actual Output'].quantile(0.90)
    
    projection_days = (target_date - start_date).days
    if projection_days < 1: projection_days = 1
    future_dates = [start_date + timedelta(days=i) for i in range(projection_days + 1)]
    
    proj_actual = [start_val + (current_daily_rate * i) for i in range(len(future_dates))]
    proj_peak = [start_val + (peak_daily * i) for i in range(len(future_dates))]
    
    proj_target = []
    if df['Target Output'].sum() > 0:
        avg_target = df['Target Output'].mean()
        proj_target = [start_val + (avg_target * i) for i in range(len(future_dates))]
        
    return future_dates, proj_actual, proj_peak, proj_target, start_val, current_daily_rate, peak_daily

def calculate_capacity_risk_scores(df_all, config):
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

# ==============================================================================
# --- PLOTTING FUNCTIONS ---
# ==============================================================================

def create_donut_chart(value, title, color_scheme='blue'):
    """Creates a donut chart similar to the Run Rate app style."""
    if color_scheme == 'blue': main_color = PASTEL_COLORS['blue']
    elif color_scheme == 'green': main_color = PASTEL_COLORS['green']
    elif color_scheme == 'dynamic':
        if value < 70: main_color = PASTEL_COLORS['red']
        elif value < 90: main_color = PASTEL_COLORS['orange']
        else: main_color = PASTEL_COLORS['green']
    else: main_color = color_scheme

    plot_val = min(value, 100)
    remainder = 100 - plot_val
    
    fig = go.Figure(data=[go.Pie(
        values=[plot_val, remainder], hole=0.75, sort=False, direction='clockwise',
        textinfo='none', marker=dict(colors=[main_color, '#e6e6e6']), hoverinfo='none'
    )])

    fig.add_annotation(text=f"{value:.1f}%", x=0.5, y=0.5, font=dict(size=24, weight='bold', color=main_color), showarrow=False)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', y=0.95, font=dict(size=14)),
        margin=dict(l=20, r=20, t=30, b=20), height=180, showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

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
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_agg[x_col], y=df_agg['Actual Output'], name='Actual Output', marker_color=PASTEL_COLORS['blue']))
    
    cycle_loss_net = df_agg['Slow Loss'] - df_agg['Fast Gain']
    cycle_loss_plot = cycle_loss_net.clip(lower=0)
    
    fig.add_trace(go.Bar(x=df_agg[x_col], y=cycle_loss_plot, name='Net Cycle Loss', marker_color=PASTEL_COLORS['orange']))
    fig.add_trace(go.Bar(x=df_agg[x_col], y=df_agg['Downtime Loss'], name='Downtime Loss', marker_color=PASTEL_COLORS['grey']))
    
    fig.add_trace(go.Scatter(x=df_agg[x_col], y=df_agg['Optimal Output'], name='Optimal Output', mode='lines', line=dict(color=PASTEL_COLORS['optimal_line'], dash='dot')))
    
    if benchmark_mode == "Target Output":
        fig.add_trace(go.Scatter(x=df_agg[x_col], y=df_agg['Target Output'], name='Target Output', mode='lines', line=dict(color=PASTEL_COLORS['target_line'], dash='dash')))

    fig.update_layout(barmode='stack', title="Performance Breakdown", hovermode="x unified", height=450)
    return fig

def plot_prediction_chart(dates, actual, peak, target, demand, start_date, start_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Projected (Current Rate)', line=dict(color=PASTEL_COLORS['blue'], dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=peak, mode='lines', name='Theoretical Max (P90)', line=dict(color=PASTEL_COLORS['green'], dash='dot')))
    
    if target: fig.add_trace(go.Scatter(x=dates, y=target, mode='lines', name='Target Trend', line=dict(color=PASTEL_COLORS['target_line'], dash='longdashdot')))
    if demand > 0: fig.add_hline(y=demand, line_dash="solid", line_color=PASTEL_COLORS['purple'], annotation_text=f"Demand: {demand:,.0f}")

    fig.add_vline(x=start_date, line_width=1, line_color="grey")
    fig.add_annotation(x=start_date, y=start_val, text="Start", showarrow=True, arrowhead=1)

    fig.update_layout(title="Future Capacity Prediction", xaxis_title="Date", yaxis_title="Cumulative Output", hovermode="x unified", height=500)
    return fig

def plot_shot_analysis(df_shots, zoom_y=None):
    if df_shots.empty: return go.Figure()
    fig = go.Figure()
    color_map = {'Slow Cycle': PASTEL_COLORS['red'], 'Fast Cycle': PASTEL_COLORS['orange'], 'On Target': PASTEL_COLORS['blue'], 'Downtime (Stop)': PASTEL_COLORS['grey'], 'Run Break (Excluded)': '#d3d3d3'}
    
    for shot_type, color in color_map.items():
        subset = df_shots[df_shots['shot_type'] == shot_type]
        if not subset.empty:
            fig.add_trace(go.Bar(x=subset['shot_time'], y=subset['actual_ct'], name=shot_type, marker_color=color, hovertemplate='Time: %{x}<br>CT: %{y:.2f}s<extra></extra>'))
            
    for r_id, run_df in df_shots.groupby('run_id'):
        lower = run_df['mode_lower'].iloc[0]
        upper = run_df['mode_upper'].iloc[0]
        start = run_df['shot_time'].min()
        end = run_df['shot_time'].max()
        
        fig.add_shape(type="rect", x0=start, x1=end, y0=lower, y1=upper, fillcolor="grey", opacity=0.2, line_width=0)
        if r_id == 0: fig.add_annotation(x=start, y=upper, text="Mode Tolerance Band", showarrow=False, yshift=10, font=dict(color="grey", size=10))

    avg_ref = df_shots['approved_ct'].mean()
    fig.add_hline(y=avg_ref, line_dash="dash", line_color="green", annotation_text=f"Avg Approved CT: {avg_ref:.2f}s")
    
    # Auto-zoom if not provided
    if zoom_y is None and not df_shots.empty:
        # Use 95th percentile or 4x Approved CT as a heuristic to avoid extreme stop outliers
        cts = df_shots['actual_ct']
        if len(cts) > 0:
            # Filter extremely large values just for the range calculation
            # Reasonable max is often just above the normal operating window
            ref_max = df_shots['approved_ct'].max() * 4
            
            # Also look at distribution, ignoring the top 5% which are usually stops
            dist_max = cts.quantile(0.95) * 1.5
            
            zoom_y = max(ref_max, dist_max)

    layout_args = dict(title="Shot-by-Shot Analysis", yaxis_title="Cycle Time (sec)", hovermode="closest")
    if zoom_y: layout_args['yaxis_range'] = [0, zoom_y]
    fig.update_layout(**layout_args)
    return fig