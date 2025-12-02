import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import cr_utils as cr_utils
import importlib

# Force reload of utils to ensure latest logic is used
importlib.reload(cr_utils)

# ==============================================================================
# --- PAGE CONFIG ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v9.8)")

# ==============================================================================
# --- 1. RENDER FUNCTIONS ---
# ==============================================================================

def render_risk_tower(df_all_tools, config):
    """Renders the Risk Tower (Tab 1)."""
    st.title("Capacity Risk Tower")
    st.info("This tower identifies tools at risk by analyzing weekly production gaps over the last 4 weeks.")

    with st.expander("ℹ️ How the Risk Tower Works"):
        st.markdown("""
        The Risk Tower evaluates each tool based on its performance over its own most recent 4-week period.
        
        - **Gap % (Avg)**: average percentage gap between Actual and Target output over the last 4 weeks.
        - **Weekly Trend**: Visual indicator of whether the gap is widening or closing.
        - **Risk Score**: Calculated based on the magnitude of the gap and the trend direction.
        - **Color Coding**:
            - <span style='background-color:#ff6961; color: black; padding: 2px 5px; border-radius: 5px;'>Red (High Gap)</span>: High Risk
            - <span style='background-color:#ffb347; color: black; padding: 2px 5px; border-radius: 5px;'>Orange (Moderate Gap)</span>: Medium Risk
            - <span style='background-color:#77dd77; color: black; padding: 2px 5px; border-radius: 5px;'>Green (Low/No Gap)</span>: Low Risk
        """, unsafe_allow_html=True)

    risk_df = cr_utils.calculate_capacity_risk_scores(df_all_tools, config)

    if risk_df.empty:
        st.warning("Not enough data to generate the Risk Tower.")
        return

    def style_risk(row):
        score = row['Risk Score']
        if score > 80: color = cr_utils.PASTEL_COLORS['green']
        elif score > 50: color = cr_utils.PASTEL_COLORS['orange']
        else: color = cr_utils.PASTEL_COLORS['red']
        return [f'background-color: {color}' for _ in row]

    st.dataframe(
        risk_df.style.apply(style_risk, axis=1)
        .format({'Risk Score': '{:.0f}', 'Achievement %': '{:.1f}%'}),
        use_container_width=True, 
        hide_index=True
    )

def render_trends_tab(df_tool, config):
    """Renders the Trends Tab (Tab 3) - Modeled on Run Rate App."""
    st.header("Historical Performance Trends")
    st.info("Trends are calculated using 'Run-Based' logic consistent with the Dashboard.")

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        trend_freq = st.selectbox("Select Trend Frequency", ["Daily", "Weekly", "Monthly"], key="cr_trend_freq")

    # Pre-process ALL data for the tool to establish Run IDs and standard metrics first
    # This prevents grouping errors where chunks of data might not have run_id
    base_calc = cr_utils.CapacityRiskCalculator(df_tool, **config)
    df_tool_processed = base_calc.results['processed_df']
    
    if df_tool_processed.empty:
        st.warning("No data available after processing.")
        return

    # Generate Trend Data
    trend_data = []
    
    if trend_freq == "Daily":
        grouper = df_tool_processed.groupby(df_tool_processed['shot_time'].dt.date)
        period_name = "Date"
    elif trend_freq == "Weekly":
        grouper = df_tool_processed.groupby(df_tool_processed['shot_time'].dt.to_period('W'))
        period_name = "Week"
    else:
        grouper = df_tool_processed.groupby(df_tool_processed['shot_time'].dt.to_period('M'))
        period_name = "Month"

    for period, df_period in grouper:
        if df_period.empty: continue
        
        # Calculate metrics for this specific chunk of time using aggregation logic
        # Note: run_id already exists in df_period, so calculate_run_summaries will work
        run_breakdown_df = cr_utils.calculate_run_summaries(df_period, config)
        if run_breakdown_df.empty: continue

        total_runtime = run_breakdown_df['total_runtime_sec'].sum()
        prod_time = run_breakdown_df['production_time_sec'].sum()
        downtime = run_breakdown_df['downtime_sec'].sum()
        
        total_shots = run_breakdown_df['total_shots'].sum()
        normal_shots = run_breakdown_df['normal_shots'].sum()
        stops = run_breakdown_df['stop_events'].sum()
        
        act_output = run_breakdown_df['actual_output_parts'].sum()
        opt_output = run_breakdown_df['optimal_output_parts'].sum()

        eff_rate = (normal_shots / total_shots) * 100 if total_shots > 0 else 0
        stab_index = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
        
        mttr = (downtime / 60 / stops) if stops > 0 else 0
        mtbf = (prod_time / 60 / stops) if stops > 0 else (prod_time/60)

        # Format Label
        if trend_freq == "Daily": label = period.strftime('%Y-%m-%d')
        elif trend_freq == "Weekly": label = f"W{period.week} {period.year}"
        else: label = period.strftime('%B %Y')

        trend_data.append({
            period_name: label,
            'SortKey': period if trend_freq == "Daily" else period.start_time,
            'Stability Index (%)': stab_index,
            'Efficiency (%)': eff_rate, 
            'Actual Output': act_output,
            'Optimal Output': opt_output,
            'MTTR (min)': mttr,
            'MTBF (min)': mtbf,
            'Production Time (h)': prod_time / 3600
        })

    if not trend_data:
        st.warning("No trend data available.")
        return

    df_trends = pd.DataFrame(trend_data).sort_values('SortKey', ascending=False).drop(columns=['SortKey'])

    # 1. Trend Table
    st.dataframe(
        df_trends.style.format({
            'Stability Index (%)': '{:.1f}',
            'Efficiency (%)': '{:.1f}',
            'MTTR (min)': '{:.1f}',
            'MTBF (min)': '{:.1f}',
            'Actual Output': '{:,.0f}',
            'Optimal Output': '{:,.0f}',
            'Production Time (h)': '{:.1f}'
        }).background_gradient(subset=['Stability Index (%)'], cmap='RdYlGn', vmin=0, vmax=100),
        use_container_width=True
    )

    # 2. Visual Chart
    st.subheader("Visual Trend")
    metric_to_plot = st.selectbox("Select Metric to Visualize", 
                                  ['Stability Index (%)', 'Actual Output', 'Efficiency (%)', 'MTTR (min)'],
                                  key="cr_trend_viz_select")
    
    fig = px.line(df_trends.sort_index(ascending=False), x=period_name, y=metric_to_plot, markers=True, title=f"{metric_to_plot} Trend")
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard(df_tool, tool_name, config):
    """Renders the Main Capacity Dashboard (Tab 2)."""

    # --- 1. Analysis Level & Filter Controls ---
    analysis_level = st.radio(
        "Select Analysis Level",
        options=["Daily (by Run)", "Weekly (by Run)", "Monthly (by Run)", "Custom Period"],
        horizontal=True,
        key="cr_analysis_level"
    )

    st.markdown("---")

    # --- 2. Data Processing & Run Filtering ---
    # Pre-process to get Runs
    base_calc = cr_utils.CapacityRiskCalculator(df_tool, **config)
    df_processed = base_calc.results.get('processed_df', pd.DataFrame())
    
    if df_processed.empty:
        st.error("No data available for processing."); return

    # Filter Logic (Toggle)
    enable_filter = st.toggle("Filter Small Production Runs", value=False, key="cr_filter_runs")
    min_shots_filter = 1
    
    if enable_filter:
        run_counts = df_processed.groupby('run_id').size()
        max_shots = int(run_counts.max()) if not run_counts.empty else 100
        default_val = min(10, max_shots)
        min_shots_filter = st.slider("Remove Runs with Fewer Than X Shots", 1, max_shots, default_val, key="cr_min_shots")

    # --- 3. Date/Period Selection ---
    df_view = pd.DataFrame()
    sub_header = ""

    if "Daily" in analysis_level:
        dates = sorted(df_processed['shot_time'].dt.date.unique())
        if not dates: st.warning("No dates found."); return
        sel_date = st.selectbox("Select Date", dates, index=len(dates)-1, format_func=lambda x: x.strftime('%d %b %Y'), key="cr_date_select")
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Summary for (Combined Runs) {sel_date.strftime('%d %b %Y')}"
    
    elif "Weekly" in analysis_level:
        df_processed['week_lbl'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week_lbl'].unique())
        sel_week = st.selectbox("Select Week", weeks, index=len(weeks)-1, key="cr_week_select")
        df_view = df_processed[df_processed['week_lbl'] == sel_week]
        sub_header = f"Summary for (Combined Runs) {sel_week}"

    elif "Monthly" in analysis_level:
        df_processed['month_lbl'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month_lbl'].unique())
        sel_month = st.selectbox("Select Month", months, index=len(months)-1, format_func=lambda x: x.strftime('%B %Y'), key="cr_month_select")
        df_view = df_processed[df_processed['month_lbl'] == sel_month]
        sub_header = f"Summary for (Combined Runs) {sel_month.strftime('%B %Y')}"

    else: # Custom
        d_min = df_processed['shot_time'].min().date()
        d_max = df_processed['shot_time'].max().date()
        c1, c2 = st.columns(2)
        s_date = c1.date_input("Start Date", d_min)
        e_date = c2.date_input("End Date", d_max)
        if s_date and e_date:
            df_view = df_processed[(df_processed['shot_time'].dt.date >= s_date) & (df_processed['shot_time'].dt.date <= e_date)]
            sub_header = f"Summary for (Combined Runs) {s_date} to {e_date}"

    # Apply Small Run Filter to the View
    if enable_filter and not df_view.empty:
        run_counts_view = df_view.groupby('run_id')['run_id'].transform('count')
        df_view = df_view[run_counts_view >= min_shots_filter]

    if df_view.empty:
        st.warning("No data found for the selected criteria.")
        return

    # --- 4. Main Title & Summary Header ---
    st.markdown(f"### {tool_name} Overview")
    st.subheader(sub_header)

    # --- 5. Calculate Metrics (AGGREGATED from Runs) ---
    run_breakdown_df = cr_utils.calculate_run_summaries(df_view, config)
    
    if run_breakdown_df.empty:
        st.warning("No valid runs found in selected period.")
        return

    # --- Aggregation Logic (The "Single Source of Truth") ---
    total_runtime = run_breakdown_df['total_runtime_sec'].sum()
    prod_time = run_breakdown_df['production_time_sec'].sum()
    downtime = run_breakdown_df['downtime_sec'].sum()
    total_cap_loss_sec = run_breakdown_df['total_capacity_loss_sec'].sum()
    
    total_shots = run_breakdown_df['total_shots'].sum()
    normal_shots = run_breakdown_df['normal_shots'].sum()
    stop_events = run_breakdown_df['stop_events'].sum()
    
    opt_output = run_breakdown_df['optimal_output_parts'].sum()
    
    # FIX: Calculate Target Output from Optimal if column missing to avoid KeyError
    if 'target_output_parts' in run_breakdown_df.columns:
        tgt_output = run_breakdown_df['target_output_parts'].sum() 
    else:
        tgt_output = opt_output * (config['target_output_perc'] / 100.0)

    act_output = run_breakdown_df['actual_output_parts'].sum()
    
    # Capacity Losses
    loss_downtime = run_breakdown_df['capacity_loss_downtime_parts'].sum()
    loss_slow = run_breakdown_df['capacity_loss_slow_parts'].sum()
    gain_fast = run_breakdown_df['capacity_gain_fast_parts'].sum()
    total_loss_parts = run_breakdown_df['total_capacity_loss_parts'].sum()

    # Derived Metrics
    eff_rate = (normal_shots / total_shots) if total_shots > 0 else 0
    stab_index = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
    true_loss = res['optimal_output_parts'] - res['actual_output_parts'] if 'optimal_output_parts' in locals() else opt_output - act_output
    
    # Construct Results Dict for Display
    res = {
        'total_runtime_sec': total_runtime,
        'production_time_sec': prod_time,
        'downtime_sec': downtime,
        'total_capacity_loss_sec': total_cap_loss_sec,
        'efficiency_rate': eff_rate,
        'stability_index': stab_index,
        'optimal_output_parts': opt_output,
        'target_output_parts': tgt_output, 
        'actual_output_parts': act_output,
        'total_shots': total_shots,
        'normal_shots': normal_shots,
        'capacity_loss_downtime_parts': loss_downtime,
        'capacity_loss_slow_parts': loss_slow,
        'capacity_gain_fast_parts': gain_fast,
        'total_capacity_loss_parts': total_loss_parts,
        'processed_df': df_view # Use the view directly for charts (preserves global flags)
    }
    
    # --- 6. KPI BOARDS ---

    # --- KPI Board 1: Run Rate Metrics (Time & Stability) ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        
        # 1. Total Run Duration
        k1.metric("Total Run Duration", cr_utils.format_seconds_to_dhm(res['total_runtime_sec']))
        
        # 2. Production Time (with %)
        prod_perc = (res['production_time_sec'] / res['total_runtime_sec'] * 100) if res['total_runtime_sec'] > 0 else 0
        k2.metric("Production Time", cr_utils.format_seconds_to_dhm(res['production_time_sec']))
        k2.markdown(f"<span style='background-color:{cr_utils.PASTEL_COLORS['green']}; color:black; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.8em'>{prod_perc:.1f}%</span>", unsafe_allow_html=True)
        
        # 3. Downtime (with %)
        down_perc = (res['downtime_sec'] / res['total_runtime_sec'] * 100) if res['total_runtime_sec'] > 0 else 0
        k3.metric("Run Rate Downtime", cr_utils.format_seconds_to_dhm(res['downtime_sec']))
        k3.markdown(f"<span style='background-color:{cr_utils.PASTEL_COLORS['red']}; color:black; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.8em'>{down_perc:.1f}%</span>", unsafe_allow_html=True)

        # 4. Efficiency (%) - Metric Only
        k4.metric("Run Rate Efficiency (%)", f"{res['efficiency_rate']*100:.1f}%")
        
        # 5. Stability Index (%) - Metric Only
        k5.metric("Run Rate Stability Index (%)", f"{res['stability_index']:.1f}%")

    # --- Donut Charts Row (Capacity & Shots) ---
    with st.container(border=True):
        c1, c2 = st.columns(2)
        
        # Chart 1: Actual vs Optimal Output
        with c1:
            act_perc = (res['actual_output_parts'] / res['optimal_output_parts'] * 100) if res['optimal_output_parts'] > 0 else 0
            st.plotly_chart(cr_utils.create_donut_chart(act_perc, f"Actual Output ({res['actual_output_parts']:,.0f}) vs Optimal ({res['optimal_output_parts']:,.0f})", color_scheme='blue'), use_container_width=True)
            
        # Chart 2: Normal vs Total Shots
        with c2:
            norm_perc = (res['normal_shots'] / res['total_shots'] * 100) if res['total_shots'] > 0 else 0
            st.plotly_chart(cr_utils.create_donut_chart(norm_perc, f"Normal Shots ({res['normal_shots']:,.0f}) vs Total ({res['total_shots']:,.0f})", color_scheme='green'), use_container_width=True)

    # --- MOVED: Overall Totals (Production Totals) ---
    st.subheader("Overall Totals")
    with st.container(border=True):
        pt1, pt2, pt3, pt4 = st.columns(4)
        pt1.metric("Total Actual Output", f"{res['actual_output_parts']:,.0f}")
        pt2.metric("Total Optimal Output", f"{res['optimal_output_parts']:,.0f}")
        pt3.metric("Net Loss", f"{true_loss:,.0f} parts")
        pt4.metric("Total Shots", f"{res['total_shots']:,.0f}")

    # --- KPI Board 3: Cycle Time Metrics ---
    with st.container(border=True):
        ct1, ct2, ct3, ct4 = st.columns(4)
        
        # 1. Approved CT
        avg_app_ct = df_view['approved_ct'].mean()
        ct1.metric("Approved Cycle Time", f"{avg_app_ct:.2f} s")
        
        # Metrics for limits (Using aggregated ranges from run breakdown)
        min_mode = run_breakdown_df['mode_ct'].min()
        max_mode = run_breakdown_df['mode_ct'].max()
        min_lower = run_breakdown_df['mode_lower'].min()
        max_upper = run_breakdown_df['mode_upper'].max()

        # 2. Lower Limit
        ct2.metric("Lower Limit Range", f"{min_lower:.2f} - {run_breakdown_df['mode_lower'].max():.2f} s")
        
        # 3. Mode CT
        ct3.metric("Mode CT Range", f"{min_mode:.2f} - {max_mode:.2f} s")
        
        # 4. Upper Limit
        ct4.metric("Upper Limit Range", f"{run_breakdown_df['mode_upper'].min():.2f} - {max_upper:.2f} s")

    # --- What do these metrics mean? (Collapsed) ---
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        **Run Rate Metrics:**
        - **Run Rate Efficiency (%)**: Percentage of shots that are "Normal" (not stops). *Formula: Normal Shots / Total Shots*
        - **Run Rate Stability Index (%)**: Percentage of total run time spent producing. *Formula: Production Time / Total Run Duration*
        - **Total Run Duration**: Sum of all individual production runs (excludes large gaps like weekends).
        - **Downtime**: Total Run Duration minus Production Time.

        **Capacity Metrics:**
        - **Optimal Output**: Theoretical maximum parts possible if running at Approved Cycle Time for the entire duration.
        - **Actual Output**: The actual number of parts produced (Total Shots x Cavities).
        - **Capacity Loss**: The difference between Optimal and Actual output, broken down by cause (Downtime vs Slow Cycles).
        """)

    st.markdown("---")

    # --- 7. Lower Section: Breakdown & Charts ---
    # Re-ordering: Waterfall -> Cycle Time Chart -> Run Breakdown -> Production Totals
    
    # A. Capacity Loss Waterfall
    st.subheader("Capacity Loss Waterfall")
    c_chart, c_details = st.columns([1.5, 1]) 
    
    with c_chart:
        # FIX: Ensure res contains all keys needed for plot_waterfall
        st.plotly_chart(cr_utils.plot_waterfall(res, "Optimal Output"), use_container_width=True)
        
    with c_details:
        
        with st.container(border=True):
            st.markdown("**Total Net Impact**")
            st.markdown(f"<h2 style='color:#ff6961; margin:0;'>{true_loss:,.0f} parts</h2>", unsafe_allow_html=True)
            st.caption(f"Net Time Lost: {cr_utils.format_seconds_to_dhm(res['total_capacity_loss_sec'])}")
        
        net_cycle_loss = res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts']
        
        # Construct the table data
        breakdown_data = [
            {"Metric": "Loss (RR Downtime)", "Parts": res['capacity_loss_downtime_parts'], "Time": cr_utils.format_seconds_to_dhm(res['downtime_sec'])},
            {"Metric": "Net Loss (Cycle Time)", "Parts": net_cycle_loss, "Time": "N/A"},
            {"Metric": "└ Loss (Slow Cycles)", "Parts": res['capacity_loss_slow_parts'], "Time": cr_utils.format_seconds_to_dhm(res['capacity_loss_slow_parts'] * (res['production_time_sec']/res['actual_output_parts']) if res['actual_output_parts'] else 0)},
            {"Metric": "└ Gain (Fast Cycles)", "Parts": res['capacity_gain_fast_parts'], "Time": cr_utils.format_seconds_to_dhm(res['capacity_gain_fast_parts'] * (res['production_time_sec']/res['actual_output_parts']) if res['actual_output_parts'] else 0)},
        ]
        
        df_breakdown = pd.DataFrame(breakdown_data)
        
        def style_breakdown(row):
            styles = [''] * len(row)
            if row['Metric'] == "Loss (RR Downtime)":
                styles[1] = 'color: #ff6961; font-weight: bold;'
            elif row['Metric'] == "Net Loss (Cycle Time)":
                color = '#ff6961' if row['Parts'] > 0 else '#77dd77'
                styles[1] = f'color: {color}; font-weight: bold;'
            elif "Gain" in row['Metric']:
                styles[1] = 'color: #77dd77;'
            elif "Loss" in row['Metric']:
                styles[1] = 'color: #ff6961;'
            return styles

        st.dataframe(
            df_breakdown.style.apply(style_breakdown, axis=1).format({"Parts": "{:,.0f}"}), 
            use_container_width=True, 
            hide_index=True
        )

    st.markdown("---")

    # B. Shot Analysis (Cycle Time Graph)
    st.subheader("Shot-by-Shot Visualization")
    st.plotly_chart(cr_utils.plot_shot_analysis(res['processed_df']), use_container_width=True)
    with st.expander("View Shot Data Table (Collapsed)", expanded=False):
        st.dataframe(res['processed_df'][['shot_time', 'actual_ct', 'run_id', 'shot_type', 'stop_flag']], use_container_width=True)

    st.markdown("---")

    # C. Run-Based Analysis & Performance Breakdown
    st.header("Performance Analysis")
    
    # --- Added Dropdown for View Control ---
    freq_options = ['by Run', 'Hourly', 'Daily', 'Weekly', 'Monthly']
    default_ix = 2 if 'Weekly' in analysis_level or 'Monthly' in analysis_level else 0
    if 'Daily' in analysis_level: default_ix = 1 # Default to Hourly for Daily view
    
    selected_freq = st.selectbox("View Analysis By", freq_options, index=default_ix, key='perf_breakdown_freq')

    # 1. Performance Breakdown Chart (Stacked Bar)
    st.subheader(f"Performance Breakdown ({selected_freq})")
    
    # Aggregate data for bar chart - Reuse existing function but pass config
    agg_df_chart = cr_utils.get_aggregated_data(df_view, selected_freq, config)
    
    if not agg_df_chart.empty:
        st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df_chart, 'Period', "Optimal Output"), use_container_width=True)

        st.markdown("---")

        # --- RE-INTRODUCED: Production Totals Report ---
        st.header(f"Production Totals Report ({selected_freq})")
        
        # Calculate derived columns for the table
        totals_df = agg_df_chart.copy()
        
        # Formatting helper
        totals_df['Actual Production Time'] = totals_df.apply(
            lambda r: f"{cr_utils.format_seconds_to_dhm(r['Production Time Sec'])} ({r['Production Time Sec']/r['Run Time Sec']:.1%})" if r['Run Time Sec'] > 0 else "0m (0.0%)", 
            axis=1
        )
        totals_df['Production Shots (Pct)'] = totals_df.apply(
            lambda r: f"{r['Normal Shots']:,.0f} ({r['Normal Shots']/r['Total Shots']:.1%})" if r['Total Shots'] > 0 else "0 (0.0%)", 
            axis=1
        )
        
        # Select and Rename Columns
        totals_table = pd.DataFrame()
        totals_table['Period'] = totals_df['Period']
        totals_table['Overall Run Time'] = totals_df['Run Time'] + " (" + totals_df['Run Time Sec'].apply(lambda x: f"{x:.0f}s") + ")"
        totals_table['Actual Production Time'] = totals_df['Actual Production Time']
        totals_table['Total Shots'] = totals_df['Total Shots'].map('{:,.0f}'.format)
        totals_table['Production Shots'] = totals_df['Production Shots (Pct)']
        totals_table['Downtime Shots'] = totals_df['Downtime Shots'].map('{:,.0f}'.format)
        
        st.dataframe(totals_table, use_container_width=True, hide_index=True)

        # --- RE-INTRODUCED: Capacity Loss & Gain Report ---
        st.header(f"Capacity Loss & Gain Report (vs Optimal) ({selected_freq})")
        
        # Calculate derived columns
        loss_gain_df = agg_df_chart.copy()
        
        # Formatting for Table
        lg_table = pd.DataFrame()
        lg_table['Period'] = loss_gain_df['Period']
        lg_table['Optimal Output'] = loss_gain_df['Optimal Output'].map('{:,.2f}'.format)
        
        lg_table['Actual Output'] = loss_gain_df.apply(
            lambda r: f"{r['Actual Output']:,.2f} ({r['Actual Output']/r['Optimal Output']:.1%})" if r['Optimal Output'] > 0 else "0.00 (0.0%)", 
            axis=1
        )
        lg_table['Loss (RR Downtime)'] = loss_gain_df.apply(
            lambda r: f"{r['Downtime Loss']:,.2f} ({r['Downtime Loss']/r['Optimal Output']:.1%})" if r['Optimal Output'] > 0 else "0.00 (0.0%)", 
            axis=1
        )
        lg_table['Loss (Slow Cycles)'] = loss_gain_df.apply(
            lambda r: f"{r['Slow Loss']:,.2f} ({r['Slow Loss']/r['Optimal Output']:.1%})" if r['Optimal Output'] > 0 else "0.00 (0.0%)", 
            axis=1
        )
        lg_table['Gain (Fast Cycles)'] = loss_gain_df.apply(
            lambda r: f"{r['Fast Gain']:,.2f} ({r['Fast Gain']/r['Optimal Output']:.1%})" if r['Optimal Output'] > 0 else "0.00 (0.0%)", 
            axis=1
        )
        lg_table['Total Net Loss'] = loss_gain_df.apply(
            lambda r: f"{r['Total Loss']:,.2f} ({r['Total Loss']/r['Optimal Output']:.1%})" if r['Optimal Output'] > 0 else "0.00 (0.0%)", 
            axis=1
        )

        # Styling Logic
        def style_loss_gain(col):
            col_name = col.name
            if col_name == 'Loss (RR Downtime)': return ['color: #ff6961'] * len(col) # Red
            if col_name == 'Loss (Slow Cycles)': return ['color: #ffb347'] * len(col) # Orange/Red
            if col_name == 'Gain (Fast Cycles)': return ['color: #77dd77'] * len(col) # Green
            if col_name == 'Total Net Loss':
                # Check underlying numeric value from the source DF
                # This is tricky in Pandas styling on a different DF. 
                # Simplified: Just make it bold/colored based on string inspection or assumption.
                return ['font-weight: bold'] * len(col)
            return [''] * len(col)

        st.dataframe(
            lg_table.style.apply(style_loss_gain, axis=0),
            use_container_width=True,
            hide_index=True
        )

    
    # 2. Run Breakdown Table
    with st.expander("View Run Breakdown Table (Details)", expanded=False):
        # Create a display copy
        d_df = run_breakdown_df.copy()
        
        # --- Format Columns ---
        d_df = d_df.sort_values('start_time').reset_index(drop=True)
        d_df['RUN ID'] = d_df.index.map(lambda x: f"Run {x+1:03d}")
        
        d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
        
        d_df["Normal Shots"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
        
        d_df["Stop Events"] = d_df.apply(lambda r: f"{r['stop_events']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
        
        d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(cr_utils.format_seconds_to_dhm)
        
        d_df["Production Time (d/h/m)"] = d_df.apply(lambda r: f"{cr_utils.format_seconds_to_dhm(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
        
        d_df["Downtime (d/h/m)"] = d_df.apply(lambda r: f"{cr_utils.format_seconds_to_dhm(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
        
        # Rename for display
        rename_map = {
            'total_shots': 'Total shots',
            'mode_ct': 'Mode CT (for the run)',
            'lower_limit': 'Lower limit CT (sec)',
            'upper_limit': 'Upper Limit CT (sec)'
        }
        d_df.rename(columns=rename_map, inplace=True)
        
        # Select Final Columns
        cols_order = [
            'RUN ID', 'Period (date/time from to)', 'Total shots', 'Normal Shots', 'Stop Events',
            'Mode CT (for the run)', 'Lower limit CT (sec)', 'Upper Limit CT (sec)',
            'Total Run duration (d/h/m)', 'Production Time (d/h/m)', 'Downtime (d/h/m)'
        ]
        
        st.dataframe(
            d_df[cols_order].style.format({
                'Mode CT (for the run)': '{:.2f}',
                'Lower limit CT (sec)': '{:.2f}',
                'Upper Limit CT (sec)': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")


# ==============================================================================
# --- MAIN ENTRY POINT ---
# ==============================================================================

def main():
    st.sidebar.title("Capacity Risk v9.8")
    
    # --- File Upload ---
    st.sidebar.markdown("### File Upload")
    files = st.sidebar.file_uploader("Upload Data (Excel/CSV)", accept_multiple_files=True, type=['xlsx', 'csv', 'xls'])
    
    if not files:
        st.info("👈 Upload files to begin.")
        st.stop()
        
    df_all = cr_utils.load_all_data_cr(files)
    if df_all.empty:
        st.error("No valid data found. Please check your file columns.")
        st.stop()

    # --- Tool Selection ---
    tool_ids = sorted(df_all['tool_id'].unique())
    selected_tool = st.sidebar.selectbox("Select Tool ID", tool_ids)

    # --- Analysis Parameters ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analysis Parameters ⚙️")
    
    with st.sidebar.expander("Configure Metrics", expanded=True):
        tolerance = st.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.05, 0.01)
        downtime_gap_tolerance = st.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5)
        run_interval_hours = st.slider("Run Interval Threshold (hours)", 1, 24, 8, 1)

    with st.sidebar.expander("Capacity Settings", expanded=False):
        target_output_perc = st.slider("Target Output %", 50, 100, 90)
        default_cavities = st.number_input("Default Cavities", value=1, min_value=1)
        remove_maint = st.checkbox("Remove Maintenance/Warehouse", value=False)

    # Config Dictionary
    config = {
        'target_output_perc': target_output_perc,
        'tolerance': tolerance,
        'downtime_gap_tolerance': downtime_gap_tolerance,
        'run_interval_hours': run_interval_hours,
        'default_cavities': default_cavities,
        'remove_maintenance': remove_maint
    }
    
    # Placeholder Demand Info (could be expanded later)
    # demand_info = {'target': 0, 'received': 0}

    # --- Filter Data for Selected Tool ---
    df_tool = df_all[df_all['tool_id'] == selected_tool]

    # --- TABS: Order Updated (Risk Tower -> Dashboard -> Trends) ---
    t_risk, t_dash, t_trend = st.tabs(["Risk Tower", "Capacity Dashboard", "Trends"])

    with t_risk:
        render_risk_tower(df_all, config)
    
    with t_dash:
        if not df_tool.empty:
            render_dashboard(df_tool, selected_tool, config)
        else:
            st.warning("No data for selected tool.")

    with t_trend:
        if not df_tool.empty:
            render_trends_tab(df_tool, config)

if __name__ == "__main__":
    main()