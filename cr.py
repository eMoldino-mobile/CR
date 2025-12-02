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
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v9.95)")

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
    """Renders the Trends Tab (Tab 4)."""
    st.header("Historical Performance Trends")
    st.info("Trends are calculated using 'Run-Based' logic consistent with the Dashboard.")

    col_ctrl, _ = st.columns([1, 3])
    with col_ctrl:
        trend_freq = st.selectbox("Select Trend Frequency", ["Daily", "Weekly", "Monthly"], key="cr_trend_freq")

    base_calc = cr_utils.CapacityRiskCalculator(df_tool, **config)
    df_tool_processed = base_calc.results['processed_df']
    
    if df_tool_processed.empty:
        st.warning("No data available after processing.")
        return

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
        
        tgt_output = run_breakdown_df['target_output_parts'].sum() if 'target_output_parts' in run_breakdown_df.columns else (opt_output * (config['target_output_perc'] / 100.0))

        eff_rate = (normal_shots / total_shots) * 100 if total_shots > 0 else 0
        stab_index = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
        
        mttr = (downtime / 60 / stops) if stops > 0 else 0
        mtbf = (prod_time / 60 / stops) if stops > 0 else (prod_time/60)

        target_met_perc = (act_output / tgt_output * 100) if tgt_output > 0 else 0
        optimal_met_perc = (act_output / opt_output * 100) if opt_output > 0 else 0

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
            'Target Output': tgt_output,
            'Target Met (%)': target_met_perc,
            'Optimal Met (%)': optimal_met_perc,
            'MTTR (min)': mttr,
            'MTBF (min)': mtbf,
            'Production Time (h)': prod_time / 3600
        })

    if not trend_data:
        st.warning("No trend data available.")
        return

    df_trends = pd.DataFrame(trend_data).sort_values('SortKey', ascending=False).drop(columns=['SortKey'])

    st.dataframe(
        df_trends.style.format({
            'Stability Index (%)': '{:.1f}',
            'Efficiency (%)': '{:.1f}',
            'MTTR (min)': '{:.1f}',
            'MTBF (min)': '{:.1f}',
            'Actual Output': '{:,.0f}',
            'Optimal Output': '{:,.0f}',
            'Target Output': '{:,.0f}',
            'Target Met (%)': '{:.1f}%',
            'Optimal Met (%)': '{:.1f}%',
            'Production Time (h)': '{:.1f}'
        })
        .background_gradient(subset=['Target Met (%)'], cmap='RdYlGn', vmin=50, vmax=110)
        .background_gradient(subset=['Optimal Met (%)'], cmap='RdYlGn', vmin=50, vmax=110),
        use_container_width=True
    )

    st.subheader("Visual Trend")
    metric_to_plot = st.selectbox("Select Metric to Visualize", 
                                  ['Target Met (%)', 'Optimal Met (%)', 'Stability Index (%)', 'Actual Output', 'Efficiency (%)', 'MTTR (min)'],
                                  key="cr_trend_viz_select")
    
    fig = px.line(df_trends.sort_index(ascending=False), x=period_name, y=metric_to_plot, markers=True, title=f"{metric_to_plot} Trend")
    
    if "Met (%)" in metric_to_plot:
        fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="100% Target")
    
    st.plotly_chart(fig, use_container_width=True)


def render_dashboard(df_tool, tool_name, config, dashboard_mode="Optimal"):
    """
    Renders the Main Capacity Dashboard.
    dashboard_mode: 'Optimal' or 'Target' - controls benchmark view and table visibility.
    """
    
    benchmark_mode = "Optimal Output" if dashboard_mode == "Optimal" else "Target Output"
    key_suffix = f"_{dashboard_mode.lower()}" # Unique key suffix for widgets

    # --- 1. Analysis Level & Filter Controls ---
    c1, c2 = st.columns([2, 1])
    with c1:
        analysis_level = st.radio(
            f"Select Analysis Level ({dashboard_mode})",
            options=["Daily (by Run)", "Weekly (by Run)", "Monthly (by Run)", "Custom Period"],
            horizontal=True,
            key=f"cr_analysis_level{key_suffix}"
        )
    with c2:
        # Filter Logic (Toggle)
        enable_filter = st.toggle("Filter Small Runs", value=False, key=f"cr_filter_runs{key_suffix}")
        min_shots_filter = 1
        if enable_filter:
            min_shots_filter = st.number_input("Min Shots per Run", 1, 1000, 10, key=f"cr_min_shots{key_suffix}")

    st.markdown("---")

    # --- 2. Data Processing & Run Filtering ---
    base_calc = cr_utils.CapacityRiskCalculator(df_tool, **config)
    df_processed = base_calc.results.get('processed_df', pd.DataFrame())
    
    if df_processed.empty:
        st.error("No data available for processing."); return

    if enable_filter:
        run_counts = df_processed.groupby('run_id')['run_id'].transform('count')
        df_processed = df_processed[run_counts >= min_shots_filter]

    # --- 3. Date/Period Selection ---
    df_view = pd.DataFrame()
    sub_header = ""

    if "Daily" in analysis_level:
        dates = sorted(df_processed['shot_time'].dt.date.unique())
        if not dates: st.warning("No dates found."); return
        sel_date = st.selectbox("Select Date", dates, index=len(dates)-1, format_func=lambda x: x.strftime('%d %b %Y'), key=f"cr_date_select{key_suffix}")
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Summary for (Combined Runs) {sel_date.strftime('%d %b %Y')}"
    
    elif "Weekly" in analysis_level:
        df_processed['week_lbl'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week_lbl'].unique())
        sel_week = st.selectbox("Select Week", weeks, index=len(weeks)-1, key=f"cr_week_select{key_suffix}")
        df_view = df_processed[df_processed['week_lbl'] == sel_week]
        sub_header = f"Summary for (Combined Runs) {sel_week}"

    elif "Monthly" in analysis_level:
        df_processed['month_lbl'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month_lbl'].unique())
        sel_month = st.selectbox("Select Month", months, index=len(months)-1, format_func=lambda x: x.strftime('%B %Y'), key=f"cr_month_select{key_suffix}")
        df_view = df_processed[df_processed['month_lbl'] == sel_month]
        sub_header = f"Summary for (Combined Runs) {sel_month.strftime('%B %Y')}"

    else: # Custom
        d_min = df_processed['shot_time'].min().date()
        d_max = df_processed['shot_time'].max().date()
        c1, c2 = st.columns(2)
        s_date = c1.date_input("Start Date", d_min, key=f"d1{key_suffix}")
        e_date = c2.date_input("End Date", d_max, key=f"d2{key_suffix}")
        if s_date and e_date:
            df_view = df_processed[(df_processed['shot_time'].dt.date >= s_date) & (df_processed['shot_time'].dt.date <= e_date)]
            sub_header = f"Summary for (Combined Runs) {s_date} to {e_date}"

    if df_view.empty:
        st.warning("No data found for the selected criteria.")
        return

    # --- 4. Main Title & Summary Header ---
    st.markdown(f"### {tool_name} Overview - {dashboard_mode} View")
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
    
    opt_output = run_breakdown_df['optimal_output_parts'].sum()
    
    tgt_output = run_breakdown_df['target_output_parts'].sum() if 'target_output_parts' in run_breakdown_df.columns else (opt_output * (config['target_output_perc'] / 100.0))

    act_output = run_breakdown_df['actual_output_parts'].sum()
    
    loss_downtime = run_breakdown_df['capacity_loss_downtime_parts'].sum()
    loss_slow = run_breakdown_df['capacity_loss_slow_parts'].sum()
    gain_fast = run_breakdown_df['capacity_gain_fast_parts'].sum()
    total_loss_parts = run_breakdown_df['total_capacity_loss_parts'].sum()

    eff_rate = (normal_shots / total_shots) if total_shots > 0 else 0
    stab_index = (prod_time / total_runtime * 100) if total_runtime > 0 else 0
    
    # Calculate True Net Loss based on Benchmark
    if dashboard_mode == "Target":
        true_loss = max(0, tgt_output - act_output)
        benchmark_output = tgt_output
    else:
        true_loss = opt_output - act_output
        benchmark_output = opt_output
    
    # Construct Results Dict
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
        'processed_df': df_view 
    }
    
    # --- 6. KPI BOARDS ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Run Duration", cr_utils.format_seconds_to_dhm(res['total_runtime_sec']))
        prod_perc = (res['production_time_sec'] / res['total_runtime_sec'] * 100) if res['total_runtime_sec'] > 0 else 0
        k2.metric("Production Time", cr_utils.format_seconds_to_dhm(res['production_time_sec']))
        k2.markdown(f"<span style='background-color:{cr_utils.PASTEL_COLORS['green']}; color:black; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.8em'>{prod_perc:.1f}%</span>", unsafe_allow_html=True)
        down_perc = (res['downtime_sec'] / res['total_runtime_sec'] * 100) if res['total_runtime_sec'] > 0 else 0
        k3.metric("Run Rate Downtime", cr_utils.format_seconds_to_dhm(res['downtime_sec']))
        k3.markdown(f"<span style='background-color:{cr_utils.PASTEL_COLORS['red']}; color:black; padding:2px 6px; border-radius:4px; font-weight:bold; font-size:0.8em'>{down_perc:.1f}%</span>", unsafe_allow_html=True)
        k4.metric("Run Rate Efficiency (%)", f"{res['efficiency_rate']*100:.1f}%")
        k5.metric("Run Rate Stability Index (%)", f"{res['stability_index']:.1f}%")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            act_perc = (res['actual_output_parts'] / benchmark_output * 100) if benchmark_output > 0 else 0
            st.plotly_chart(cr_utils.create_donut_chart(act_perc, f"Actual ({res['actual_output_parts']:,.0f}) vs {dashboard_mode} ({benchmark_output:,.0f})", color_scheme='blue'), use_container_width=True)
        with c2:
            norm_perc = (res['normal_shots'] / res['total_shots'] * 100) if res['total_shots'] > 0 else 0
            st.plotly_chart(cr_utils.create_donut_chart(norm_perc, f"Normal Shots ({res['normal_shots']:,.0f}) vs Total ({res['total_shots']:,.0f})", color_scheme='green'), use_container_width=True)

    st.subheader("Overall Totals")
    with st.container(border=True):
        pt1, pt2, pt3, pt4 = st.columns(4)
        pt1.metric("Total Actual Output", f"{res['actual_output_parts']:,.0f}")
        pt2.metric(f"Total {dashboard_mode} Output", f"{benchmark_output:,.0f}")
        pt3.metric(f"Net Loss (vs {dashboard_mode})", f"{true_loss:,.0f} parts")
        pt4.metric("Total Shots", f"{res['total_shots']:,.0f}")

    with st.container(border=True):
        ct1, ct2, ct3, ct4 = st.columns(4)
        avg_app_ct = df_view['approved_ct'].mean()
        ct1.metric("Approved Cycle Time", f"{avg_app_ct:.2f} s")
        min_mode = run_breakdown_df['mode_ct'].min()
        max_mode = run_breakdown_df['mode_ct'].max()
        min_lower = run_breakdown_df['mode_lower'].min()
        max_upper = run_breakdown_df['mode_upper'].max()
        ct2.metric("Lower Limit Range", f"{min_lower:.2f} - {run_breakdown_df['mode_lower'].max():.2f} s")
        ct3.metric("Mode CT Range", f"{min_mode:.2f} - {max_mode:.2f} s")
        ct4.metric("Upper Limit Range", f"{run_breakdown_df['mode_upper'].min():.2f} - {max_upper:.2f} s")

    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        **Run Rate Metrics:**
        - **Run Rate Efficiency (%)**: Normal Shots / Total Shots
        - **Run Rate Stability Index (%)**: Production Time / Total Run Duration
        - **Total Run Duration**: Sum of all individual production runs.
        
        **Capacity Metrics:**
        - **Optimal Output**: Theoretical max at Approved Cycle Time (100%).
        - **Target Output**: Your configured target percentage of Optimal.
        """)

    st.markdown("---")

    # --- 7. Lower Section: Breakdown & Charts ---
    st.subheader(f"Capacity Loss Waterfall (vs {dashboard_mode})")
    c_chart, c_details = st.columns([1.5, 1]) 
    
    with c_chart:
        st.plotly_chart(cr_utils.plot_waterfall(res, benchmark_mode), use_container_width=True)
        
    with c_details:
        with st.container(border=True):
            st.markdown(f"**Total Net Impact (vs {dashboard_mode})**")
            st.markdown(f"<h2 style='color:#ff6961; margin:0;'>{true_loss:,.0f} parts</h2>", unsafe_allow_html=True)
            if dashboard_mode == "Optimal":
                st.caption(f"Net Time Lost: {cr_utils.format_seconds_to_dhm(res['total_capacity_loss_sec'])}")
        
        # Breakdown table (Standard logic is vs Optimal, so we keep that consistent in right panel for details)
        net_cycle_loss = res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts']
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

    st.subheader("Shot-by-Shot Visualization")
    st.plotly_chart(cr_utils.plot_shot_analysis(res['processed_df']), use_container_width=True)
    with st.expander("View Shot Data Table (Collapsed)", expanded=False):
        st.dataframe(res['processed_df'][['shot_time', 'actual_ct', 'run_id', 'shot_type', 'stop_flag']], use_container_width=True)

    st.markdown("---")

    # C. Run-Based Analysis & Performance Breakdown
    st.header("Performance Analysis")
    
    freq_options = ['by Run', 'Hourly', 'Daily', 'Weekly', 'Monthly']
    default_ix = 2 if 'Weekly' in analysis_level or 'Monthly' in analysis_level else 0
    if 'Daily' in analysis_level: default_ix = 1 
    
    selected_freq = st.selectbox("View Analysis By", freq_options, index=default_ix, key=f'perf_breakdown_freq_{key_suffix}')

    st.subheader(f"Performance Breakdown ({selected_freq} - vs {benchmark_mode})")
    
    agg_df_chart = cr_utils.get_aggregated_data(df_view, selected_freq, config)
    
    if not agg_df_chart.empty:
        st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df_chart, 'Period', benchmark_mode), use_container_width=True)

        st.markdown("---")

        st.header(f"Production Totals Report ({selected_freq})")
        
        totals_df = agg_df_chart.copy()
        totals_df['Actual Production Time'] = totals_df.apply(
            lambda r: f"{cr_utils.format_seconds_to_dhm(r['Production Time Sec'])} ({r['Production Time Sec']/r['Run Time Sec']:.1%})" if r['Run Time Sec'] > 0 else "0m (0.0%)", 
            axis=1
        )
        totals_df['Production Shots (Pct)'] = totals_df.apply(
            lambda r: f"{r['Normal Shots']:,.0f} ({r['Normal Shots']/r['Total Shots']:.1%})" if r['Total Shots'] > 0 else "0 (0.0%)", 
            axis=1
        )
        
        totals_table = pd.DataFrame()
        totals_table['Period'] = totals_df['Period']
        totals_table['Overall Run Time'] = totals_df['Run Time'] + " (" + totals_df['Run Time Sec'].apply(lambda x: f"{x:.0f}s") + ")"
        totals_table['Actual Production Time'] = totals_df['Actual Production Time']
        totals_table['Total Shots'] = totals_df['Total Shots'].map('{:,.0f}'.format)
        totals_table['Production Shots'] = totals_df['Production Shots (Pct)']
        totals_table['Downtime Shots'] = totals_df['Downtime Shots'].map('{:,.0f}'.format)
        
        st.dataframe(totals_table, use_container_width=True, hide_index=True)

        # --- Always Show: Capacity Loss & Gain Report (vs Optimal) ---
        st.header(f"Capacity Loss & Gain Report (vs Optimal) ({selected_freq})")
        
        loss_gain_df = agg_df_chart.copy()
        
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

        def style_loss_gain(col):
            col_name = col.name
            if col_name == 'Loss (RR Downtime)': return ['color: #ff6961'] * len(col) # Red
            if col_name == 'Loss (Slow Cycles)': return ['color: #ffb347'] * len(col) # Orange/Red
            if col_name == 'Gain (Fast Cycles)': return ['color: #77dd77'] * len(col) # Green
            if col_name == 'Total Net Loss':
                return ['font-weight: bold'] * len(col)
            return [''] * len(col)

        st.dataframe(
            lg_table.style.apply(style_loss_gain, axis=0),
            use_container_width=True,
            hide_index=True
        )

        # --- Conditional: Target Loss & Gain Report (Only if Target Mode) ---
        if dashboard_mode == "Target":
            st.header(f"Target Report ({config['target_output_perc']}%) ({selected_freq})")
            st.info(f"This table allocates the gap to your target ({config['target_output_perc']}%) based on the root causes identified in the Optimal analysis.")
            
            target_df = agg_df_chart.copy()
            
            # --- Allocation Logic ---
            target_df['Ref_Net_Loss_Total'] = target_df['Downtime Loss'] + target_df['Slow Loss'] - target_df['Fast Gain']
            
            target_df['ratio_downtime'] = np.where(target_df['Ref_Net_Loss_Total'] != 0, target_df['Downtime Loss'] / target_df['Ref_Net_Loss_Total'], 0)
            target_df['ratio_slow'] = np.where(target_df['Ref_Net_Loss_Total'] != 0, target_df['Slow Loss'] / target_df['Ref_Net_Loss_Total'], 0)
            target_df['ratio_fast'] = np.where(target_df['Ref_Net_Loss_Total'] != 0, -target_df['Fast Gain'] / target_df['Ref_Net_Loss_Total'], 0)
            
            target_df['Loss_vs_Target'] = np.maximum(0, target_df['Target Output'] - target_df['Actual Output'])
            target_df['Gap_to_Target_Signed'] = target_df['Actual Output'] - target_df['Target Output']
            
            target_df['Alloc_Loss_Downtime'] = target_df['Loss_vs_Target'] * target_df['ratio_downtime']
            target_df['Alloc_Loss_Slow'] = target_df['Loss_vs_Target'] * target_df['ratio_slow']
            target_df['Alloc_Gain_Fast'] = target_df['Loss_vs_Target'] * target_df['ratio_fast'] 
            
            tgt_table = pd.DataFrame()
            tgt_table['Period'] = target_df['Period']
            tgt_table['Target Output'] = target_df['Target Output'].map('{:,.2f}'.format)
            
            tgt_table['Actual Output'] = target_df.apply(
                lambda r: f"{r['Actual Output']:,.2f} ({r['Actual Output']/r['Target Output']:.1%})" if r['Target Output'] > 0 else "0.00 (0.0%)",
                axis=1
            )
            
            tgt_table['Net Gap to Target'] = target_df['Gap_to_Target_Signed'].map('{:+,.2f}'.format)
            tgt_table['Capacity Loss (vs Target)'] = target_df['Loss_vs_Target'].map('{:,.2f}'.format)
            
            tgt_table['Allocated Loss (RR Downtime)'] = target_df.apply(
                lambda r: f"{r['Alloc_Loss_Downtime']:,.2f} ({r['ratio_downtime']:.1%})", axis=1
            )
            tgt_table['Allocated Loss (Slow Cycles)'] = target_df.apply(
                lambda r: f"{r['Alloc_Loss_Slow']:,.2f} ({r['ratio_slow']:.1%})", axis=1
            )
            tgt_table['Allocated Gain (Fast Cycles)'] = target_df.apply(
                lambda r: f"{r['Alloc_Gain_Fast']:,.2f} ({r['ratio_fast']:.1%})", axis=1
            )

            def style_target_table(col):
                col_name = col.name
                if col_name == 'Net Gap to Target':
                    return ['color: #77dd77' if float(v.replace(',','').replace('+','')) > 0 else 'color: #ff6961' for v in col]
                if col_name == 'Actual Output':
                    return ['color: #77dd77' if float(v.split('(')[1].strip('%)')) >= 100 else 'color: #ff6961' for v in col]
                if 'Loss' in col_name and 'Allocated' in col_name: return ['color: #ff6961'] * len(col)
                if 'Gain' in col_name and 'Allocated' in col_name: return ['color: #77dd77'] * len(col)
                if col_name == 'Capacity Loss (vs Target)': return ['color: #ff6961; font-weight: bold'] * len(col)
                return [''] * len(col)

            st.dataframe(
                tgt_table.style.apply(style_target_table, axis=0),
                use_container_width=True,
                hide_index=True
            )

    
    with st.expander("View Run Breakdown Table (Details)", expanded=False):
        d_df = run_breakdown_df.copy()
        
        d_df = d_df.sort_values('start_time').reset_index(drop=True)
        d_df['RUN ID'] = d_df.index.map(lambda x: f"Run {x+1:03d}")
        
        d_df["Period (date/time from to)"] = d_df.apply(lambda row: f"{row['start_time'].strftime('%Y-%m-%d %H:%M')} to {row['end_time'].strftime('%Y-%m-%d %H:%M')}", axis=1)
        
        d_df["Normal Shots"] = d_df.apply(lambda r: f"{r['normal_shots']:,} ({r['normal_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
        d_df["Stop Events"] = d_df.apply(lambda r: f"{r['stop_events']} ({r['stopped_shots']/r['total_shots']*100:.1f}%)" if r['total_shots']>0 else "0 (0.0%)", axis=1)
        d_df["Total Run duration (d/h/m)"] = d_df['total_runtime_sec'].apply(cr_utils.format_seconds_to_dhm)
        d_df["Production Time (d/h/m)"] = d_df.apply(lambda r: f"{cr_utils.format_seconds_to_dhm(r['production_time_sec'])} ({r['production_time_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
        d_df["Downtime (d/h/m)"] = d_df.apply(lambda r: f"{cr_utils.format_seconds_to_dhm(r['downtime_sec'])} ({r['downtime_sec']/r['total_runtime_sec']*100:.1f}%)" if r['total_runtime_sec']>0 else "0m (0.0%)", axis=1)
        
        rename_map = {
            'total_shots': 'Total shots',
            'mode_ct': 'Mode CT (for the run)',
            'lower_limit': 'Lower limit CT (sec)',
            'upper_limit': 'Upper Limit CT (sec)'
        }
        d_df.rename(columns=rename_map, inplace=True)
        
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
    st.sidebar.title("Capacity Risk v9.95")
    
    st.sidebar.markdown("### File Upload")
    files = st.sidebar.file_uploader("Upload Data (Excel/CSV)", accept_multiple_files=True, type=['xlsx', 'csv', 'xls'])
    
    if not files:
        st.info("👈 Upload files to begin.")
        st.stop()
        
    df_all = cr_utils.load_all_data_cr(files)
    if df_all.empty:
        st.error("No valid data found. Please check your file columns.")
        st.stop()

    tool_ids = sorted(df_all['tool_id'].unique())
    selected_tool = st.sidebar.selectbox("Select Tool ID", tool_ids)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Analysis Parameters ⚙️")
    
    with st.sidebar.expander("Configure Metrics", expanded=True):
        tolerance = st.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.05, 0.01)
        downtime_gap_tolerance = st.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5)
        run_interval_hours = st.slider("Run Interval Threshold (hours)", 1, 24, 8, 1)

    with st.sidebar.expander("Capacity Settings", expanded=False):
        # NOTE: Benchmark Mode Toggle REMOVED as requested. Controlled by tabs now.
        target_output_perc = st.slider("Target Output %", 50, 100, 90)
        default_cavities = st.number_input("Default Cavities", value=1, min_value=1)
        remove_maint = st.checkbox("Remove Maintenance/Warehouse", value=False)

    config = {
        'target_output_perc': target_output_perc,
        'tolerance': tolerance,
        'downtime_gap_tolerance': downtime_gap_tolerance,
        'run_interval_hours': run_interval_hours,
        'default_cavities': default_cavities,
        'remove_maintenance': remove_maint
    }
    
    df_tool = df_all[df_all['tool_id'] == selected_tool]

    # --- TABS: Updated for Optimal vs Target Views ---
    t_risk, t_opt, t_tgt, t_trend = st.tabs(["Risk Tower", "Capacity (Optimal)", "Capacity (Target)", "Trends"])

    with t_risk:
        render_risk_tower(df_all, config)
    
    with t_opt:
        if not df_tool.empty:
            render_dashboard(df_tool, selected_tool, config, dashboard_mode="Optimal")
        else:
            st.warning("No data for selected tool.")

    with t_tgt:
        if not df_tool.empty:
            render_dashboard(df_tool, selected_tool, config, dashboard_mode="Target")
        else:
            st.warning("No data for selected tool.")

    with t_trend:
        if not df_tool.empty:
            render_trends_tab(df_tool, config)

if __name__ == "__main__":
    main()