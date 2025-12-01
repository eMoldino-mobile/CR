import streamlit as st
import pandas as pd
import datetime
import cr_utils as cr_utils  # Keep import name exact

# ==============================================================================
# --- PAGE SETUP ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v2.0)")

# ==============================================================================
# --- UI RENDERING ---
# ==============================================================================

def render_risk_tower():
    st.header("Capacity Risk Tower")
    st.info("🚧 Under Construction. This module will compare multiple tools to identify which ones are consistently failing to meet Capacity Targets.")
    # Future logic: Loop through all tools, calc gap to target, sort by biggest gap.

def render_trends_tab(df_tool):
    st.header("Capacity Trends")
    st.info("🚧 Under Construction. This module will show week-over-week capacity loss trends.")
    # Future logic: use cr_utils.aggregate_by_period with 'week' or 'month'.

def render_dashboard(df_tool, tool_name, config):
    """
    Main Capacity Dashboard.
    """
    
    # --- Local Controls ---
    st.markdown(f"### {tool_name} Capacity Analysis")
    
    # Period Selector matching RR app style
    period_mode = st.radio("Select Analysis Level", ["Daily", "Weekly", "Monthly", "Custom"], horizontal=True)
    
    df_processed = df_tool.copy()
    
    # Date Filtering Logic
    df_view = pd.DataFrame()
    sub_header = ""
    
    if period_mode == "Daily":
        dates = sorted(df_processed['shot_time'].dt.date.unique())
        if not dates: st.warning("No data"); return
        sel_date = st.selectbox("Select Date", dates, index=len(dates)-1)
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Analysis for {sel_date.strftime('%Y-%m-%d')}"
        
    elif period_mode == "Weekly":
        df_processed['week'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week'].unique())
        sel_week = st.selectbox("Select Week", weeks, index=len(weeks)-1)
        df_view = df_processed[df_processed['week'] == sel_week]
        sub_header = f"Analysis for {sel_week}"

    elif period_mode == "Monthly":
        df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month'].unique())
        sel_month = st.selectbox("Select Month", months, index=len(months)-1)
        df_view = df_processed[df_processed['month'] == sel_month]
        sub_header = f"Analysis for {sel_month}"
        
    else: # Custom
        min_d, max_d = df_processed['shot_time'].min().date(), df_processed['shot_time'].max().date()
        c1, c2 = st.columns(2)
        s_date = c1.date_input("Start", min_d)
        e_date = c2.date_input("End", max_d)
        if s_date <= e_date:
            mask = (df_processed['shot_time'].dt.date >= s_date) & (df_processed['shot_time'].dt.date <= e_date)
            df_view = df_processed[mask]
            sub_header = f"Analysis from {s_date} to {e_date}"

    if df_view.empty:
        st.warning("No data for selected period.")
        return

    # --- Run Calculation ---
    calc = cr_utils.CapacityRiskCalculator(
        df_view, 
        config['tolerance'], 
        config['dt_gap'], 
        config['run_int'], 
        config['target_perc'], 
        config['def_cav']
    )
    res = calc.results
    
    st.subheader(sub_header)

    # --- TOP KPIS ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("Filtered Run Time", cr_utils.format_seconds_to_dhm(res['total_run_time_sec']), help="Total time minus breaks > interval")
        k2.metric("Optimal Output (100%)", f"{res['optimal_output_parts']:,.0f}", help="Based on Approved CT")
        k3.metric("Actual Output", f"{res['actual_output_parts']:,.0f}", delta=f"{res['efficiency_rate']:.1%}")
        
        gap = res['gap_to_target_parts']
        gap_color = "normal" if gap >= 0 else "inverse"
        k4.metric(f"Target Output ({config['target_perc']}%)", f"{res['target_output_parts']:,.0f}")
        k5.metric("Gap to Target", f"{gap:+,.0f}", delta_color=gap_color)

    # --- WATERFALL CHART & BREAKDOWN ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.plotly_chart(cr_utils.plot_waterfall(res, benchmark_mode="Optimal"), use_container_width=True)
        
    with c2:
        st.write("### Loss Breakdown (Parts)")
        loss_data = {
            "Category": ["Downtime Loss", "Slow Cycle Loss", "Fast Cycle Gain (Credit)", "Net Capacity Loss"],
            "Parts": [
                res['capacity_loss_downtime_parts'],
                res['capacity_loss_slow_parts'],
                -res['capacity_gain_fast_parts'],
                res['total_capacity_loss_parts']
            ]
        }
        st.dataframe(pd.DataFrame(loss_data).style.format({"Parts": "{:,.0f}"}), use_container_width=True, hide_index=True)
        
        net_loss_color = "red" if res['total_capacity_loss_parts'] > 0 else "green"
        st.markdown(f"**Total Capacity Loss:** :{net_loss_color}[{res['total_capacity_loss_parts']:,.0f} parts]")

    # --- RUN BREAKDOWN (Data Grid) ---
    st.markdown("---")
    st.subheader("Run-by-Run Breakdown")
    
    # We need to aggregate the processed df by run_id
    df_runs = res['processed_df']
    run_stats = []
    
    for r_id, df_r in df_runs.groupby('run_id'):
        # Mini calc for each run
        r_calc = cr_utils.CapacityRiskCalculator(df_r, config['tolerance'], config['dt_gap'], config['run_int'], config['target_perc'], config['def_cav'])
        r_res = r_calc.results
        run_stats.append({
            "Run ID": r_id,
            "Start": df_r['shot_time'].min(),
            "End": df_r['shot_time'].max(),
            "Run Time": cr_utils.format_seconds_to_dhm(r_res['total_run_time_sec']),
            "Optimal": r_res['optimal_output_parts'],
            "Actual": r_res['actual_output_parts'],
            "Efficiency": r_res['efficiency_rate'],
            "Net Loss": r_res['total_capacity_loss_parts']
        })
    
    run_df = pd.DataFrame(run_stats)
    if not run_df.empty:
        st.dataframe(
            run_df.style.format({
                "Optimal": "{:,.0f}", "Actual": "{:,.0f}", "Net Loss": "{:,.0f}", 
                "Efficiency": "{:.1%}", "Start": "{:%Y-%m-%d %H:%M}"
            }), 
            use_container_width=True
        )
    else:
        st.info("No runs detected in this period.")


# ==============================================================================
# --- MAIN APP LOGIC ---
# ==============================================================================

def main():
    # --- Sidebar ---
    st.sidebar.title("Capacity Risk v2")
    
    files = st.sidebar.file_uploader("Upload Data", accept_multiple_files=True, type=['xlsx', 'csv'])
    
    if not files:
        st.info("Upload files to begin.")
        st.stop()
        
    df_all = cr_utils.load_all_data_cr(files)
    
    if df_all.empty:
        st.error("No valid data found. Check columns: TOOLING ID, SHOT TIME, ACTUAL CT, APPROVED CT.")
        st.stop()
        
    tools = sorted(df_all['tool_id'].unique())
    selected_tool = st.sidebar.selectbox("Select Tool", tools)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    config = {
        'target_perc': st.sidebar.slider("Target Output %", 50, 100, 90),
        'tolerance': st.sidebar.slider("Mode Tolerance", 0.01, 0.20, 0.05),
        'dt_gap': st.sidebar.number_input("Downtime Gap (sec)", value=2.0),
        'run_int': st.sidebar.number_input("Run Interval (hrs)", value=8.0),
        'def_cav': st.sidebar.number_input("Default Cavities", value=1)
    }

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["Capacity Dashboard", "Risk Tower", "Trends"])
    
    df_tool = df_all[df_all['tool_id'] == selected_tool]
    
    with tab1:
        render_dashboard(df_tool, selected_tool, config)
        
    with tab2:
        render_risk_tower()
        
    with tab3:
        render_trends_tab(df_tool)

if __name__ == "__main__":
    main()