import streamlit as st
import pandas as pd
import cr_utils as cr_utils  # Keep import name exact

# ==============================================================================
# --- PAGE SETUP ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v3.0)")

# ==============================================================================
# --- DASHBOARD LOGIC ---
# ==============================================================================

def render_dashboard(df_tool, tool_name, config):
    st.markdown(f"### {tool_name} Capacity Analysis")
    
    # 1. Period & Frequency Selectors
    c1, c2 = st.columns(2)
    with c1:
        period_mode = st.radio("Analysis Period", ["Entire File", "Daily", "Weekly", "Monthly", "Custom"], horizontal=True)
    with c2:
        freq_mode = st.radio("Display Frequency", ["Daily", "Weekly", "Monthly", "by Run"], index=3, horizontal=True)

    # 2. Filter Data
    df_processed = df_tool.copy()
    df_view = pd.DataFrame()
    sub_header = ""

    if period_mode == "Entire File":
        df_view = df_processed
        sub_header = "Entire File Analysis"
    elif period_mode == "Daily":
        dates = sorted(df_processed['shot_time'].dt.date.unique(), reverse=True)
        sel_date = st.selectbox("Select Date", dates)
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Analysis for {sel_date}"
    elif period_mode == "Weekly":
        df_processed['week'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week'].unique(), reverse=True)
        sel_week = st.selectbox("Select Week", weeks)
        df_view = df_processed[df_processed['week'] == sel_week]
        sub_header = f"Analysis for {sel_week}"
    elif period_mode == "Monthly":
        df_processed['month'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month'].unique(), reverse=True)
        sel_month = st.selectbox("Select Month", months)
        df_view = df_processed[df_processed['month'] == sel_month]
        sub_header = f"Analysis for {sel_month}"
    else:
        min_d, max_d = df_processed['shot_time'].min().date(), df_processed['shot_time'].max().date()
        s = st.date_input("Start", min_d); e = st.date_input("End", max_d)
        if s <= e: df_view = df_processed[(df_processed['shot_time'].dt.date >= s) & (df_processed['shot_time'].dt.date <= e)]
        sub_header = f"Analysis {s} to {e}"

    if df_view.empty: st.warning("No data."); return

    # 3. Calculate Metrics
    calc = cr_utils.CapacityRiskCalculator(df_view, **config)
    res = calc.results
    
    st.subheader(sub_header)

    # --- KPI SECTION ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        
        # Determine Benchmark (Optimal or Target)
        bm_view = st.radio("Benchmark", ["Optimal Output", "Target Output"], horizontal=True, label_visibility="collapsed")
        
        k1.metric("Run Time", cr_utils.format_seconds_to_dhm(res['total_runtime_sec']))
        
        if bm_view == "Target Output":
            k2.metric(f"Target ({config['target_output_perc']}%)", f"{res['target_output_parts']:,.0f}")
            k2.caption(f"Optimal: {res['optimal_output_parts']:,.0f}")
            
            gap = res['gap_to_target_parts']
            color = "normal" if gap >= 0 else "inverse"
            k3.metric("Actual", f"{res['actual_output_parts']:,.0f}", delta=f"{gap:+,.0f} Gap", delta_color=color)
            
            k4.markdown("**Loss vs Target**")
            k4.markdown(f"<h3 style='color:red'>{res['capacity_loss_vs_target_parts']:,.0f}</h3>", unsafe_allow_html=True)
        else:
            k2.metric("Optimal (100%)", f"{res['optimal_output_parts']:,.0f}")
            k3.metric("Actual", f"{res['actual_output_parts']:,.0f}", delta=f"{res['efficiency_rate']:.1%}")
            
            k4.markdown("**Total Capacity Loss**")
            k4.markdown(f"<h3 style='color:red'>{res['total_capacity_loss_parts']:,.0f}</h3>", unsafe_allow_html=True)

        k5.metric("Downtime", cr_utils.format_seconds_to_dhm(res['downtime_sec']), help="Total time of all stop events")

    # --- CHART SECTION (Waterfall & Breakdown) ---
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(cr_utils.plot_waterfall(res, bm_view), use_container_width=True)
    with c2:
        # Mini Table for Waterfall
        loss_data = {
            "Category": ["Downtime Loss", "Slow Cycle Loss", "Fast Cycle Gain", "Net Cycle Loss", "Total Net Loss"],
            "Parts": [
                res['capacity_loss_downtime_parts'],
                res['capacity_loss_slow_parts'],
                res['capacity_gain_fast_parts'],
                res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts'],
                res['total_capacity_loss_parts']
            ]
        }
        st.dataframe(pd.DataFrame(loss_data).style.format({"Parts": "{:,.0f}"}), use_container_width=True)

    st.divider()

    # --- AGGREGATED REPORT SECTION ---
    st.header(f"{freq_mode} Performance Breakdown")
    
    # Calculate Aggregated Data
    agg_df = cr_utils.get_aggregated_data(df_view, freq_mode, config)
    
    if not agg_df.empty:
        # Stacked Bar Chart
        st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df, 'Period', bm_view), use_container_width=True)
        
        # Detailed Data Table
        st.subheader(f"Detailed {freq_mode} Data")
        st.dataframe(
            agg_df.style.format({
                'Actual Output': '{:,.0f}', 'Optimal Output': '{:,.0f}', 'Target Output': '{:,.0f}',
                'Total Loss': '{:,.0f}', 'Gap to Target': '{:+,.0f}'
            }).background_gradient(subset=['Gap to Target'], cmap='RdYlGn', vmin=-1000, vmax=1000),
            use_container_width=True
        )
    
    st.divider()

    # --- SHOT ANALYSIS SECTION ---
    st.header("Shot-by-Shot Analysis")
    with st.expander("Expand to view Shot Chart", expanded=True):
        # Allow selecting a single run if there are too many shots
        runs = sorted(res['processed_df']['run_id'].unique())
        run_filter = st.selectbox("Filter by Run (Optional)", ["All Runs"] + list(runs))
        
        zoom_y = st.slider("Zoom Y-Axis (sec)", 10, 500, 100)
        
        df_shots = res['processed_df']
        if run_filter != "All Runs":
            df_shots = df_shots[df_shots['run_id'] == run_filter]
            
        if len(df_shots) > 5000:
            st.warning(f"Displaying first 5,000 shots of {len(df_shots)}. Filter by Run to see specific details.")
            df_shots = df_shots.head(5000)
            
        st.plotly_chart(cr_utils.plot_shot_analysis(df_shots, zoom_y), use_container_width=True)


# ==============================================================================
# --- MAIN APP ENTRY ---
# ==============================================================================

def main():
    st.sidebar.title("Capacity Risk v3")
    
    files = st.sidebar.file_uploader("Upload Data", accept_multiple_files=True, type=['xlsx', 'csv'])
    if not files: st.info("Upload files."); st.stop()
        
    df_all = cr_utils.load_all_data_cr(files)
    if df_all.empty: st.error("No valid data."); st.stop()
        
    tools = sorted(df_all['tool_id'].unique())
    selected_tool = st.sidebar.selectbox("Select Tool", tools)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")
    
    config = {
        'target_output_perc': st.sidebar.slider("Target Output %", 50, 100, 90),
        'tolerance': st.sidebar.slider("Mode Tolerance", 0.01, 0.20, 0.05),
        'downtime_gap_tolerance': st.sidebar.number_input("Downtime Gap (sec)", value=2.0),
        'run_interval_hours': st.sidebar.number_input("Run Interval (hrs)", value=8.0),
        'default_cavities': st.sidebar.number_input("Default Cavities", value=1),
        'remove_maintenance': st.sidebar.checkbox("Remove Maint./Warehouse", value=False)
    }

    tab1, tab2 = st.tabs(["Capacity Dashboard", "Risk Tower"])
    
    df_tool = df_all[df_all['tool_id'] == selected_tool]
    
    with tab1:
        render_dashboard(df_tool, selected_tool, config)
    with tab2:
        if not df_all.empty:
            risk_df = cr_utils.calculate_capacity_risk_scores(df_all, config)
            st.dataframe(risk_df.style.format({'Achievement %': '{:.1f}%', 'Risk Score': '{:.0f}'}), use_container_width=True)

if __name__ == "__main__":
    main()