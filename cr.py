import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cr_utils as cr_utils  # Keep import name exact

# ==============================================================================
# --- PAGE SETUP ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v8.2)")

# ==============================================================================
# --- DASHBOARD LOGIC ---
# ==============================================================================

def render_dashboard(df_tool, tool_name, config, demand_info):
    
    # --- 1. Analysis Level Selector (Matching Run Rate) ---
    analysis_level = st.radio(
        "Select Analysis Level",
        options=["Daily", "Weekly", "Monthly", "Custom Period"],
        horizontal=True,
        key="cr_analysis_level"
    )
    
    st.markdown("---")
    
    # --- 2. Data Filtering Logic ---
    df_processed = df_tool.copy()
    df_view = pd.DataFrame()
    sub_header = ""
    
    # Default display frequency for aggregation
    if analysis_level == "Daily":
        display_freq = "by Run"
        dates = sorted(df_processed['shot_time'].dt.date.unique(), reverse=True)
        if not dates: st.warning("No data available."); return
        sel_date = st.selectbox("Select Date", dates, index=0)
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Analysis for {sel_date.strftime('%d %b %Y')}"
        
    elif analysis_level == "Weekly":
        display_freq = "Daily"
        df_processed['week_period'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week_period'].unique(), reverse=True)
        if not weeks: st.warning("No data available."); return
        sel_week = st.selectbox("Select Week", weeks, index=0)
        df_view = df_processed[df_processed['week_period'] == sel_week]
        sub_header = f"Analysis for Week {sel_week}"

    elif analysis_level == "Monthly":
        display_freq = "Weekly"
        df_processed['month_period'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month_period'].unique(), reverse=True)
        if not months: st.warning("No data available."); return
        sel_month = st.selectbox("Select Month", months, index=0)
        df_view = df_processed[df_processed['month_period'] == sel_month]
        sub_header = f"Analysis for {sel_month.strftime('%B %Y')}"
        
    else: # Custom Period
        display_freq = "Daily"
        min_d, max_d = df_processed['shot_time'].min().date(), df_processed['shot_time'].max().date()
        c1, c2 = st.columns(2)
        s = c1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
        e = c2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
        if s <= e:
            df_view = df_processed[(df_processed['shot_time'].dt.date >= s) & (df_processed['shot_time'].dt.date <= e)]
            sub_header = f"Analysis from {s} to {e}"
        else:
            st.error("Start date must be before end date.")
            return

    if df_view.empty:
        st.warning("No data found for the selected period.")
        return

    st.markdown(f"### {tool_name} Overview")
    st.caption(sub_header)

    # --- 3. Calculate Metrics ---
    calc = cr_utils.CapacityRiskCalculator(df_view, **config)
    res = calc.results
    
    # Aggregated Data for Charts
    agg_df = cr_utils.get_aggregated_data(df_view, display_freq, config)

    # --- 4. OVERALL SUMMARY (MATCHING SCREENSHOT LAYOUT) ---
    st.header("Overall Summary")
    
    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        
        # K1: Overall Run Time
        k1.metric("Overall Run Time", cr_utils.format_seconds_to_dhm(res['total_runtime_sec']))
        
        # K2: Optimal Output (100%)
        k2.metric("Optimal Output (100%)", f"{res['optimal_output_parts']:,.0f}")
        
        # K3: Actual Output (78.9%)
        actual_perc = (res['actual_output_parts'] / res['optimal_output_parts'] * 100) if res['optimal_output_parts'] > 0 else 0
        k3.metric(f"Actual Output ({actual_perc:.1f}%)", f"{res['actual_output_parts']:,.0f} parts")
        k3.caption(f"Actual Production Time: {cr_utils.format_seconds_to_dhm(res['production_time_sec'])}")
        
        # K4: Total Capacity Loss (True)
        true_loss = res['optimal_output_parts'] - res['actual_output_parts']
        
        k4.markdown("**Total Capacity Loss (True)**")
        k4.markdown(f"<h3 style='color:#ff6961; margin-top: -10px;'>{true_loss:,.0f} parts</h3>", unsafe_allow_html=True)
        # Use total_capacity_loss_sec which sums up downtime + inefficiency time
        k4.caption(f"Total Time Lost: {cr_utils.format_seconds_to_dhm(res['total_capacity_loss_sec'])}")

    # --- 5. BREAKDOWN SECTION ---
    st.header("Capacity Loss Breakdown (vs Optimal Output)")
    st.info("These values are calculated based on the time-based logic (Downtime + Slow/Fast Cycles) using Optimal Output as the benchmark.")
    
    c_chart, c_details = st.columns([1.5, 1]) # Chart takes more space
    
    with c_chart:
        st.subheader("Overall Performance Breakdown")
        st.plotly_chart(cr_utils.plot_waterfall(res, "Optimal Output"), use_container_width=True)
        
    with c_details:
        # Total Net Impact Box
        with st.container(border=True):
            st.markdown("**Total Net Impact**")
            st.markdown(f"<h2 style='color:#ff6961; margin:0;'>{true_loss:,.0f} parts</h2>", unsafe_allow_html=True)
            st.caption(f"Net Time Lost: {cr_utils.format_seconds_to_dhm(res['total_capacity_loss_sec'])}")
        
        # Detailed Table
        net_cycle_loss = res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts']
        
        # Construct the table data
        breakdown_data = [
            {"Metric": "Loss (RR Downtime)", "Parts": res['capacity_loss_downtime_parts'], "Time": cr_utils.format_seconds_to_dhm(res['downtime_sec'])},
            {"Metric": "Net Loss (Cycle Time)", "Parts": net_cycle_loss, "Time": "N/A"}, # Time is mixed
            {"Metric": "└ Loss (Slow Cycles)", "Parts": res['capacity_loss_slow_parts'], "Time": cr_utils.format_seconds_to_dhm(res['capacity_loss_slow_parts'] * (res['production_time_sec']/res['actual_output_parts']) if res['actual_output_parts'] else 0)},
            {"Metric": "└ Gain (Fast Cycles)", "Parts": res['capacity_gain_fast_parts'], "Time": cr_utils.format_seconds_to_dhm(res['capacity_gain_fast_parts'] * (res['production_time_sec']/res['actual_output_parts']) if res['actual_output_parts'] else 0)},
        ]
        
        df_breakdown = pd.DataFrame(breakdown_data)
        
        # Custom Styling for the table
        def style_breakdown(row):
            styles = [''] * len(row)
            if row['Metric'] == "Loss (RR Downtime)":
                styles[1] = 'color: #ff6961; font-weight: bold;' # Red
            elif row['Metric'] == "Net Loss (Cycle Time)":
                color = '#ff6961' if row['Parts'] > 0 else '#77dd77' # Red if loss, Green if gain
                styles[1] = f'color: {color}; font-weight: bold;'
            elif "Gain" in row['Metric']:
                styles[1] = 'color: #77dd77;' # Green
            elif "Loss" in row['Metric']:
                styles[1] = 'color: #ff6961;' # Red
            return styles

        st.dataframe(
            df_breakdown.style.apply(style_breakdown, axis=1).format({"Parts": "{:,.0f}"}), 
            use_container_width=True, 
            hide_index=True
        )

    st.divider()

    # --- 6. DETAILED REPORTS & SHOT ANALYSIS ---
    
    # Use tabs for the bottom section to keep it clean
    tab_details, tab_shots, tab_predict = st.tabs(["Detailed Trends", "Shot-by-Shot Analysis", "Future Forecast"])
    
    with tab_details:
        st.header(f"{display_freq} Performance Breakdown")
        if not agg_df.empty:
            st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df, 'Period', "Optimal Output"), use_container_width=True)
            
            # Rich Table
            opt_table = agg_df[['Period', 'Optimal Output', 'Actual Output', 'Downtime Loss', 'Slow Loss', 'Fast Gain', 'Total Loss']].copy()
            opt_table['Actual %'] = (opt_table['Actual Output'] / opt_table['Optimal Output']).fillna(0)
            
            display_opt = pd.DataFrame()
            display_opt[display_freq] = opt_table['Period']
            display_opt['Optimal Output'] = opt_table['Optimal Output']
            display_opt['Actual Output'] = opt_table.apply(lambda x: f"{x['Actual Output']:,.0f} ({x['Actual %']:.1%})", axis=1)
            display_opt['Loss (Downtime)'] = opt_table['Downtime Loss']
            display_opt['Loss (Slow)'] = opt_table['Slow Loss']
            display_opt['Gain (Fast)'] = opt_table['Fast Gain']
            display_opt['Total Net Loss'] = opt_table['Total Loss']

            def style_loss_gain(col):
                if 'Loss' in col.name: return ['color: #ff6961'] * len(col)
                if 'Gain' in col.name: return ['color: #77dd77'] * len(col)
                return [''] * len(col)

            st.dataframe(
                display_opt.style.format({'Optimal Output': '{:,.0f}', 'Loss (Downtime)': '{:,.0f}', 'Loss (Slow)': '{:,.0f}', 'Gain (Fast)': '{:,.0f}', 'Total Net Loss': '{:,.0f}'})
                .apply(style_loss_gain, axis=0),
                use_container_width=True
            )
            
    with tab_shots:
        st.header("Shot-by-Shot Analysis")
        with st.expander("Expand to view Shot Chart & Data", expanded=True):
            if 'processed_df' in res and 'run_id' in res['processed_df'].columns:
                runs = sorted(res['processed_df']['run_id'].unique())
                run_filter = st.selectbox("Filter by Run", ["All Runs"] + list(runs))
                zoom_y = st.slider("Zoom Y-Axis (sec)", 10, 500, 100)
                
                df_shots = res['processed_df']
                if run_filter != "All Runs":
                    df_shots = df_shots[df_shots['run_id'] == run_filter]
                    
                if len(df_shots) > 5000:
                    st.warning(f"Displaying first 5,000 shots of {len(df_shots)}.")
                    df_shots_chart = df_shots.head(5000)
                else:
                    df_shots_chart = df_shots
                    
                st.plotly_chart(cr_utils.plot_shot_analysis(df_shots_chart, zoom_y), use_container_width=True)
                
                display_shots = df_shots[[
                    'shot_time', 'actual_ct', 'approved_ct', 'working_cavities', 
                    'run_id', 'shot_type', 'stop_flag'
                ]].copy()
                display_shots['run_id'] = display_shots['run_id'] + 1 
                st.dataframe(display_shots.style.format({'actual_ct': '{:.2f}', 'approved_ct': '{:.2f}', 'working_cavities': '{:.0f}'}), use_container_width=True)

    with tab_predict:
        # Theoretical Calc
        daily_agg_for_theo = cr_utils.get_aggregated_data(df_view, 'Daily', config)
        if daily_agg_for_theo.empty:
            st.info("Not enough data for prediction.")
        else:
            st.header("Capacity Forecast 🔮")
            
            # Show Theoretical Metric Here
            theo_monthly, _, days_week, peak_daily = cr_utils.calculate_theoretical_capacity(daily_agg_for_theo)
            st.metric("Theoretical Monthly Capacity", f"{theo_monthly:,.0f}", help=f"Based on Peak P90 Daily: {peak_daily:,.0f} parts")
            
            pc1, pc2 = st.columns(2)
            max_date = pd.to_datetime(daily_agg_for_theo['Period']).max().date()
            default_future = max_date + timedelta(days=30)
            with pc1: pred_start = st.date_input("Prediction Start", max_date)
            with pc2: pred_end = st.date_input("Target Date", default_future, min_value=pred_start)
            
            dates, proj_act, proj_peak, proj_tgt, start_val, daily_rate, peak_rate = cr_utils.generate_prediction_data(
                daily_agg_for_theo, pred_start, pred_end, demand_info['target']
            )
            
            fig_pred = cr_utils.plot_prediction_chart(dates, proj_act, proj_peak, proj_tgt, demand_info['target'], pred_start, start_val)
            st.plotly_chart(fig_pred, use_container_width=True)


# ==============================================================================
# --- MAIN APP ENTRY ---
# ==============================================================================

def main():
    # --- Sidebar Title ---
    st.sidebar.title("Capacity Risk v8.2")
    
    # --- 1. File Upload ---
    st.sidebar.markdown("### File Upload")
    files = st.sidebar.file_uploader("Upload Data", accept_multiple_files=True, type=['xlsx', 'csv'])
    
    if not files:
        st.info("👈 Upload files to begin.")
        st.stop()
        
    df_all = cr_utils.load_all_data_cr(files)
    if df_all.empty:
        st.error("No valid data found. Check columns.")
        st.stop()
        
    # --- 2. Tool Selection ---
    st.sidebar.markdown("### Select Tool ID")
    tools = sorted(df_all['tool_id'].unique())
    selected_tool = st.sidebar.selectbox("Select Tool ID for Dashboard Analysis", tools)
    
    # --- 3. Analysis Parameters (Run Rate Style) ---
    st.sidebar.title("Analysis Parameters ⚙️")
    
    with st.sidebar.expander("Configure Metrics", expanded=True):
        tolerance = st.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.05, 0.01, help="Defines the ±% around Mode CT.")
        downtime_gap_tolerance = st.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Minimum idle time between shots to be considered a stop.")
        run_interval_hours = st.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Max hours between shots before a new Run is identified.")

    with st.sidebar.expander("Capacity Settings", expanded=False):
        target_output_perc = st.slider("Target Output % (of Optimal)", 50, 100, 90)
        default_cavities = st.number_input("Default Cavities", value=1)
        remove_maintenance = st.checkbox("Remove Maint./Warehouse", value=False)

    with st.sidebar.expander("Demand Targets", expanded=False):
        target_demand = st.number_input("Target Demand", min_value=0)
        received_parts = st.number_input("Received Parts", min_value=0)

    config = {
        'target_output_perc': target_output_perc,
        'tolerance': tolerance,
        'downtime_gap_tolerance': downtime_gap_tolerance,
        'run_interval_hours': run_interval_hours,
        'default_cavities': default_cavities,
        'remove_maintenance': remove_maintenance
    }
    demand_info = {'target': target_demand, 'received': received_parts}

    # --- Tabs ---
    tab1, tab2 = st.tabs(["Capacity Dashboard", "Risk Tower"])
    
    df_tool = df_all[df_all['tool_id'] == selected_tool]
    
    with tab1:
        render_dashboard(df_tool, selected_tool, config, demand_info)
    with tab2:
        if not df_all.empty:
            risk_df = cr_utils.calculate_capacity_risk_scores(df_all, config)
            def style_risk_score(val):
                if val >= 80: color = '#77dd77'
                elif val >= 50: color = '#ffb347'
                else: color = '#ff6961'
                return f'background-color: {color}'
            st.dataframe(
                risk_df.style.format({'Achievement %': '{:.1f}%', 'Risk Score': '{:.0f}'})
                             .map(style_risk_score, subset=['Risk Score']), 
                use_container_width=True
            )

if __name__ == "__main__":
    main()