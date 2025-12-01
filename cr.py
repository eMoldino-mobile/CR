import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cr_utils as cr_utils  # Keep import name exact

# ==============================================================================
# --- PAGE SETUP ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v7.0)")

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
    freq_mode = analysis_level # Mapping for aggregation functions

    if analysis_level == "Daily":
        dates = sorted(df_processed['shot_time'].dt.date.unique(), reverse=True)
        if not dates: st.warning("No data available."); return
        sel_date = st.selectbox("Select Date", dates, index=0)
        df_view = df_processed[df_processed['shot_time'].dt.date == sel_date]
        sub_header = f"Summary for {sel_date.strftime('%d %b %Y')}"
        
    elif analysis_level == "Weekly":
        df_processed['week_period'] = df_processed['shot_time'].dt.to_period('W')
        weeks = sorted(df_processed['week_period'].unique(), reverse=True)
        if not weeks: st.warning("No data available."); return
        sel_week = st.selectbox("Select Week", weeks, index=0)
        df_view = df_processed[df_processed['week_period'] == sel_week]
        sub_header = f"Summary for Week {sel_week}"

    elif analysis_level == "Monthly":
        df_processed['month_period'] = df_processed['shot_time'].dt.to_period('M')
        months = sorted(df_processed['month_period'].unique(), reverse=True)
        if not months: st.warning("No data available."); return
        sel_month = st.selectbox("Select Month", months, index=0)
        df_view = df_processed[df_processed['month_period'] == sel_month]
        sub_header = f"Summary for {sel_month.strftime('%B %Y')}"
        
    else: # Custom Period
        min_d, max_d = df_processed['shot_time'].min().date(), df_processed['shot_time'].max().date()
        c1, c2 = st.columns(2)
        s = c1.date_input("Start Date", min_d, min_value=min_d, max_value=max_d)
        e = c2.date_input("End Date", max_d, min_value=min_d, max_value=max_d)
        if s <= e:
            df_view = df_processed[(df_processed['shot_time'].dt.date >= s) & (df_processed['shot_time'].dt.date <= e)]
            sub_header = f"Summary from {s} to {e}"
        else:
            st.error("Start date must be before end date.")
            return

    if df_view.empty:
        st.warning("No data found for the selected period.")
        return

    st.markdown(f"### {tool_name} Overview")
    st.subheader(sub_header)

    # --- 3. Calculate Metrics ---
    calc = cr_utils.CapacityRiskCalculator(df_view, **config)
    res = calc.results
    
    # 4. Calculate Theoretical Metrics & Aggregations
    # For chart display, we need to decide aggregation based on the level
    # Daily -> Hourly or by Run? CR usually aggregates by Day/Week. 
    # If viewing a single Day, "by Run" is the best breakdown.
    if analysis_level == "Daily":
        display_freq = "by Run"
    elif analysis_level == "Weekly":
        display_freq = "Daily"
    elif analysis_level == "Monthly":
        display_freq = "Weekly"
    else:
        display_freq = "Daily"

    agg_df = cr_utils.get_aggregated_data(df_view, display_freq, config)
    
    # Theoretical calc usually needs daily data
    daily_agg_for_theo = cr_utils.get_aggregated_data(df_view, 'Daily', config)
    theo_monthly, monthly_factor, days_week, peak_daily = cr_utils.calculate_theoretical_capacity(daily_agg_for_theo)

    # --- KPI SECTION (MATCHING RR LAYOUT) ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        
        bm_view = st.radio("Benchmark", ["Optimal Output", "Target Output"], horizontal=True, label_visibility="collapsed")
        
        # Use 'Filtered Run Time' label to match v7.53
        k1.metric("Run Time", cr_utils.format_seconds_to_dhm(res['total_runtime_sec']))
        
        if bm_view == "Target Output":
            k2.metric(f"Target ({config['target_output_perc']}%)", f"{res['target_output_parts']:,.0f}")
            k3.metric("Actual", f"{res['actual_output_parts']:,.0f}", delta=f"{res['gap_to_target_parts']:+,.0f} Gap", delta_color="normal" if res['gap_to_target_parts'] >= 0 else "inverse")
        else:
            k2.metric("Optimal (100%)", f"{res['optimal_output_parts']:,.0f}")
            k3.metric("Actual", f"{res['actual_output_parts']:,.0f}", delta=f"{res['efficiency_rate']:.1%}")

        if theo_monthly > 0:
            k4.metric("Theoretical Monthly", f"{theo_monthly:,.0f}", help=f"Based on Peak P90: {peak_daily:,.0f}")
        else:
            k4.metric("Downtime", cr_utils.format_seconds_to_dhm(res['downtime_sec']))
            
        k5.metric("Stability Index", f"{res['stability_index']:.1f}%")

    # --- DONUT CHARTS (MATCHING RR VISUALS) ---
    with st.container(border=True):
        d1, d2 = st.columns(2)
        
        # 1. Capacity Utilization
        utilization_perc = (res['actual_output_parts'] / res['optimal_output_parts'] * 100) if res['optimal_output_parts'] > 0 else 0
        d1.plotly_chart(cr_utils.create_donut_chart(utilization_perc, "Capacity Utilization (%)", color_scheme='blue'), use_container_width=True)
        
        # 2. Target Achievement OR Efficiency
        if bm_view == "Target Output":
            target_ach_perc = (res['actual_output_parts'] / res['target_output_parts'] * 100) if res['target_output_parts'] > 0 else 0
            d2.plotly_chart(cr_utils.create_donut_chart(target_ach_perc, "Target Achievement (%)", color_scheme='dynamic'), use_container_width=True)
        else:
            d2.plotly_chart(cr_utils.create_donut_chart(res['stability_index'], "Run Stability (%)", color_scheme='dynamic'), use_container_width=True)

    # --- DAILY SUMMARY EXPANDER ---
    with st.expander("View Daily Summary Data"):
        if not daily_agg_for_theo.empty:
            daily_disp = daily_agg_for_theo.copy()
            daily_disp['Actual %'] = (daily_disp['Actual Output'] / daily_disp['Optimal Output']).fillna(0)
            daily_disp['Loss %'] = (daily_disp['Total Loss'] / daily_disp['Optimal Output']).fillna(0)
            st.dataframe(daily_disp.style.format({
                'Actual Output': '{:,.0f}', 'Optimal Output': '{:,.0f}', 'Total Loss': '{:,.0f}', 
                'Actual %': '{:.1%}', 'Loss %': '{:.1%}', 'Gap to Target': '{:+,.0f}'
            }), use_container_width=True)

    # --- DASHBOARD TABS ---
    d_tab1, d_tab2 = st.tabs(["Performance Analysis", "Future Prediction"])

    with d_tab1:
        # --- CHART SECTION ---
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(cr_utils.plot_waterfall(res, bm_view), use_container_width=True)
        with c2:
            loss_data = {
                "Category": ["Downtime Loss", "Slow Cycle Loss", "Fast Cycle Gain", "Net Cycle Loss", "Total Net Loss"],
                "Parts": [
                    res['capacity_loss_downtime_parts'], res['capacity_loss_slow_parts'],
                    res['capacity_gain_fast_parts'], res['capacity_loss_slow_parts'] - res['capacity_gain_fast_parts'],
                    res['total_capacity_loss_parts']
                ]
            }
            st.dataframe(pd.DataFrame(loss_data).style.format({"Parts": "{:,.0f}"}), use_container_width=True)

        st.divider()

        # --- AGGREGATED REPORT ---
        st.header(f"{display_freq} Performance Breakdown")
        
        if not agg_df.empty:
            st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df, 'Period', bm_view), use_container_width=True)
            
            st.subheader(f"Detailed Capacity Loss & Gain Report (vs Optimal)")
            
            # --- Rich Table Logic ---
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

            # --- Target Table ---
            if bm_view == "Target Output":
                st.subheader("Target Report (Allocated Losses)")
                tgt_table = agg_df.copy()
                tgt_table['Total Loss Base'] = tgt_table['Total Loss'].replace(0, 1)
                tgt_table['Ratio Downtime'] = tgt_table['Downtime Loss'] / tgt_table['Total Loss Base']
                tgt_table['Ratio Slow'] = tgt_table['Slow Loss'] / tgt_table['Total Loss Base']
                tgt_table['Ratio Fast'] = -tgt_table['Fast Gain'] / tgt_table['Total Loss Base']
                tgt_table['Loss vs Target'] = (tgt_table['Target Output'] - tgt_table['Actual Output']).clip(lower=0)
                
                tgt_table['Allocated Downtime'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Downtime']
                tgt_table['Allocated Slow'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Slow']
                tgt_table['Allocated Fast'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Fast']

                display_tgt = pd.DataFrame()
                display_tgt[display_freq] = tgt_table['Period']
                display_tgt['Target Output'] = tgt_table['Target Output']
                display_tgt['Actual Output'] = tgt_table['Actual Output']
                display_tgt['Gap to Target'] = tgt_table['Gap to Target']
                display_tgt['Allocated (Downtime)'] = tgt_table['Allocated Downtime']
                display_tgt['Allocated (Slow)'] = tgt_table['Allocated Slow']
                display_tgt['Allocated (Fast)'] = tgt_table['Allocated Fast']

                def style_target(col):
                    if col.name == 'Gap to Target': return ['color: #77dd77' if v >= 0 else 'color: #ff6961' for v in display_tgt['Gap to Target']]
                    if 'Allocated' in col.name: return ['color: #ff6961'] * len(col)
                    return [''] * len(col)

                st.dataframe(
                    display_tgt.style.format("{:,.0f}").apply(style_target, axis=0),
                    use_container_width=True
                )
        
        st.divider()

        # --- SHOT ANALYSIS ---
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

    with d_tab2:
        if daily_agg_for_theo.empty:
            st.info("Not enough data for prediction.")
        else:
            st.header("Capacity Forecast 🔮")
            pc1, pc2 = st.columns(2)
            max_date = pd.to_datetime(daily_agg_for_theo['Period']).max().date()
            default_future = max_date + timedelta(days=30)
            with pc1: pred_start = st.date_input("Prediction Start", max_date)
            with pc2: pred_end = st.date_input("Target Date", default_future, min_value=pred_start)
            
            dates, proj_act, proj_peak, proj_tgt, start_val, daily_rate, peak_rate = cr_utils.generate_prediction_data(
                daily_agg_for_theo, pred_start, pred_end, demand_info['target']
            )
            
            fig_pred = cr_utils.plot_prediction_chart(dates, proj_act, proj_peak, proj_tgt, demand_info['target'], pred_start, start_val)
            
            # Add Historical Line (need daily data for this chart)
            hist_df = daily_agg_for_theo.copy()
            hist_df['Period'] = pd.to_datetime(hist_df['Period'])
            hist_df = hist_df.sort_values('Period')
            hist_df['Cum Actual'] = hist_df['Actual Output'].cumsum()
            fig_pred.add_trace(go.Scatter(x=hist_df['Period'], y=hist_df['Cum Actual'], mode='lines', name='Historical Actual', line=dict(color='#3498DB', width=3)))
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            total_proj = proj_act[-1] - start_val
            total_peak = proj_peak[-1] - start_val
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Predicted Output", f"{total_proj:,.0f}")
            mc2.metric("Theoretical Max Output", f"{total_peak:,.0f}")
            mc3.metric("Projected Opportunity Gap", f"{total_peak - total_proj:,.0f}", delta_color="inverse")


# ==============================================================================
# --- MAIN APP ENTRY ---
# ==============================================================================

def main():
    # --- Sidebar Title ---
    st.sidebar.title("Capacity Risk v7.0")
    
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
    
    # --- 3. Analysis Parameters (Matching Run Rate) ---
    st.sidebar.title("Analysis Parameters ⚙️")
    
    with st.sidebar.expander("Configure Metrics", expanded=True):
        tolerance = st.slider("Tolerance Band (% of Mode CT)", 0.01, 0.50, 0.05, 0.01, help="Defines the ±% around Mode CT.")
        downtime_gap_tolerance = st.slider("Downtime Gap Tolerance (sec)", 0.0, 5.0, 2.0, 0.5, help="Minimum idle time between shots to be considered a stop.")
        run_interval_hours = st.slider("Run Interval Threshold (hours)", 1, 24, 8, 1, help="Max hours between shots before a new Run is identified.")

    # --- 4. Capacity Specific Settings ---
    with st.sidebar.expander("Capacity Settings", expanded=False):
        target_output_perc = st.slider("Target Output % (of Optimal)", 50, 100, 90)
        default_cavities = st.number_input("Default Cavities", value=1)
        remove_maintenance = st.checkbox("Remove Maint./Warehouse", value=False)

    # --- 5. Demand Targets ---
    with st.sidebar.expander("Demand Targets", expanded=False):
        target_demand = st.number_input("Target Demand", min_value=0)
        received_parts = st.number_input("Received Parts", min_value=0)

    # Bundle config
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