import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import cr_utils as cr_utils  # Keep import name exact

# ==============================================================================
# --- PAGE SETUP ---
# ==============================================================================
st.set_page_config(layout="wide", page_title="Capacity Risk Dashboard (v6.1)")

# ==============================================================================
# --- DASHBOARD LOGIC ---
# ==============================================================================

def render_dashboard(df_tool, tool_name, config, demand_info):
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
    
    # 4. Calculate Theoretical Metrics & Aggregations
    daily_agg = cr_utils.get_aggregated_data(df_view, 'Daily', config)
    theo_monthly, monthly_factor, days_week, peak_daily = cr_utils.calculate_theoretical_capacity(daily_agg)

    st.subheader(sub_header)

    # --- KPI SECTION ---
    with st.container(border=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        
        bm_view = st.radio("Benchmark", ["Optimal Output", "Target Output"], horizontal=True, label_visibility="collapsed")
        
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

    # --- DONUT CHARTS ---
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

    # --- DAILY SUMMARY EXPANDER (v7.53 Feature) ---
    with st.expander("View Daily Summary Data"):
        if not daily_agg.empty:
            daily_disp = daily_agg.copy()
            daily_disp['Actual %'] = (daily_disp['Actual Output'] / daily_disp['Optimal Output']).fillna(0)
            daily_disp['Loss %'] = (daily_disp['Total Loss'] / daily_disp['Optimal Output']).fillna(0)
            
            # Format and Display
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
        st.header(f"{freq_mode} Performance Breakdown")
        agg_df = cr_utils.get_aggregated_data(df_view, freq_mode, config)
        
        if not agg_df.empty:
            st.plotly_chart(cr_utils.plot_performance_breakdown(agg_df, 'Period', bm_view), use_container_width=True)
            
            st.subheader(f"Detailed Capacity Loss & Gain Report (vs Optimal)")
            
            # --- Rich Table Logic (from v7.53) ---
            opt_table = agg_df[['Period', 'Optimal Output', 'Actual Output', 'Downtime Loss', 'Slow Loss', 'Fast Gain', 'Total Loss']].copy()
            opt_table['Actual %'] = (opt_table['Actual Output'] / opt_table['Optimal Output']).fillna(0)
            opt_table['Downtime %'] = (opt_table['Downtime Loss'] / opt_table['Optimal Output']).fillna(0)
            opt_table['Slow %'] = (opt_table['Slow Loss'] / opt_table['Optimal Output']).fillna(0)
            opt_table['Fast %'] = (opt_table['Fast Gain'] / opt_table['Optimal Output']).fillna(0)
            opt_table['Total Loss %'] = (opt_table['Total Loss'] / opt_table['Optimal Output']).fillna(0)

            display_opt = pd.DataFrame()
            display_opt[freq_mode] = opt_table['Period']
            display_opt['Optimal Output'] = opt_table['Optimal Output']
            display_opt['Actual Output'] = opt_table.apply(lambda x: f"{x['Actual Output']:,.0f} ({x['Actual %']:.1%})", axis=1)
            display_opt['Loss (Downtime)'] = opt_table.apply(lambda x: f"{x['Downtime Loss']:,.0f} ({x['Downtime %']:.1%})", axis=1)
            display_opt['Loss (Slow)'] = opt_table.apply(lambda x: f"{x['Slow Loss']:,.0f} ({x['Slow %']:.1%})", axis=1)
            display_opt['Gain (Fast)'] = opt_table.apply(lambda x: f"{x['Fast Gain']:,.0f} ({x['Fast %']:.1%})", axis=1)
            display_opt['Total Net Loss'] = opt_table.apply(lambda x: f"{x['Total Loss']:,.0f} ({x['Total Loss %']:.1%})", axis=1)

            def style_loss_gain(col):
                if 'Loss' in col.name: return ['color: #ff6961'] * len(col)
                if 'Gain' in col.name: return ['color: #77dd77'] * len(col)
                return [''] * len(col)

            st.dataframe(
                display_opt.style.format({'Optimal Output': '{:,.0f}'}).apply(style_loss_gain, axis=0),
                use_container_width=True
            )

            # --- Target Table Logic (if Target Selected) ---
            if bm_view == "Target Output":
                st.subheader("Target Report (Allocated Losses)")
                tgt_table = agg_df.copy()
                
                # Allocation Logic from v7.53
                tgt_table['Total Loss Base'] = tgt_table['Total Loss'].replace(0, 1)
                tgt_table['Ratio Downtime'] = tgt_table['Downtime Loss'] / tgt_table['Total Loss Base']
                tgt_table['Ratio Slow'] = tgt_table['Slow Loss'] / tgt_table['Total Loss Base']
                tgt_table['Ratio Fast'] = -tgt_table['Fast Gain'] / tgt_table['Total Loss Base']
                
                tgt_table['Loss vs Target'] = (tgt_table['Target Output'] - tgt_table['Actual Output']).clip(lower=0)
                
                tgt_table['Allocated Downtime'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Downtime']
                tgt_table['Allocated Slow'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Slow']
                tgt_table['Allocated Fast'] = tgt_table['Loss vs Target'] * tgt_table['Ratio Fast']

                display_tgt = pd.DataFrame()
                display_tgt[freq_mode] = tgt_table['Period']
                display_tgt['Target Output'] = tgt_table['Target Output']
                display_tgt['Actual Output'] = tgt_table['Actual Output']
                display_tgt['Gap to Target'] = tgt_table['Gap to Target']
                display_tgt['Allocated (Downtime)'] = tgt_table['Allocated Downtime']
                display_tgt['Allocated (Slow)'] = tgt_table['Allocated Slow']
                display_tgt['Allocated (Fast)'] = tgt_table['Allocated Fast']

                def style_target(col):
                    if col.name == 'Gap to Target':
                        return ['color: #77dd77' if v >= 0 else 'color: #ff6961' for v in display_tgt['Gap to Target']]
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
                
                # Detailed Shot Data Table
                display_shots = df_shots[[
                    'shot_time', 'actual_ct', 'approved_ct', 'working_cavities', 
                    'run_id', 'shot_type', 'stop_flag'
                ]].copy()
                display_shots['run_id'] = display_shots['run_id'] + 1 
                st.dataframe(display_shots.style.format({'actual_ct': '{:.2f}', 'approved_ct': '{:.2f}', 'working_cavities': '{:.0f}'}), use_container_width=True)

    with d_tab2:
        if daily_agg.empty:
            st.info("Not enough data for prediction.")
        else:
            st.header("Capacity Forecast 🔮")
            pc1, pc2 = st.columns(2)
            max_date = pd.to_datetime(daily_agg['Period']).max().date()
            default_future = max_date + timedelta(days=30)
            with pc1: pred_start = st.date_input("Prediction Start", max_date)
            with pc2: pred_end = st.date_input("Target Date", default_future, min_value=pred_start)
            
            dates, proj_act, proj_peak, proj_tgt, start_val, daily_rate, peak_rate = cr_utils.generate_prediction_data(
                daily_agg, pred_start, pred_end, demand_info['target']
            )
            
            fig_pred = cr_utils.plot_prediction_chart(dates, proj_act, proj_peak, proj_tgt, demand_info['target'], pred_start, start_val)
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
    st.sidebar.title("Capacity Risk v6.1")
    
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
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Demand Targets")
    demand_info = {
        'target': st.sidebar.number_input("Target Demand", min_value=0),
        'received': st.sidebar.number_input("Received Parts", min_value=0)
    }

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