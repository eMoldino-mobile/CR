import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
__version__ = "4.6" # UPDATED VERSION
# ==================================================================

# ==================================================================
#                        HELPER FUNCTION
# ==================================================================

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
    return " ".join(parts)

# ==================================================================
#                        DATA CALCULATION
# ==================================================================

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            uploaded_file.seek(0) # Reset file pointer for reading
            df = pd.read_csv(uploaded_file, header=0)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            uploaded_file.seek(0) # Reset file pointer for reading
            df = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Error: Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Caching is REMOVED from the core calculation function.
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc):
    """
    Core function to process the raw DataFrame and calculate all Capacity Risk fields
    using the "Net Loss" model.
    """

    # --- 1. Standardize and Prepare Data ---
    df = _df_raw.copy()

    # --- Flexible Column Name Mapping ---
    column_variations = {
        'SHOT TIME': ['shot time', 'shot_time', 'timestamp', 'datetime'],
        'Approved CT': ['approved ct', 'approved_ct', 'approved cycle time', 'std ct', 'standard ct'],
        'Actual CT': ['actual ct', 'actual_ct', 'actual cycle time', 'cycle time', 'ct'],
        'Working Cavities': ['working cavities', 'working_cavities', 'cavities', 'cavity'],
        'Plant Area': ['plant area', 'plant_area', 'area']
    }

    rename_dict = {}
    found_cols = {}

    for standard_name, variations in column_variations.items():
        found = False
        for col in df.columns:
            col_str_lower = str(col).strip().lower()
            if col_str_lower in variations:
                rename_dict[col] = standard_name
                found_cols[standard_name] = True
                found = True
                break
        if not found:
            found_cols[standard_name] = False

    df.rename(columns=rename_dict, inplace=True)

    # --- 2. Check for Required Columns ---
    required_cols = ['SHOT TIME', 'Approved CT', 'Actual CT']
    missing_cols = [col for col in required_cols if not found_cols.get(col)]

    if missing_cols:
        st.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return None, None

    # --- 3. Handle Optional Columns and Data Types ---
    if not found_cols.get('Working Cavities'):
        st.info(f"'Working Cavities' column not found. Using default value: {default_cavities}")
        df['Working Cavities'] = default_cavities
    else:
        df['Working Cavities'].fillna(1, inplace=True)

    if not found_cols.get('Plant Area'):
        if toggle_filter:
            st.warning("'Plant Area' column not found. Cannot apply Maintenance/Warehouse filter.")
            toggle_filter = False
        df['Plant Area'] = 'Production'
    else:
        df['Plant Area'].fillna('Production', inplace=True)

    try:
        df['SHOT TIME'] = pd.to_datetime(df['SHOT TIME'])
        df['Actual CT'] = pd.to_numeric(df['Actual CT'])
        df['Approved CT'] = pd.to_numeric(df['Approved CT'])
        df['Working Cavities'] = pd.to_numeric(df['Working Cavities'])
    except Exception as e:
        st.error(f"Error converting data types: {e}. Check for non-numeric values in CT or Cavities columns.")
        return None, None


    # --- 4. Apply Filters (The Toggle) ---

    if df.empty or len(df) < 2:
        st.error("Error: Not enough data in the file to calculate run time.")
        return None, None

    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty:
        st.error("Error: No 'Production' data found after filtering.")
        return None, None

    # --- 5. Group by Day ---
    df_production_only['date'] = df_production_only['SHOT TIME'].dt.date

    daily_results_list = []
    all_valid_shots_list = []

    # Iterate over each day's data
    for date, daily_df in df_production_only.groupby('date'):

        results = {}
        results['Date'] = date

        # Create the final 'valid' dataframe (for cycle calcs)
        df_valid = daily_df[daily_df['Actual CT'] < 999.9].copy()

        if df_valid.empty or len(daily_df) < 2:
            results['Total Shots (all)'] = len(daily_df); results['Valid Shots (non 999.9)'] = 0
            daily_results_list.append(results)
            continue

        # --- 6. Get Daily Configuration Values ---
        try:
            APPROVED_CT = df_valid['Approved CT'].mode().iloc[0]
        except IndexError:
            APPROVED_CT = daily_df['Approved CT'].mode().iloc[0]

        PERFORMANCE_BENCHMARK = APPROVED_CT
        max_cavities = daily_df['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1 # Fallback

        # --- 7. Calculate Per-Shot Metrics (for this day) ---

        df_valid['parts_gain'] = np.where(
            df_valid['Actual CT'] < PERFORMANCE_BENCHMARK,
            ((PERFORMANCE_BENCHMARK - df_valid['Actual CT']) / PERFORMANCE_BENCHMARK) * max_cavities,
            0
        )
        df_valid['parts_loss'] = np.where(
            df_valid['Actual CT'] > PERFORMANCE_BENCHMARK,
            ((df_valid['Actual CT'] - PERFORMANCE_BENCHMARK) / PERFORMANCE_BENCHMARK) * max_cavities,
            0
        )
        df_valid['actual_output'] = df_valid['Working Cavities']

        conditions = [
            (df_valid['Actual CT'] > PERFORMANCE_BENCHMARK),
            (df_valid['Actual CT'] < PERFORMANCE_BENCHMARK),
            (df_valid['Actual CT'] == PERFORMANCE_BENCHMARK)
        ]
        choices = ['Slow', 'Fast', 'On Target']
        df_valid['Shot Type'] = np.select(conditions, choices, default='N/A')

        df_valid['Approved CT'] = APPROVED_CT
        all_valid_shots_list.append(df_valid)

        # --- 8. Calculate Daily Aggregates ---

        # A. Basic Counts
        results['Total Shots (all)'] = len(daily_df)
        results['Valid Shots (non 999.9)'] = len(df_valid)
        results['Invalid Shots (999.9 removed)'] = results['Total Shots (all)'] - results['Valid Shots (non 999.9)']

        # B. Time Calculations
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0

        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        results['Filtered Run Time (sec)'] = time_span_sec + last_shot_ct

        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()

        # Fix for negative downtime (float rounding)
        downtime_sec_calc = results['Filtered Run Time (sec)'] - results['Actual Cycle Time Total (sec)']
        results['Capacity Loss (downtime) (sec)'] = np.maximum(0, downtime_sec_calc)


        # C. Output Calculations
        results['Actual Output (parts)'] = df_valid['actual_output'].sum()
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / APPROVED_CT) * max_cavities

        # D. Loss & Gap Calculations
        results['Capacity Loss (downtime) (parts)'] = (results['Capacity Loss (downtime) (sec)'] / APPROVED_CT) * max_cavities
        results['Capacity Loss (slow cycle time) (parts)'] = df_valid['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = df_valid['parts_gain'].sum()

        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']

        # Total Capacity Loss (sec)
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        net_cycle_loss_sec = (net_cycle_loss_parts * APPROVED_CT) / max_cavities if max_cavities > 0 else 0
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # F. Target
        results['Target Output (parts)'] = results['Optimal Output (parts)'] * (target_output_perc / 100.0)

        # Capacity Loss vs Target
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        results['Capacity Loss (vs Target) (parts)'] = results['Target Output (parts)'] - results['Actual Output (parts)']
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * APPROVED_CT) / max_cavities if max_cavities > 0 else 0

        daily_results_list.append(results)

    # --- 9. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list).fillna(0) # Fill NaNs with 0
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')

    if not all_valid_shots_list:
        return final_df, pd.DataFrame()

    all_shots_df = pd.concat(all_valid_shots_list, ignore_index=True)
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date

    return final_df, all_shots_df

# ==================================================================
#                        CACHING WRAPPER
# ==================================================================

@st.cache_data
def run_capacity_calculation(raw_data_df, toggle, cavities, target_perc):
    return calculate_capacity_risk(
        raw_data_df,
        toggle,
        cavities,
        target_perc
    )


# ==================================================================
#                        STREAMLIT APP LAYOUT
# ==================================================================

# --- Page Config ---
st.set_page_config(
    page_title="Capacity Risk Calculator",
    layout="wide"
)

st.title("Capacity Risk Report")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Raw Data File (CSV or Excel)", type=["csv", "xlsx", "xls"])

data_frequency = st.sidebar.radio(
    "Select Graph Frequency",
    ['Daily', 'Weekly', 'Monthly'],
    index=0,
    horizontal=True
)

st.sidebar.markdown("---")

# --- DEFAULT UPDATED: value=False ---
toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots",
    value=False, # <-- SET TO FALSE (OFF)
    help="If ON, all calculations will exclude shots where 'Plant Area' is 'Maintenance' or 'Warehouse'."
)

default_cavities = st.sidebar.number_input(
    "Default Working Cavities",
    min_value=1,
    value=2,
    help="This value will be used if the 'Working Cavities' column is not found in the file."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Target & View")

# --- Global Benchmark Selector (Default is Optimal Output) ---
benchmark_view = st.sidebar.radio(
    "Select Report Benchmark",
    ['Optimal Output', 'Target Output'],
    index=0, # <-- SET TO 0 (Optimal Output)
    horizontal=False,
    help="Select the benchmark to compare against (e.g., 'Total Capacity Loss' vs 'Optimal' or 'Target')."
)

# --- CONDITIONAL SLIDER ---
if benchmark_view == "Target Output":
    # --- DEFAULT UPDATED: value=90.0 ---
    target_output_perc = st.sidebar.slider(
        "Target Output % (of Optimal)",
        min_value=0.0, max_value=100.0,
        value=90.0, # <-- SET TO 90%
        step=1.0,
        format="%.0f%%",
        help="Sets the 'Target Output (parts)' goal as a percentage of 'Optimal Output (parts)'."
    )
else:
    target_output_perc = 100.0 
    
st.sidebar.caption(f"App Version: **{__version__}**")


# --- Main Page Display ---
if uploaded_file is not None:

    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")

        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk..."):

            # CALL THE CACHED WRAPPER FUNCTION
            results_df, all_shots_df = run_capacity_calculation(
                df_raw,
                toggle_filter,
                default_cavities,
                target_output_perc
            )

            if results_df is not None and not results_df.empty:

                # --- 1. All-Time Summary Dashboard Calculations ---
                st.header("All-Time Summary")

                # 1. Calculate totals
                total_produced = results_df['Actual Output (parts)'].sum()
                total_downtime_loss = results_df['Capacity Loss (downtime) (parts)'].sum()
                total_slow_loss = results_df['Capacity Loss (slow cycle time) (parts)'].sum()
                total_fast_gain = results_df['Capacity Gain (fast cycle time) (parts)'].sum()
                total_net_cycle_loss = total_slow_loss - total_fast_gain
                
                # Total positive net cycle loss for stacked chart only
                total_positive_net_cycle_loss = np.maximum(0, total_net_cycle_loss)

                total_optimal = results_df['Optimal Output (parts)'].sum()
                total_target = results_df['Target Output (parts)'].sum() # This will be equal to total_optimal if target_output_perc=100

                total_loss_parts = results_df['Total Capacity Loss (parts)'].sum()
                total_loss_sec = results_df['Total Capacity Loss (sec)'].sum()
                total_loss_dhm = format_seconds_to_dhm(total_loss_sec)

                total_actual_ct_sec = results_df['Actual Cycle Time Total (sec)'].sum()
                total_actual_ct_dhm = format_seconds_to_dhm(total_actual_ct_sec)

                # Use sum() on the underlying time column for correct time aggregation
                run_time_sec_total = results_df['Filtered Run Time (sec)'].sum()
                run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)
                run_time_label = "Filtered Run Time" if toggle_filter else "Overall Run Time"

                # 2. Calculate percentages for metrics
                actual_ct_perc_val = (total_actual_ct_sec / run_time_sec_total) if run_time_sec_total > 0 else 0
                actual_output_perc_val = (total_produced / total_optimal) if total_optimal > 0 else 0
                loss_time_perc_val = (total_loss_sec / run_time_sec_total) if run_time_sec_total > 0 else 0
                loss_parts_perc_val = (total_loss_parts / total_optimal) if total_optimal > 0 else 0

                # 3. Create 3 columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(run_time_label, run_time_dhm_total)
                    st.metric(f"Actual Cycle Time Total ({actual_ct_perc_val:.1%})", total_actual_ct_dhm)

                with col2:
                    # --- DYNAMIC TARGET/OPTIMAL METRIC ---
                    if benchmark_view == "Target Output":
                        st.metric("Target Output (parts)", f"{total_target:,.0f}")
                    else:
                        st.metric("Optimal Output (parts) (100%)", f"{total_optimal:,.0f}")
                        
                    st.metric(f"Actual Output (parts) ({actual_output_perc_val:.1%})", f"{total_produced:,.0f}")

                with col3:
                    # --- Dynamic Total Capacity Loss Metric (vs Optimal or Target) ---
                    if benchmark_view == "Optimal Output":
                        st.metric(f"Total Capacity Loss (Time) ({loss_time_perc_val:.1%})", total_loss_dhm)
                        st.metric(
                            f"Total Capacity Loss (parts) ({loss_parts_perc_val:.1%})",
                            f"{total_loss_parts:,.0f}",
                            delta=f"{-total_loss_parts:,.0f}",
                            delta_color="inverse"
                        )
                    else: # Target Output
                        total_loss_vs_target_sec = results_df['Capacity Loss (vs Target) (sec)'].sum()
                        total_loss_vs_target_dhm = format_seconds_to_dhm(total_loss_vs_target_sec)
                        total_loss_vs_target_parts = results_df['Capacity Loss (vs Target) (parts)'].sum()
                        
                        loss_vs_target_time_perc_val = (total_loss_vs_target_sec / run_time_sec_total) if run_time_sec_total > 0 else 0
                        loss_vs_target_parts_perc_val = (total_loss_vs_target_parts / total_target) if total_target > 0 else 0

                        st.metric(f"Capacity Loss (vs Target) (Time) ({loss_vs_target_time_perc_val:.1%})", total_loss_vs_target_dhm)
                        st.metric(
                            f"Capacity Loss (vs Target) (parts) ({loss_vs_target_parts_perc_val:.1%})",
                            f"{total_loss_vs_target_parts:,.0f}",
                            delta=f"{-total_loss_vs_target_parts:,.0f}", 
                            delta_color=("inverse" if total_loss_vs_target_parts > 0 else "normal")
                        )


                # --- Collapsible Daily Summary Table ---
                with st.expander("View Daily Summary Data"):

                    # Create a new summary DF for this table
                    daily_summary_df = results_df.copy()

                    # Calculate all % and formatted columns needed for the table
                    daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    daily_summary_df['Actual Output (parts %)'] = np.where( daily_summary_df['Optimal Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / daily_summary_df['Optimal Output (parts)'], 0 )
                    daily_summary_df['Total Capacity Loss (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Total Capacity Loss (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    daily_summary_df['Total Capacity Loss (parts %)'] = np.where( daily_summary_df['Optimal Output (parts)'] > 0, daily_summary_df['Total Capacity Loss (parts)'] / daily_summary_df['Optimal Output (parts)'], 0 )
                    daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)

                    daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( daily_summary_df['Target Output (parts)'] > 0, daily_summary_df['Capacity Loss (vs Target) (parts)'] / daily_summary_df['Target Output (parts)'], 0 )
                    daily_summary_df['Capacity Loss (vs Target) (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Capacity Loss (vs Target) (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = daily_summary_df['Capacity Loss (vs Target) (sec)'].apply(format_seconds_to_dhm)

                    daily_summary_df['Filtered Run Time (d/h/m)'] = daily_summary_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                    daily_summary_df['Actual Cycle Time Total (d/h/m)'] = daily_summary_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                    # Build the final table
                    daily_kpi_table = pd.DataFrame(index=daily_summary_df.index)
                    daily_kpi_table[run_time_label] = daily_summary_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                    daily_kpi_table['Actual Cycle Time Total'] = daily_summary_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                    
                    # --- CONDITIONAL COLUMN ---
                    if benchmark_view == "Target Output":
                        daily_kpi_table['Target Output (parts)'] = daily_summary_df['Target Output (parts)'].map('{:,.2f}'.format)
                        
                    daily_kpi_table['Actual Output (parts)'] = daily_summary_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (parts %)']:.1%})", axis=1)

                    # Dynamic Column Logic
                    if benchmark_view == "Optimal Output":
                        daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                        daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                    else: # Target Output
                        daily_kpi_table['Capacity Loss (vs Target) (Time)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (d/h/m)']} ({r['Capacity Loss (vs Target) (time %)']:.1%})", axis=1)
                        daily_kpi_table['Capacity Loss (vs Target) (parts)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (parts)']:,.2f} ({r['Capacity Loss (vs Target) (parts %)']:.1%})", axis=1)

                    st.dataframe(daily_kpi_table, use_container_width=True)

                st.divider()

                # --- 2. Production Output Overview Chart (Build-Up Chart) ---
                st.header("Production Output Overview (parts)")

                # --- Final Bar Dynamic Check ---
                if benchmark_view == 'Optimal Output':
                    benchmark_value = total_optimal
                    benchmark_label = "Optimal"
                    benchmark_hover_name = "Optimal Output (parts)"
                else:
                    benchmark_value = total_target
                    benchmark_label = "Target"
                    benchmark_hover_name = "Target Output (parts)"

                # --- Dynamic Loss Check for Build-up Chart ---
                downtime_loss_chart = total_downtime_loss
                net_cycle_loss_chart = total_net_cycle_loss
                
                # If Actual Output exceeds the Target, the capacity loss is visually zero
                if benchmark_view == 'Target Output' and total_produced > total_target:
                    downtime_loss_chart = 0
                    net_cycle_loss_chart = 0
                    
                # Ensure net cycle loss for the stacked chart is at least 0 (no negative stacking)
                positive_net_cycle_loss_chart = np.maximum(0, net_cycle_loss_chart)
                
                # --- Bar Build-up Chart ---
                categories = [
                    'Actual Output (parts)',
                    'Capacity Loss (slow/fast cycle time)',
                    'Capacity Loss (downtime)',
                    benchmark_hover_name
                ]

                fig_summary = go.Figure()

                # --- Bar 1: Actual Production ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[0]], y=[total_produced], name='Actual Output (parts)',
                    marker_color='green', text=[f"{total_produced:,.0f}<br>Actual Output"],
                    textposition='auto', hovertemplate='<b>Actual Output (parts)</b><br>Parts: %{y:,.0f}<extra></extra>'
                ))

                # --- Bar 2: Net Cycle Loss (Stacked) ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[1]], y=[total_produced], name='Base (Produced)',
                    marker_color='rgba(0, 128, 0, 0.2)', showlegend=False, hoverinfo='none'
                ))
                fig_summary.add_trace(go.Bar(
                    x=[categories[1]], y=[positive_net_cycle_loss_chart], name='Capacity Loss (slow/fast cycle time)',
                    marker_color='gold', 
                    text=[f"{positive_net_cycle_loss_chart:,.0f}<br>Parts Lost"] if positive_net_cycle_loss_chart > 0 else None,
                    textposition='auto', 
                    # Use actual net loss for hover, even if chart draws 0
                    hovertemplate='<b>Capacity Loss (slow/fast cycle time)</b><br>Slow Loss: %{customdata[0]:,.0f}<br>Fast Gain: -%{customdata[1]:,.0f}<br><b>Net: %{customdata[2]:,.0f}</b><extra></extra>',
                    customdata=np.array([[total_slow_loss, total_fast_gain, total_net_cycle_loss]])
                ))

                # --- Bar 3: Downtime Loss (Stacked) ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[total_produced], name='Base (Produced)',
                    marker_color='rgba(0, 128, 0, 0.2)', showlegend=False, hoverinfo='none'
                ))
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[net_cycle_loss_chart], name='Base (Cycle Loss)',
                    marker_color='rgba(255, 215, 0, 0.2)', showlegend=False, hoverinfo='none'
                ))
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[downtime_loss_chart], name='Capacity Loss (downtime)',
                    marker_color='red', 
                    text=[f"{downtime_loss_chart:,.0f}<br>Parts Lost"] if downtime_loss_chart > 0 else None,
                    textposition='auto', hovertemplate='<b>Capacity Loss (downtime)</b><br>Parts: %{y:,.0f}<extra></extra>'
                ))

                # --- Bar 4: Dynamic Benchmark ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[3]], y=[benchmark_value], name=benchmark_hover_name,
                    marker_color='grey', text=[f"{benchmark_value:,.0f}<br>{benchmark_label}"],
                    textposition='auto', hovertemplate=f'<b>{benchmark_hover_name}</b><br>Parts: %{{y:,.0f}}<extra></extra>'
                ))

                fig_summary.update_layout(barmode='stack')

                # --- Add Horizontal Lines ---
                fig_summary.add_shape(
                    type='line',
                    x0=-0.5, x1=3.5, # Span all 4 bars
                    y0=total_target, y1=total_target,
                    line=dict(color='blue', dash='dash'),
                    name='Target Output (parts)'
                )
                fig_summary.add_trace(go.Scatter(
                    x=[3.5], y=[total_target],
                    mode='lines', line=dict(color='blue', dash='dash'),
                    name='Target Output (parts)',
                    hoverinfo='skip'
                ))

                fig_summary.add_shape(
                    type='line',
                    x0=-0.5, x1=3.5,
                    y0=total_optimal, y1=total_optimal,
                    line=dict(color='purple', dash='dot'),
                    name='Optimal Output (parts)'
                )
                fig_summary.add_trace(go.Scatter(
                    x=[3.5], y=[total_optimal],
                    mode='lines', line=dict(color='purple', dash='dot'),
                    name='Optimal Output (parts)',
                    hoverinfo='skip'
                ))

                fig_summary.update_layout(
                    title='Production Output Overview (All Time)',
                    yaxis_title='Parts',
                    showlegend=True,
                    legend_title='Metric',
                    xaxis=dict(
                        categoryorder='array', # Enforce the category order
                        categoryarray=categories
                    )
                )

                st.plotly_chart(fig_summary, use_container_width=True)

                st.divider()

                # --- 3. AGGREGATED REPORT (Chart & Table) ---

                # --- Aggregate data based on frequency selection ---
                if data_frequency == 'Weekly':
                    agg_df = results_df.resample('W').sum()
                    chart_title = "Weekly Capacity Report"
                    xaxis_title = "Week"
                    display_df = agg_df
                elif data_frequency == 'Monthly':
                    agg_df = results_df.resample('ME').sum()
                    chart_title = "Monthly Capacity Report"
                    xaxis_title = "Month"
                    display_df = agg_df
                else: # Daily
                    display_df = results_df
                    chart_title = "Daily Capacity Report"
                    xaxis_title = "Date"

                # --- Calculate Percentage Columns AFTER aggregation ---
                display_df['Actual Output (%)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0,
                    display_df['Actual Output (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                display_df['Valid Shots (non 999.9) (%)'] = np.where(
                    display_df['Total Shots (all)'] > 0,
                    display_df['Valid Shots (non 999.9)'] / display_df['Total Shots (all)'], 0
                )
                display_df['Actual Cycle Time Total (time %)'] = np.where(
                    display_df['Filtered Run Time (sec)'] > 0,
                    display_df['Actual Cycle Time Total (sec)'] / display_df['Filtered Run Time (sec)'], 0
                )

                display_df['Capacity Loss (downtime) (parts %)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0,
                    display_df['Capacity Loss (downtime) (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                display_df['Capacity Loss (slow cycle time) (parts %)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0,
                    display_df['Capacity Loss (slow cycle time) (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                display_df['Capacity Gain (fast cycle time) (parts %)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0,
                    display_df['Capacity Gain (fast cycle time) (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                display_df['Total Capacity Loss (parts %)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0,
                    display_df['Total Capacity Loss (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                
                # Add Capacity Loss vs Target %
                display_df['Capacity Loss (vs Target) (parts %)'] = np.where(
                    display_df['Target Output (parts)'] > 0,
                    display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0
                )


                _target_output_perc_array = np.full(len(display_df), target_output_perc / 100.0)

                display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                chart_df = display_df.reset_index()

                # --- Performance Breakdown Chart (Time Series) ---
                st.header(f"{data_frequency} Performance Breakdown")
                fig = go.Figure()

                # --- Dynamic Chart Logic ---
                if benchmark_view == "Optimal Output":
                    # --- STACKED BAR CHART (vs Optimal) ---
                    fig.add_trace(go.Bar(
                        x=chart_df['Date'],
                        y=chart_df['Actual Output (parts)'],
                        name='Actual Output (parts)',
                        marker_color='green',
                        customdata=np.stack((
                            chart_df['Actual Output (%)']
                        ), axis=-1),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual Output (parts): %{y:,.0f} (%{customdata[0]:.1%})<extra></extra>'
                    ))
                    fig.add_trace(go.Bar(
                        x=chart_df['Date'],
                        y=chart_df['Total Capacity Loss (parts)'],
                        name='Total Capacity Loss (Net)',
                        marker_color='red',
                        customdata=np.stack((
                            chart_df['Total Capacity Loss (parts %)'],
                            chart_df['Capacity Loss (downtime) (parts)'],
                            chart_df['Capacity Loss (slow cycle time) (parts)'],
                            chart_df['Capacity Gain (fast cycle time) (parts)']
                        ), axis=-1),
                        hovertemplate=
                            '<b>%{x|%Y-%m-%d}</b><br>' +
                            '<b>Total Net Loss: %{y:,.0f} (%{customdata[0]:.1%})</b><br><br>' +
                            'Loss (Downtime): %{customdata[1]:,.0f}<br>' +
                            'Loss (Slow Cycles): %{customdata[2]:,.0f}<br>' +
                            'Gain (Fast Cycles): -%{customdata[3]:,.0f}<br>' +
                            '<extra></extra>'
                    ))
                    fig.update_layout(barmode='stack')
                else: # Target View (FIXED: Added Actual Output Bar)
                    
                    # 1. ACTUAL OUTPUT (Stack Base for Loss/Gain Bar)
                    fig.add_trace(go.Bar(
                        x=chart_df['Date'],
                        y=chart_df['Actual Output (parts)'],
                        name='Actual Output (parts)',
                        marker_color='green',
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual Output (parts): %{y:,.0f}<extra></extra>'
                    ))
                    
                    # 2. Capacity Loss/Gain (vs Target) - Drawn relative to Actual Output
                    fig.add_trace(go.Bar(
                        x=chart_df['Date'],
                        y=chart_df['Capacity Loss (vs Target) (parts)'],
                        name='Capacity Loss (vs Target)',
                        customdata=np.stack((
                            chart_df['Capacity Loss (vs Target) (parts %)'],
                        ), axis=-1),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Capacity Loss (vs Target): %{y:,.0f} (%{customdata[0]:.1%})<extra></extra>'
                    ))
                    
                    # Color the Loss/Gain bars red or blue based on value
                    fig.update_traces(
                        selector=dict(name='Capacity Loss (vs Target)'),
                        marker_color=['red' if v > 0 else 'blue' for v in chart_df['Capacity Loss (vs Target) (parts)']] 
                    )
                    
                    fig.update_layout(barmode='relative') # Stacks the Loss/Gain bar on top of the Actual Output bar


                # --- OVERLAY LINES (ON PRIMARY Y-AXIS) ---
                if benchmark_view == "Target Output": # Only show Target line when Target view is selected
                    fig.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Target Output (parts)'],
                        name='Target Output (parts)',
                        mode='lines',
                        line=dict(color='blue', dash='dash'),
                        customdata=_target_output_perc_array,
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Target Output (parts): %{y:,.0f} (%{customdata:.0%})<extra></extra>'
                    ))
                    
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Optimal Output (parts)'],
                    name='Optimal Output (100%)',
                    mode='lines',
                    line=dict(color='purple', dash='dot'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Optimal Output (parts): %{y:,.0f}<extra></extra>'
                ))

                # --- LAYOUT UPDATE ---
                fig.update_layout(
                    title=chart_title,
                    xaxis_title=xaxis_title,
                    yaxis_title='Parts (Output & Loss)',
                    legend_title='Metric',
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Full Data Table (Open by Default) ---

                # --- Create Table 1 (Totals Report) ---
                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df.index)

                report_table_1['Total Shots (all)'] = display_df['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Valid Shots (non 999.9)'] = display_df.apply(lambda r: f"{r['Valid Shots (non 999.9)']:,.0f} ({r['Valid Shots (non 999.9) (%)']:.1%})", axis=1)
                report_table_1['Invalid Shots (999.9 removed)'] = display_df['Invalid Shots (999.9 removed)'].map('{:,.0f}'.format)
                report_table_1[run_time_label] = display_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Cycle Time Total'] = display_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                st.dataframe(report_table_1, use_container_width=True)

                # --- Create Table 2 (Capacity Loss Report) ---
                # Title is now dynamic
                table_2_title = "Capacity Loss & Gain Report" if benchmark_view == "Optimal Output" else "Capacity Loss (vs Target) Report"
                st.header(f"{table_2_title} ({data_frequency})")

                report_table_2 = pd.DataFrame(index=display_df.index)

                report_table_2['Optimal Output (parts)'] = display_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                
                # --- CONDITIONAL COLUMN ---
                if benchmark_view == "Target Output":
                    report_table_2['Target Output (parts)'] = display_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f} ({target_output_perc / 100.0:.0%})", axis=1)
                    
                report_table_2['Actual Output (parts)'] = display_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)

                # --- Dynamic Columns ---
                if benchmark_view == "Optimal Output":
                    
                    report_table_2['Capacity Loss (downtime)'] = display_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                    report_table_2['Capacity Loss (slow cycles)'] = display_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                    report_table_2['Capacity Gain (fast cycles)'] = display_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                    report_table_2['Total Capacity Loss (Net)'] = display_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                else: # Target View
                    report_table_2['Capacity Loss (vs Target) (parts)'] = display_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (parts)']:,.2f} ({r['Capacity Loss (vs Target) (parts %)']:.1%})", axis=1)

                st.dataframe(report_table_2, use_container_width=True)

                # --- 4. SHOT-BY-SHOT ANALYSIS ---
                st.divider()
                st.header("Shot-by-Shot Cycle Time Analysis")

                if all_shots_df.empty:
                    st.warning("No valid shots were found in the file to analyze.")
                else:
                    # --- Create the daily date selector ---
                    available_dates = sorted(all_shots_df['date'].unique(), reverse=True)
                    selected_date = st.selectbox(
                        "Select a Date to Analyze",
                        options=available_dates,
                        format_func=lambda d: d.strftime('%Y-%m-%d') # Format for display
                    )

                    # --- NEW: Add Y-Axis Zoom Slider ---
                    st.subheader("Chart Controls")
                    df_day_shots_for_range = all_shots_df[all_shots_df['date'] == selected_date]
                    
                    # Set a sensible default max value based on the day's data, or use 50
                    max_ct_for_day = df_day_shots_for_range['Actual CT'].max()
                    slider_max = int(np.ceil(max_ct_for_day / 10.0)) * 10
                    slider_max = max(slider_max, 50)
                    slider_max = min(slider_max, 1000)

                    y_axis_max = st.slider(
                        "Zoom Y-Axis (sec)",
                        min_value=10,
                        max_value=1000, # Max to see all outliers
                        value=min(slider_max, 50), # Default to a "zoomed in" view
                        step=10,
                        help="Adjust the max Y-axis to zoom in on the cluster. (Set to 1000 to see all outliers)."
                    )

                    # Filter the shot data to the selected day
                    df_day_shots = df_day_shots_for_range.copy()

                    if df_day_shots.empty:
                        st.warning(f"No valid shots found for {selected_date}.")
                    else:
                        # Get the Approved CT for this specific day
                        approved_ct_for_day = df_day_shots['Approved CT'].iloc[0]

                        # --- Create the Plotly Scatter Chart ---
                        fig_ct = go.Figure()

                        # --- Define colors for each shot type ---
                        color_map = {
                            'Slow': 'red',
                            'Fast': 'gold',
                            'On Target': 'darkblue'
                        }

                        # Add traces for each shot type
                        for shot_type, color in color_map.items():
                            df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                            if not df_subset.empty:
                                fig_ct.add_trace(go.Bar(
                                    x=df_subset['SHOT TIME'],
                                    y=df_subset['Actual CT'],
                                    name=shot_type,
                                    marker_color=color,
                                    hovertemplate='<b>%{x|%H:%M:%S}</b><br>Actual CT: %{y:.2f}s<extra></extra>'
                                ))

                        # Add the green Approved CT line
                        fig_ct.add_shape(
                            type='line',
                            x0=df_day_shots['SHOT TIME'].min(),
                            x1=df_day_shots['SHOT TIME'].max(),
                            y0=approved_ct_for_day,
                            y1=approved_ct_for_day,
                            line=dict(color='green', dash='dash'),
                            name=f'Approved CT ({approved_ct_for_day}s)'
                        )

                        fig_ct.update_layout(
                            title=f'Shot-by-Shot Cycle Time for {selected_date}',
                            xaxis_title='Time of Day',
                            yaxis_title='Actual Cycle Time (sec)',
                            hovermode="closest",
                            yaxis_range=[0, y_axis_max], # Apply the zoom
                            barmode='overlay' # Ensure bars draw from 0
                        )
                        st.plotly_chart(fig_ct, use_container_width=True)

                        # --- Display the daily shot table ---
                        st.subheader(f"Data for all {len(df_day_shots)} valid shots on {selected_date}")
                        st.dataframe(
                            df_day_shots[[
                                'SHOT TIME',
                                'Actual CT',
                                'Approved CT',
                                'Working Cavities',
                                'Shot Type'
                            ]].style.format({
                                'Actual CT': '{:.2f}',
                                'Approved CT': '{:.1f}',
                                'SHOT TIME': lambda t: t.strftime('%H:%M:%S')
                            }),
                            use_container_width=True
                        )

            elif results_df is not None:
                st.warning("No valid data was found after filtering. Cannot display results.")

else:
    st.info("Please upload a data file to begin.")