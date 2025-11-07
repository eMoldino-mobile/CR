import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
#                        HELPER FUNCTION
# ==================================================================

def format_seconds_to_dhm(total_seconds):
    """Converts total seconds into a 'Xd Yh Zm' string."""
    # Ensure non-negative value for display
    total_seconds = max(0, total_seconds) 
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
    
    # Handle the case where total_seconds is a tiny negative float close to zero (for display only)
    if not parts and total_seconds < 0.1 and total_seconds > -0.1:
        return "0m" 
    
    return " ".join(parts)

# ==================================================================
#                        DATA CALCULATION
# ==================================================================

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=0)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # Requires openpyxl
            df = pd.read_excel(uploaded_file, header=0)
        else:
            st.error("Error: Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- Caching decorator ---
@st.cache_data
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc):
    """
    Main function to process the raw DataFrame and calculate all Capacity Risk fields.
    
    This version groups the calculations by day.
    """
    
    # --- 1. Standardize and Prepare Data (Flexible Column Matching) ---
    df = _df_raw.copy()
    
    column_variations = {
        'SHOT TIME': ['SHOT TIME', 'SHOT_TIME', 'SHOTTIME', 'Time'],
        'Approved CT': ['APPROVED CT', 'APPROVED_CT', 'Approved CT', 'Approved Cycle Time', 'CycleTimeAppr'],
        'Actual CT': ['ACTUAL CT', 'ACTUAL_CT', 'Actual CT', 'Actual Cycle Time', 'CycleTimeActual'],
        'Working Cavities': ['Working Cavities', 'CAVITIES', 'CAVITY', 'WorkingCavities'],
        'Plant Area': ['Plant Area', 'AREA', 'Location']
    }
    
    rename_dict = {}
    for standard_name, variations in column_variations.items():
        found = False
        for col in df.columns:
            # Check if the stripped, lowercased column name matches any variation
            if str(col).strip().lower() in [v.lower() for v in variations]:
                rename_dict[col] = standard_name
                found = True
                break
        
    df.rename(columns=rename_dict, inplace=True)

    # --- 2. Check for Required Columns ---
    required_cols = ['SHOT TIME', 'Approved CT', 'Actual CT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return None, None

    # --- 3. Handle Optional Columns and Data Types ---
    if 'Working Cavities' not in df.columns:
        st.info(f"'Working Cavities' column not found. Using default value: {default_cavities}")
        df['Working Cavities'] = default_cavities
    else:
        df['Working Cavities'].fillna(1, inplace=True) # Fill NaNs with 1

    if 'Plant Area' not in df.columns:
        if toggle_filter:
            st.warning("'Plant Area' column not found. Cannot apply Maintenance/Warehouse filter.")
            toggle_filter = False 
        df['Plant Area'] = 'Production'
    else:
        df['Plant Area'].fillna('Production', inplace=True) # Fill NaNs with 'Production'

    try:
        df['SHOT TIME'] = pd.to_datetime(df['SHOT TIME'])
        df['Actual CT'] = pd.to_numeric(df['Actual CT'], errors='coerce')
        df['Approved CT'] = pd.to_numeric(df['Approved CT'], errors='coerce')
        df['Working Cavities'] = pd.to_numeric(df['Working Cavities'], errors='coerce')
        
        # Drop rows where required numeric columns became NaN
        df.dropna(subset=['Actual CT', 'Approved CT', 'Working Cavities'], inplace=True)

    except Exception as e:
        st.error(f"Error converting data types: {e}. Check for non-numeric values in CT or Cavities columns.")
        return None, None
        
    # --- Overall Run Time (before filtering) ---
    first_shot_all = df['SHOT TIME'].min()
    last_shot_all = df['SHOT TIME'].max()
    last_shot_ct_all = df.loc[df['SHOT TIME'] == last_shot_all, 'Actual CT'].iloc[0] if not df.empty and not df.loc[df['SHOT TIME'] == last_shot_all, 'Actual CT'].empty else 0
    total_run_duration_overall = (last_shot_all - first_shot_all).total_seconds() + last_shot_ct_all

    # --- 4. Apply Filters (The Toggle) ---
    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty or len(df_production_only) < 2:
        st.error("Error: No 'Production' data found or insufficient data points after filtering.")
        return None, None

    # --- 5. Group by Day ---
    df_production_only['date'] = df_production_only['SHOT TIME'].dt.date
    
    daily_results_list = []
    all_valid_shots_list = [] 
    
    for date, daily_df in df_production_only.groupby('date'):
        
        results = {}
        results['Date'] = date
        
        # Create the final 'valid' dataframe (for cycle calcs)
        df_valid = daily_df[daily_df['Actual CT'] < 999.9].copy()

        if df_valid.empty or len(daily_df) < 2:
            results['Overall Run Time (sec)'] = 0
            results['Filtered Run Time (sec)'] = 0
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
        results['parts_gain_sum'] = df_valid['parts_gain'].sum()
        results['parts_loss_sum'] = df_valid['parts_loss'].sum()
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

        # B. Time Calculations (Filtered Run Time)
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        
        # Use Overall run time from pre-filtered block if no filter applied
        if not toggle_filter:
            results['Filtered Run Time (sec)'] = total_run_duration_overall 
            results['Overall Run Time (sec)'] = total_run_duration_overall
        else:
            # Filtered run time only applies to the production segment span
            results['Filtered Run Time (sec)'] = time_span_sec + last_shot_ct
            results['Overall Run Time (sec)'] = total_run_duration_overall 
        
        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
        
        # Capacity Loss (downtime) (sec) - ENSURE non-negative using np.maximum(0, ...)
        calculated_downtime_loss_sec = results['Filtered Run Time (sec)'] - results['Actual Cycle Time Total (sec)']
        results['Capacity Loss (downtime) (sec)'] = np.maximum(0, calculated_downtime_loss_sec)

        # C. Output Calculations
        results['Actual Output (parts)'] = df_valid['actual_output'].sum() 
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / APPROVED_CT) * max_cavities

        # D. Loss & Gap Calculations (all against Optimal)
        results['Capacity Loss (downtime) (parts)'] = (results['Capacity Loss (downtime) (sec)'] / APPROVED_CT) * max_cavities 
        results['Capacity Loss (slow cycle time) (parts)'] = results['parts_loss_sum']
        results['Capacity Gain (fast cycle time) (parts)'] = results['parts_gain_sum']
        
        # Total Capacity Loss (Net) vs. Optimal
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        
        # Total Capacity Loss (Net) in seconds (vs. Optimal)
        # Loss in Time = Downtime + (Net Cycle Loss in parts * CT / Cavities)
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + ((results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']) * APPROVED_CT) / max_cavities
        
        # E. Target
        results['Target Output (parts)'] = results['Optimal Output (parts)'] * (target_output_perc / 100.0)

        # F. Capacity Loss vs Target
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

toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots", 
    value=True, 
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

target_output_perc = st.sidebar.slider(
    "Target Output % (of Optimal)", 
    min_value=0.0, max_value=100.0, 
    value=85.0, 
    step=1.0,
    format="%.0f%%",
    help="Sets the 'Target Output' goal as a percentage of 'Optimal Output'."
)

benchmark_view = st.sidebar.radio(
    "Select Report Benchmark",
    ['Optimal Output', 'Target Output'],
    index=0,
    help="Determines whether 'Capacity Loss' metrics are calculated relative to 100% Optimal Output or the Target Output."
)


# --- Main Page Display ---
if uploaded_file is not None:
    
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")
        
        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk..."):
            
            results_df, all_shots_df = calculate_capacity_risk(
                df_raw, 
                toggle_filter, 
                default_cavities, 
                target_output_perc
            )
            
            if results_df is not None and not results_df.empty:
                
                # --- Aggregate All-Time Totals ---
                total_produced = results_df['Actual Output (parts)'].sum()
                total_optimal = results_df['Optimal Output (parts)'].sum()
                total_target = results_df['Target Output (parts)'].sum()
                
                # Loss components vs Optimal
                total_downtime_loss = results_df['Capacity Loss (downtime) (parts)'].sum()
                total_slow_loss = results_df['Capacity Loss (slow cycle time) (parts)'].sum()
                total_fast_gain = results_df['Capacity Gain (fast cycle time) (parts)'].sum()
                total_net_cycle_loss = total_slow_loss - total_fast_gain 
                
                # Positive Net Cycle Loss (for chart stacking purposes)
                total_positive_net_cycle_loss = np.maximum(0, total_net_cycle_loss)
                
                # Loss metrics vs Optimal
                total_loss_parts = results_df['Total Capacity Loss (parts)'].sum()
                total_loss_sec = results_df['Total Capacity Loss (sec)'].sum()
                total_loss_dhm = format_seconds_to_dhm(total_loss_sec)
                
                run_time_sec = results_df['Filtered Run Time (sec)'].sum() if toggle_filter else results_df['Overall Run Time (sec)'].sum()

                # --- 0. All-Time Summary Dashboard ---
                st.header("All-Time Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # 1. Run Time
                run_time_dhm = format_seconds_to_dhm(run_time_sec)
                run_time_label = "Filtered Run Time" if toggle_filter else "Overall Run Time"
                col1.metric(run_time_label, run_time_dhm)

                # 2. Actual Cycle Time Total
                actual_ct_total_sec = results_df['Actual Cycle Time Total (sec)'].sum()
                actual_ct_total_dhm = format_seconds_to_dhm(actual_ct_total_sec)
                actual_ct_time_perc = (actual_ct_total_sec / run_time_sec) if run_time_sec > 0 else 0
                col1.metric("Actual Cycle Time Total", f"{actual_ct_total_dhm} ({actual_ct_time_perc:.1%})")

                # 3. Actual vs Target/Optimal Output
                actual_output_perc = (total_produced / total_optimal) if total_optimal > 0 else 0
                col2.metric("Actual Output (parts)", f"{total_produced:,.0f} ({actual_output_perc:.1%})")
                
                if benchmark_view == "Optimal Output":
                    col2.metric("Optimal Output (parts)", f"{total_optimal:,.0f}")
                else: # Target Output
                    target_output_perc_overall = (total_target / total_optimal) if total_optimal > 0 else 0
                    col2.metric("Target Output (parts)", f"{total_target:,.0f} ({target_output_perc_overall:.1%})")

                # 4. Total Capacity Loss (vs Benchmark)
                if benchmark_view == "Optimal Output":
                    loss_time_perc_val = (total_loss_sec / run_time_sec) if run_time_sec > 0 else 0
                    loss_parts_perc_val = (total_loss_parts / total_optimal) if total_optimal > 0 else 0
                    
                    col3.metric("Total Capacity Loss (Time)", f"{total_loss_dhm} ({loss_time_perc_val:.1%})")
                    col3.metric("Total Capacity Loss (parts)", f"{total_loss_parts:,.0f}", delta=f"{-total_loss_parts:,.0f}", delta_color="inverse")
                else: # Target Output
                    total_loss_vs_target_sec = results_df['Capacity Loss (vs Target) (sec)'].sum()
                    total_loss_vs_target_dhm = format_seconds_to_dhm(total_loss_vs_target_sec)
                    total_loss_vs_target_parts = results_df['Capacity Loss (vs Target) (parts)'].sum()

                    target_loss_run_time = results_df['Target Output (parts)'].sum() * (results_df['Approved CT'].mode().iloc[0] / results_df['Working Cavities'].max()) # Approximate time target requires
                    loss_vs_target_time_perc_val = (total_loss_vs_target_sec / run_time_sec) if run_time_sec > 0 else 0
                    loss_vs_target_parts_perc_val = (total_loss_vs_target_parts / total_target) if total_target > 0 else 0
                    
                    col3.metric(f"Capacity Loss (vs Target) (Time) ({loss_vs_target_time_perc_val:.1%})", total_loss_vs_target_dhm)
                    col3.metric(
                        f"Capacity Loss (vs Target) (parts) ({loss_vs_target_parts_perc_val:.1%})", 
                        f"{total_loss_vs_target_parts:,.0f}", 
                        delta=f"{-total_loss_vs_target_parts:,.0f}", # Show gain/loss
                        delta_color=("inverse" if total_loss_vs_target_parts > 0 else "normal") # Red if loss > 0, Green if loss < 0
                    )
                
                st.divider()

                # --- Collapsed Daily Summary Table (vs Benchmark) ---
                with st.expander("View Daily Summary Data"):
                    # Calculate required columns for the table
                    daily_summary_df = results_df.copy()
                    
                    # Ensure percentages are calculated correctly for time-based totals
                    daily_summary_df['Total Capacity Loss (time %)'] = np.where(
                        daily_summary_df['Filtered Run Time (sec)'] > 0,
                        daily_summary_df['Total Capacity Loss (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0
                    )
                    daily_summary_df['Total Capacity Loss (parts %)'] = np.where(
                        daily_summary_df['Optimal Output (parts)'] > 0,
                        daily_summary_df['Total Capacity Loss (parts)'] / daily_summary_df['Optimal Output (parts)'], 0
                    )
                    daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( 
                        daily_summary_df['Target Output (parts)'] > 0, 
                        daily_summary_df['Capacity Loss (vs Target) (parts)'] / daily_summary_df['Target Output (parts)'], 0 
                    )
                    daily_summary_df['Capacity Loss (vs Target) (time %)'] = np.where( 
                        daily_summary_df['Filtered Run Time (sec)'] > 0, 
                        daily_summary_df['Capacity Loss (vs Target) (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 
                    )
                    
                    daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)
                    daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = daily_summary_df['Capacity Loss (vs Target) (sec)'].apply(format_seconds_to_dhm)
                    daily_summary_df['Filtered Run Time (d/h/m)'] = daily_summary_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)

                    # Build the table based on required KPIs
                    daily_kpi_table = pd.DataFrame(index=daily_summary_df.index)
                    daily_kpi_table.index = daily_kpi_table.index.strftime('%Y-%m-%d')
                    daily_kpi_table.index.name = "Date"

                    # Run Time
                    daily_kpi_table['Run Time'] = daily_summary_df['Filtered Run Time (d/h/m)']
                    
                    # Actual CT Total
                    daily_kpi_table['Actual Cycle Time Total'] = daily_summary_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                    # Output (Always show Optimal and Actual)
                    daily_kpi_table['Optimal Output (parts)'] = daily_summary_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                    daily_kpi_table['Actual Output (parts)'] = daily_summary_df['Actual Output (parts)'].map('{:,.2f}'.format)
                    
                    # Target Output
                    daily_kpi_table['Target Output (parts)'] = daily_summary_df['Target Output (parts)'].map('{:,.2f}'.format)

                    # Capacity Loss (Dynamic based on benchmark)
                    if benchmark_view == "Optimal Output":
                        daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                        daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                    else: # Target Output
                        daily_kpi_table['Capacity Loss (vs Target) (Time)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (d/h/m)']} ({r['Capacity Loss (vs Target) (time %)']:.1%})", axis=1)
                        daily_kpi_table['Capacity Loss (vs Target) (parts)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (parts)']:,.2f} ({r['Capacity Loss (vs Target) (parts %)']:.1%})", axis=1)

                    st.dataframe(daily_kpi_table, use_container_width=True)

                st.markdown("---")
                
                # --- [NEW] Production Output Overview Chart (ALWAYS vs Optimal, but 4th bar is dynamic) ---
                st.header("Production Output Overview (parts)")
                
                # Dynamic labels for the final bar
                if benchmark_view == "Optimal Output":
                    final_bar_value = total_optimal
                    final_bar_label = "Optimal Output (parts)"
                    final_bar_text = "Optimal"
                else:
                    final_bar_value = total_target
                    final_bar_label = "Target Output (parts)"
                    final_bar_text = "Target"

                # Dynamic calculation for loss bars when Target is exceeded
                temp_net_cycle_loss = total_net_cycle_loss
                temp_downtime_loss = total_downtime_loss
                
                if benchmark_view == 'Target Output' and total_produced >= total_target:
                    # If Target is met or exceeded, capacity loss vs target should be zero
                    temp_net_cycle_loss = 0
                    temp_downtime_loss = 0
                
                # Ensure the stacked portion is non-negative (for visual stacking only)
                total_positive_net_cycle_loss_chart = np.maximum(0, temp_net_cycle_loss)

                categories = [
                    'Actual Output (parts)', 
                    'Capacity Loss (slow/fast cycle time)', 
                    'Capacity Loss (downtime)', 
                    final_bar_label
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
                    x=[categories[1]], y=[total_positive_net_cycle_loss_chart], name='Capacity Loss (slow/fast cycle time)',
                    marker_color='gold', 
                    text=[f"{total_positive_net_cycle_loss_chart:,.0f}<br>Parts Lost"] if total_positive_net_cycle_loss_chart > 0 else None,
                    textposition='auto', 
                    hovertemplate='<b>Capacity Loss (slow/fast cycle time)</b><br>Slow Loss: %{customdata[0]:,.0f}<br>Fast Gain: -%{customdata[1]:,.0f}<br><b>Net Loss: %{y:,.0f}</b><extra></extra>',
                    customdata=np.array([[total_slow_loss, total_fast_gain]])
                ))
                
                # --- Bar 3: Downtime Loss (Stacked) ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[total_produced], name='Base (Produced)',
                    marker_color='rgba(0, 128, 0, 0.2)', showlegend=False, hoverinfo='none'
                ))
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[total_positive_net_cycle_loss_chart], name='Base (Cycle Loss)',
                    marker_color='rgba(255, 215, 0, 0.2)', showlegend=False, hoverinfo='none'
                ))
                fig_summary.add_trace(go.Bar(
                    x=[categories[2]], y=[temp_downtime_loss], name='Capacity Loss (downtime)',
                    marker_color='red', 
                    text=[f"{temp_downtime_loss:,.0f}<br>Parts Lost"] if temp_downtime_loss > 0 else None,
                    textposition='auto', hovertemplate='<b>Capacity Loss (downtime)</b><br>Parts: %{y:,.0f}<extra></extra>'
                ))
                
                # --- Bar 4: Benchmark Output ---
                fig_summary.add_trace(go.Bar(
                    x=[categories[3]], y=[final_bar_value], name=final_bar_label,
                    marker_color='grey', text=[f"{final_bar_value:,.0f}<br>{final_bar_text}"],
                    textposition='auto', hovertemplate=f'<b>{final_bar_label}</b><br>Parts: %{{y:,.0f}}<extra></extra>'
                ))
                
                # --- Horizontal Lines (Optimal & Target) ---
                fig_summary.add_hline(y=total_optimal, line_dash="dot", line_color="purple", name='Optimal Output (parts)')
                fig_summary.add_hline(y=total_target, line_dash="dash", line_color="blue", name='Target Output (parts)')

                # --- Layout Updates ---
                fig_summary.update_layout(
                    barmode='stack',
                    title='Production Output Overview (All Time)',
                    yaxis_title='Parts',
                    legend_title='Metric',
                    xaxis=dict(
                        categoryorder='array', 
                        categoryarray=categories,
                        showgrid=False
                    )
                )
                
                st.plotly_chart(fig_summary, use_container_width=True)
                
                # --- 1. AGGREGATED REPORT (Chart & Table) ---
                st.markdown("---")
                st.header(f"{data_frequency} Performance Breakdown")

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
                display_df['Total Capacity Loss (parts %)'] = np.where(
                    display_df['Optimal Output (parts)'] > 0, 
                    display_df['Total Capacity Loss (parts)'] / display_df['Optimal Output (parts)'], 0
                )
                display_df['Capacity Loss (vs Target) (parts %)'] = np.where(
                    display_df['Target Output (parts)'] > 0,
                    display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0
                )
                
                _target_output_perc_array = np.full(len(display_df), target_output_perc / 100.0)
                
                # --- Pre-calculate all d/h/m columns ---
                display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                
                chart_df = display_df.reset_index()

                # --- Performance Breakdown Chart (Time-Series) ---
                fig = go.Figure()
                
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
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual Output: %{y:,.0f} (%{customdata[0]:.1%})<extra></extra>'
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
                else: # Target View (Single Bar for Loss vs Target, which can go negative)
                    fig.add_trace(go.Bar(
                        x=chart_df['Date'],
                        y=chart_df['Capacity Loss (vs Target) (parts)'],
                        name='Capacity Loss (vs Target)',
                        marker_color='blue', # Default color
                        customdata=np.stack((
                            chart_df['Capacity Loss (vs Target) (parts %)'],
                        ), axis=-1),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Capacity Loss (vs Target): %{y:,.0f} (%{customdata[0]:.1%})<extra></extra>'
                    ))
                    # Color the bars red or blue based on value
                    fig.update_traces(
                        selector=dict(name='Capacity Loss (vs Target)'),
                        marker_color=['red' if v > 0 else 'blue' for v in chart_df['Capacity Loss (vs Target) (parts)']] # Red if loss > 0, Blue if loss <= 0
                    )
                    fig.update_layout(barmode='relative') # Show positive/negative

                # --- OVERLAY LINES (ON PRIMARY Y-AXIS) ---
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Target Output (parts)'],
                    name='Target Output (parts)', 
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    customdata=_target_output_perc_array,
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Target Output: %{y:,.0f} (%{customdata:.0%})<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Optimal Output (parts)'],
                    name='Optimal Output (parts)', 
                    mode='lines',
                    line=dict(color='purple', dash='dot'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Optimal Output: %{y:,.0f}<extra></extra>'
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
                
                # --- Full Data Table (Production Totals Report) ---
                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df.index)
                
                report_table_1['Total Shots (all)'] = display_df['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Valid Shots (non 999.9)'] = display_df.apply(lambda r: f"{r['Valid Shots (non 999.9)']:,.0f}", axis=1)
                report_table_1['Invalid Shots (999.9 removed)'] = display_df['Invalid Shots (999.9 removed)'].map('{:,.0f}'.format)
                report_table_1['Filtered Run Time'] = display_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Cycle Time Total'] = display_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                
                st.dataframe(report_table_1, use_container_width=True)

                # --- Create Table 2 (Capacity Loss Report) ---
                table_2_title = "Capacity Loss & Gain Report (vs Optimal)" if benchmark_view == "Optimal Output" else "Capacity Loss (vs Target) Report"
                st.header(f"{table_2_title} ({data_frequency})")
                report_table_2 = pd.DataFrame(index=display_df.index)
                
                report_table_2['Optimal Output (parts)'] = display_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                report_table_2['Target Output (parts)'] = display_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f} ({target_output_perc / 100.0:.0%})", axis=1)
                report_table_2['Actual Output (parts)'] = display_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)

                if benchmark_view == "Optimal Output":
                    # Re-calculate individual percentages on the aggregated level (only for display)
                    downtime_parts_perc = display_df['Capacity Loss (downtime) (parts)'] / display_df['Optimal Output (parts)']
                    slow_parts_perc = display_df['Capacity Loss (slow cycle time) (parts)'] / display_df['Optimal Output (parts)']
                    fast_parts_perc = display_df['Capacity Gain (fast cycle time) (parts)'] / display_df['Optimal Output (parts)']
                    
                    report_table_2['Capacity Loss (downtime)'] = display_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f}", axis=1) + ' (' + (downtime_parts_perc * 100).map('{:.1f}%').fillna('N/A') + ')'
                    report_table_2['Capacity Loss (slow cycles)'] = display_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f}", axis=1) + ' (' + (slow_parts_perc * 100).map('{:.1f}%').fillna('N/A') + ')'
                    report_table_2['Capacity Gain (fast cycles)'] = display_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f}", axis=1) + ' (' + (fast_parts_perc * 100).map('{:.1f}%').fillna('N/A') + ')'
                    report_table_2['Total Capacity Loss (Net)'] = display_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f}", axis=1) + ' (' + (display_df['Total Capacity Loss (parts)'] / display_df['Optimal Output (parts)'] * 100).map('{:.1f}%').fillna('N/A') + ')'
                else: # Target View
                    report_table_2['Capacity Loss (vs Target) (parts)'] = display_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (parts)']:,.2f} ({r['Capacity Loss (vs Target) (parts %)']:.1%})", axis=1)

                st.dataframe(report_table_2, use_container_width=True)

                # --- 2. SHOT-BY-SHOT ANALYSIS ---
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
                    
                    # --- Y-Axis Zoom Slider ---
                    st.subheader("Chart Controls")
                    y_axis_max = st.slider(
                        "Zoom Y-Axis (sec)",
                        min_value=10,
                        max_value=1000, 
                        value=50, 
                        step=10,
                        help="Adjust the max Y-axis to zoom in on the cluster. (Set to 1000 to see all outliers)."
                    )
                    
                    # Filter the shot data to the selected day
                    df_day_shots = all_shots_df[all_shots_df['date'] == selected_date].copy()
                    
                    if df_day_shots.empty:
                        st.warning(f"No valid shots found for {selected_date}.")
                    else:
                        # Get the Approved CT for this specific day
                        approved_ct_for_day = df_day_shots['Approved CT'].iloc[0]
                        
                        # --- Create the Plotly Scatter Chart ---
                        fig_ct = go.Figure()

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

                        # Add the green Approved CT line (as requested)
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
                            yaxis_range=[0, y_axis_max], 
                            barmode='overlay' 
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