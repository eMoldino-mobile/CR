import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
#                       HELPER FUNCTIONS
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
#                       DATA CALCULATION
# ==================================================================

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=0)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
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
    Main function to process the raw DataFrame and calculate all Capacity Risk fields
    using the "Net Loss" model.
    """
    
    # --- 1. Standardize and Prepare Data ---
    df = _df_raw.copy()
    col_map = {
        'SHOT TIME': 'SHOT TIME', 'APPROVED CT': 'Approved CT', 'ACTUAL CT': 'Actual CT',
        'Working Cavities': 'Working Cavities', 'Plant Area': 'Plant Area'
    }
    rename_dict = {}
    for col in df.columns:
        for standard_name in col_map.values():
            if str(col).strip().lower() == standard_name.strip().lower():
                rename_dict[col] = standard_name
                
    df.rename(columns=rename_dict, inplace=True)

    # --- 2. Check for Required Columns ---
    required_cols = ['SHOT TIME', 'Approved CT', 'Actual CT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return None, None

    # --- 3. Handle Optional Columns and Data Types ---
    if 'Working Cavities' not in df.columns:
        df['Working Cavities'] = default_cavities
    else:
        df['Working Cavities'].fillna(1, inplace=True)

    if 'Plant Area' not in df.columns:
        if toggle_filter:
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
    
    for date, daily_df in df_production_only.groupby('date'):
        
        results = {}
        results['Date'] = date
        
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
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1 

        # --- 7. Calculate Per-Shot Metrics ---
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
        results['Total Shots (all)'] = len(daily_df)
        results['Valid Shots (non 999.9)'] = len(df_valid) 
        results['Invalid Shots (999.9 removed)'] = results['Total Shots (all)'] - results['Valid Shots (non 999.9)']

        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        results['Total Run Duration (sec)'] = time_span_sec + last_shot_ct
        
        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
        
        results['Capacity Loss (downtime) (sec)'] = results['Total Run Duration (sec)'] - results['Actual Cycle Time Total (sec)'] 

        results['Parts Produced (parts)'] = df_valid['actual_output'].sum() 
        results['Optimal Output (parts)'] = (results['Total Run Duration (sec)'] / APPROVED_CT) * max_cavities

        results['Capacity Loss (downtime) (parts)'] = (results['Capacity Loss (downtime) (sec)'] / APPROVED_CT) * max_cavities 
        results['Capacity Loss (slow cycle time) (parts)'] = df_valid['parts_loss'].sum() 
        results['Capacity Gain (fast cycle time) (parts)'] = df_valid['parts_gain'].sum()
        
        # --- Net Loss Calculation ---
        results['Net Performance Loss (parts)'] = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Net Performance Loss (parts)']
        
        results['Target Output (parts)'] = results['Optimal Output (parts)'] * (target_output_perc / 100.0)
        
        # Loss vs Target Metrics (Used for dashboard/metric view only)
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        results['Capacity Loss (vs Target) (parts)'] = results['Target Output (parts)'] - results['Actual Output (parts)']
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * APPROVED_CT) / max_cavities if max_cavities > 0 else 0
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + ((results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']) * APPROVED_CT / max_cavities)
        
        daily_results_list.append(results)

    # --- 9. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list).fillna(0)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')
    
    if not all_valid_shots_list:
        return final_df, pd.DataFrame() 
        
    all_shots_df = pd.concat(all_valid_shots_list, ignore_index=True)
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date 
    
    return final_df, all_shots_df

# ==================================================================
#                       STREAMLIT APP LAYOUT
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
    help="Sets the 'Target Output (parts)' goal as a percentage of 'Optimal Output (parts)'."
)

# --- Benchmark Selector (Used for KPI view) ---
benchmark_view = st.sidebar.radio(
    "Select Metric Comparison",
    ['Optimal Output', 'Target Output'],
    index=0,
    horizontal=False,
    help="Select the benchmark for the KPI boxes: Total Loss vs. Optimal vs. Loss vs. Target."
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
                
                # --- 1. AGGREGATED REPORT (Chart & Table) ---
                
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
                # These are used in the tables
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
                display_df['Capacity Loss (downtime) (time %)'] = np.where(
                    display_df['Filtered Run Time (sec)'] > 0, 
                    display_df['Capacity Loss (downtime) (sec)'] / display_df['Filtered Run Time (sec)'], 0
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
                display_df['Gap to Target (parts %)'] = np.where(
                    display_df['Target Output (parts)'] > 0, 
                    display_df['Gap to Target (parts)'] / display_df['Target Output (parts)'], 0
                )
                display_df['Capacity Loss (vs Target) (parts %)'] = np.where(
                    display_df['Target Output (parts)'] > 0, 
                    display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0
                )
                
                _target_output_perc_array = np.full(len(display_df), target_output_perc / 100.0)
                
                # --- Add human-readable duration columns ---
                display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                display_df['Capacity Loss (downtime) (d/h/m)'] = display_df['Capacity Loss (downtime) (sec)'].apply(format_seconds_to_dhm)
                display_df['Total Capacity Loss (d/h/m)'] = display_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)
                
                chart_df = display_df.reset_index()

                # --- 2. KPI DASHBOARD ---
                
                # --- Calculate Aggregates for Metrics ---
                total_produced = results_df['Actual Output (parts)'].sum()
                total_optimal = results_df['Optimal Output (parts)'].sum()
                total_target = results_df['Target Output (parts)'].sum()
                total_loss_parts = results_df['Total Capacity Loss (parts)'].sum()
                run_time_sec_total = results_df['Filtered Run Time (sec)'].sum()
                run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)

                # Calculate overall percentages for the dashboard metrics
                prod_perc = (total_produced / total_optimal) * 100 if total_optimal > 0 else 0
                loss_perc = (total_loss_parts / total_optimal) * 100 if total_optimal > 0 else 0
                
                # --- KPI Display ---
                with st.container(border=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Run Duration", run_time_dhm_total)
                        st.metric("Total Valid Shots", f"{results_df['Valid Shots (non 999.9)'].sum():,.0f}")
                        
                    with col2:
                        st.metric("Total Optimal Output", f"{total_optimal:,.0f}")
                        st.metric("Total Target Output", f"{total_target:,.0f}")
                        
                    with col3:
                        st.metric("Total Parts Produced", f"{total_produced:,.0f}", f"{prod_perc:.1f}% vs Optimal")
                        st.metric("Total Capacity Loss (Net)", f"{total_loss_parts:,.0f}", f"{loss_perc:.1f}% vs Optimal", delta_color="inverse")

                # --- 3. Capacity Breakdown Chart (Horizontal) ---
                st.header(f"Capacity Breakdown ({data_frequency})")
                
                # We need to calculate the *total* target for the period for the vline
                total_target_for_period = display_df['Target Output (parts)'].sum()
                
                # --- Create the figure ---
                fig = go.Figure()

                # --- STACKED HORIZONTAL BARS ---
                fig.add_trace(go.Bar(
                    y=chart_df['Date'],
                    x=chart_df['Actual Output (parts)'],
                    name='Parts Produced',
                    marker_color='green',
                    orientation='h',
                    customdata=chart_df['Actual Output (%)'],
                    hovertemplate='<b>%{y|%Y-%m-%d}</b><br>Parts Produced: %{x:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    y=chart_df['Date'],
                    x=chart_df['Capacity Loss (downtime) (parts)'],
                    name='Capacity Loss (Downtime)',
                    marker_color='red',
                    orientation='h',
                    customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                    hovertemplate='<b>%{y|%Y-%m-%d}</b><br>Loss (Downtime): %{x:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                fig.add_trace(go.Bar(
                    y=chart_df['Date'],
                    x=chart_df['Net Performance Loss (parts)'],
                    name='Net Performance Loss',
                    marker_color='gold',
                    orientation='h',
                    customdata=np.stack((
                        chart_df['Capacity Loss (slow cycle time) (parts)'],
                        chart_df['Capacity Gain (fast cycle time) (parts)']
                    ), axis=-1),
                    hovertemplate=
                        '<b>%{y|%Y-%m-%d}</b><br>' +
                        '<b>Net Performance Loss: %{x:,.0f}</b><br><br>' +
                        'Loss (Slow Cycles): %{customdata[0]:,.0f}<br>' +
                        'Gain (Fast Cycles): -%{customdata[1]:,.0f}<br>' +
                        '<extra></extra>'
                ))
                
                # --- LAYOUT UPDATE ---
                fig.update_layout(
                    barmode='stack', 
                    title=chart_title, 
                    yaxis_title=xaxis_title, # Y-axis is now Date/Week/Month
                    xaxis_title='Parts (Output & Loss)', # X-axis is now Parts
                    legend_title='Metric',
                    hovermode="closest" # 'closest' is better for h-bars
                )
                
                # --- Add Target Line (VLINE) ---
                # This shows the target for the *entire period* as a single line
                fig.add_vline(
                    x=total_target, 
                    line_width=3, 
                    line_dash="dash", 
                    line_color="blue",
                    annotation_text=f"Total Target ({total_target:,.0f})",
                    annotation_position="top right"
                )
                
                st.plotly_chart(fig, use_container_width=True)

                
                # --- 4. Full Data Table (Open by Default) ---
                
                # --- Create Table 1 (Totals Report) ---
                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df.index)
                
                # --- Map and format the columns ---
                report_table_1['Total Shots (all)'] = display_df['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Valid Shots (non 999.9)'] = display_df.apply(lambda r: f"{r['Valid Shots (non 999.9)']:,.0f} ({r['Valid Shots (non 999.9) (%)']:.1%})", axis=1)
                report_table_1['Invalid Shots (999.9 removed)'] = display_df['Invalid Shots (999.9 removed)'].map('{:,.0f}'.format)
                report_table_1['Total Run Duration'] = display_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Cycle Time Total'] = display_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                
                st.dataframe(report_table_1, use_container_width=True)

                # --- Create Table 2 (Capacity Loss Report) ---
                st.header(f"Capacity Loss & Gain Report ({data_frequency})")
                report_table_2 = pd.DataFrame(index=display_df.index)
                
                # --- Map and format the columns ---
                report_table_2['Optimal Output (parts)'] = display_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                report_table_2['Target Output (parts)'] = display_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f} ({target_output_perc / 100.0:.0%})", axis=1)
                report_table_2['Parts Produced (parts)'] = display_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)

                # --- Loss & Gain Columns ---
                report_table_2['Capacity Loss (downtime)'] = display_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                report_table_2['Capacity Loss (slow cycles)'] = display_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                report_table_2['Capacity Gain (fast cycles)'] = display_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                report_table_2['Total Capacity Loss (Net)'] = display_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                report_table_2['Net Performance Loss'] = display_df['Net Performance Loss (parts)'].map('{:,.2f}'.format)

                st.dataframe(report_table_2, use_container_width=True)

                # --- 5. SHOT-BY-SHOT ANALYSIS ---
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
                    y_axis_max = st.slider(
                        "Zoom Y-Axis (sec)",
                        min_value=10,
                        max_value=1000, # Max to see all outliers
                        value=50, # Default to a "zoomed in" view
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

                        # --- UPDATED: Define NEW colors for each shot type ---
                        color_map = {
                            'Slow': 'red',
                            'Fast': 'gold',
                            'On Target': 'darkblue'
                        }
                        
                        # Add traces for each shot type
                        for shot_type, color in color_map.items():
                            df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                            if not df_subset.empty:
                                # --- UPDATED: Changed to go.Bar ---
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