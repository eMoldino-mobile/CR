import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
#                       DATA CALCULATION
# ==================================================================

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # Requires openpyxl
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Error: Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def calculate_capacity_risk(df_raw, toggle_filter, default_cavities, target_output_perc):
    """
    Main function to process the raw DataFrame and calculate all Capacity Risk fields
    using the "Gross Loss" model.
    
    This version groups the calculations by day.
    
    RETURNS:
    - final_df (DataFrame): The aggregated daily report.
    - all_shots_df (DataFrame): The raw, filtered, shot-by-shot data with 'Shot Type'.
    """
    
    # --- 1. Standardize and Prepare Data ---
    df = df_raw.copy()
    col_map = {
        'SHOT TIME': 'SHOT TIME', 'APPROVED CT': 'Approved CT', 'ACTUAL CT': 'Actual CT',
        'Working Cavities': 'Working Cavities', 'Plant Area': 'Plant Area'
    }
    rename_dict = {}
    for col in df.columns:
        for standard_name in col_map.values():
            if col.strip().lower() == standard_name.strip().lower():
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
        st.info(f"'Working Cavities' column not found. Using default value: {default_cavities}")
        df['Working Cavities'] = default_cavities
    else:
        df['Working Cavities'].fillna(1, inplace=True)

    if 'Plant Area' not in df.columns:
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
        st.error(f"Error converting data types: {e}")
        return None, None
        
    # --- 4. Apply Filters (The Toggle) ---
    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty:
        st.error("Error: No 'Production' data found after filtering.")
        return None, None

    # --- 5. NEW: Group by Day ---
    # Create a 'date' column for grouping
    df_production_only['date'] = df_production_only['SHOT TIME'].dt.date
    
    daily_results_list = []
    all_valid_shots_list = [] # NEW: To store shot-by-shot data
    
    # Iterate over each day's data
    for date, daily_df in df_production_only.groupby('date'):
        
        results = {}
        results['Date'] = date
        
        # Create the final 'valid' dataframe (for cycle calcs)
        df_valid = daily_df[daily_df['Actual CT'] < 999.9].copy()

        if df_valid.empty or len(daily_df) < 2:
            results['Total Shots (all)'] = len(daily_df); results['VALID SHOTS'] = 0
            daily_results_list.append(results)
            continue

        # --- 6. Get Daily Configuration Values ---
        try:
            APPROVED_CT = df_valid['Approved CT'].mode().iloc[0]
        except IndexError:
            APPROVED_CT = daily_df['Approved CT'].mode().iloc[0]
        
        PERFORMANCE_BENCHMARK = APPROVED_CT
        max_cavities = daily_df['Working Cavities'].max()

        # --- 7. Calculate Per-Shot Metrics (for this day) ---
        
        # --- LOGIC CHANGE V3.4: 'parts_loss' now uses max_cavities to align with other metrics ---
        df_valid['parts_loss'] = np.where(
            df_valid['Actual CT'] > PERFORMANCE_BENCHMARK,
            ((df_valid['Actual CT'] - PERFORMANCE_BENCHMARK) / PERFORMANCE_BENCHMARK) * max_cavities, # CHANGED
            0
        )
        df_valid['actual_output'] = df_valid['Working Cavities']

        # --- NEW: Add 'Shot Type' for color-coding ---
        conditions = [
            (df_valid['Actual CT'] > PERFORMANCE_BENCHMARK),
            (df_valid['Actual CT'] < PERFORMANCE_BENCHMARK),
            (df_valid['Actual CT'] == PERFORMANCE_BENCHMARK)
        ]
        choices = ['Slow', 'Fast', 'On Target']
        df_valid['Shot Type'] = np.select(conditions, choices, default='N/A')
        
        # Add the approved CT for this day (for the chart line)
        df_valid['Approved CT'] = APPROVED_CT 
        
        all_valid_shots_list.append(df_valid) # Add this day's shots to the master list
        # --- END NEW SHOT LOGIC ---

        # --- 8. Calculate Daily Aggregates ---
        
        # A. Basic Counts
        results['Total Shots (all)'] = len(daily_df)
        results['VALID SHOTS'] = len(df_valid)
        results['Invalid Shots (999.9 sec)'] = results['Total Shots (all)'] - results['VALID SHOTS']

        # B. Time Calculations
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT'].iloc[0]
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        # --- RENAMED ---
        results['Total Run Duration (sec)'] = time_span_sec + last_shot_ct
        
        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
        # --- UPDATED ---
        results['Downtime (sec)'] = results['Total Run Duration (sec)'] - results['Actual Cycle Time Total (sec)']

        # C. Output Calculations
        results['Parts Produced'] = df_valid['actual_output'].sum() # RENAMED
        # --- UPDATED ---
        results['Optimal Output'] = (results['Total Run Duration (sec)'] / APPROVED_CT) * max_cavities

        # D. Loss & Gap Calculations
        results['Availability Loss'] = (results['Downtime (sec)'] / APPROVED_CT) * max_cavities
        results['Slow Cycle Loss'] = df_valid['parts_loss'].sum()
        results['Gap'] = results['Availability Loss'] + results['Slow Cycle Loss']
        
        # F. Target
        results['Target Output'] = results['Optimal Output'] * (target_output_perc / 100.0)
        
        daily_results_list.append(results)

    # --- 9. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')
    
    # --- NEW: Create the master shot-by-shot dataframe ---
    if not all_valid_shots_list:
        return final_df, pd.DataFrame() # Return empty df if no shots
        
    all_shots_df = pd.concat(all_valid_shots_list, ignore_index=True)
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date # ensure 'date' col exists
    
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
st.sidebar.subheader("Target")

target_output_perc = st.sidebar.slider(
    "Target Output % (of Optimal)", 
    min_value=0.0, max_value=100.0, 
    value=85.0, 
    step=1.0,
    format="%.0f%%",
    help="Sets the 'Target Output' goal as a percentage of 'Optimal Output'."
)


# --- Main Page Display ---
if uploaded_file is not None:
    
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")
        
        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk..."):
            
            # --- UPDATED: Function now returns two dfs ---
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
                display_df['Parts Produced (%)'] = np.where(
                    display_df['Optimal Output'] > 0, 
                    display_df['Parts Produced'] / display_df['Optimal Output'], 0
                )
                display_df['VALID SHOTS (%)'] = np.where(
                    display_df['Total Shots (all)'] > 0, 
                    display_df['VALID SHOTS'] / display_df['Total Shots (all)'], 0
                )
                # --- UPDATED ---
                display_df['Actual Cycle Time Total (%)'] = np.where(
                    display_df['Total Run Duration (sec)'] > 0, 
                    display_df['Actual Cycle Time Total (sec)'] / display_df['Total Run Duration (sec)'], 0
                )
                display_df['Downtime (%)'] = np.where(
                    display_df['Total Run Duration (sec)'] > 0, 
                    display_df['Downtime (sec)'] / display_df['Total Run Duration (sec)'], 0
                )
                display_df['Availability Loss (%)'] = np.where(
                    display_df['Optimal Output'] > 0, 
                    display_df['Availability Loss'] / display_df['Optimal Output'], 0
                )
                display_df['Slow Cycle Loss (%)'] = np.where(
                    display_df['Optimal Output'] > 0, 
                    display_df['Slow Cycle Loss'] / display_df['Optimal Output'], 0
                )
                display_df['Gap (%)'] = np.where(
                    display_df['Optimal Output'] > 0, 
                    display_df['Gap'] / display_df['Optimal Output'], 0
                )
                display_df['Target Output (%)'] = target_output_perc / 100.0

                chart_df = display_df.reset_index()

                # --- Performance Breakdown Chart (using Plotly) ---
                st.header(f"{data_frequency} Performance Breakdown")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=chart_df['Date'], y=chart_df['Parts Produced'], name='Parts Produced',
                    marker_color='green',
                    customdata=chart_df['Parts Produced (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Parts Produced: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                fig.add_trace(go.Bar(
                    x=chart_df['Date'], y=chart_df['Availability Loss'], name='Availability Loss',
                    marker_color='red',
                    customdata=chart_df['Availability Loss (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Availability Loss: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                fig.add_trace(go.Bar(
                    x=chart_df['Date'], y=chart_df['Slow Cycle Loss'], name='Slow Cycle Loss',
                    marker_color='gold',
                    customdata=chart_df['Slow Cycle Loss (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Slow Cycle Loss: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'], y=chart_df['Target Output'], name='Target Output', 
                    mode='lines', line=dict(color='blue', dash='dash'),
                    customdata=chart_df['Target Output (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Target Output: %{y:,.0f} (%{customdata:.0%})<extra></extra>'
                ))
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'], y=chart_df['Optimal Output'], name='Optimal Output (100%)', 
                    mode='lines', line=dict(color='purple', dash='dot'), 
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Optimal Output: %{y:,.0f}<extra></extra>'
                ))
                fig.update_layout(
                    barmode='stack', title=chart_title, xaxis_title=xaxis_title,
                    yaxis_title='Parts (Output & Loss)', legend_title='Metric',
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Full Data Table (Open by Default) ---
                st.header(f"Full {data_frequency} Report")
                column_order = [
                    'Total Shots (all)', 
                    'VALID SHOTS', 'VALID SHOTS (%)', 
                    'Invalid Shots (999.9 sec)',
                    'Total Run Duration (sec)', # RENAMED
                    'Actual Cycle Time Total (sec)', 'Actual Cycle Time Total (%)', 
                    'Downtime (sec)', 'Downtime (%)',
                    'Parts Produced', 'Parts Produced (%)', 
                    'Availability Loss', 'Availability Loss (%)', 
                    'Slow Cycle Loss', 'Slow Cycle Loss (%)', 
                    'Gap', 'Gap (%)',
                    'Optimal Output', 
                    'Target Output', 'Target Output (%)'
                ]
                final_columns = [col for col in column_order if col in display_df.columns]
                display_df_final = display_df[final_columns]
                format_dict = {}
                for col in display_df_final.columns:
                    if "(%)" in col: format_dict[col] = "{:.1%}"
                    else: format_dict[col] = "{:,.2f}"
                st.dataframe(
                    display_df_final.style.format(format_dict, na_rep="N/A"),
                    use_container_width=True
                )
                
                # --- 2. NEW: SHOT-BY-SHOT ANALYSIS ---
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