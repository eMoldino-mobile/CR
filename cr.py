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

def calculate_capacity_risk(df_raw, toggle_filter, default_cavities, target_output_perc):
    """
    Main function to process the raw DataFrame and calculate all Capacity Risk fields
    using the "Gross Loss" model.
    
    This version groups the calculations by day.
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
        return None

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
        return None
        
    # --- 4. Apply Filters (The Toggle) ---
    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty:
        st.error("Error: No 'Production' data found after filtering.")
        return None

    # --- 5. NEW: Group by Day ---
    # Create a 'date' column for grouping
    df_production_only['date'] = df_production_only['SHOT TIME'].dt.date
    
    daily_results_list = []
    
    # Iterate over each day's data
    for date, daily_df in df_production_only.groupby('date'):
        
        results = {}
        results['Date'] = date
        
        # Create the final 'valid' dataframe (for cycle calcs)
        df_valid = daily_df[daily_df['Actual CT'] < 999.9].copy()

        if df_valid.empty:
            # Add a row of zeros or NaNs if a day has no valid shots
            results['Total Shots (all)'] = len(daily_df)
            results['VALID SHOTS'] = 0
            # ... add other keys as 0 or np.nan
            daily_results_list.append(results)
            continue

        # --- 6. Get Daily Configuration Values ---
        try:
            APPROVED_CT = df_valid['Approved CT'].mode().iloc[0]
        except IndexError:
            APPROVED_CT = daily_df['Approved CT'].mode().iloc[0]
        
        PERFORMANCE_BENCHMARK = APPROVED_CT

        # --- 7. Calculate Per-Shot Metrics (for this day) ---
        # REMOVED 'parts_gain' calculation
        
        df_valid['parts_loss'] = np.where(
            df_valid['Actual CT'] > PERFORMANCE_BENCHMARK,
            ((df_valid['Actual CT'] - PERFORMANCE_BENCHMARK) / PERFORMANCE_BENCHMARK) * df_valid['Working Cavities'],
            0
        )
        df_valid['actual_output'] = df_valid['Working Cavities']

        # --- 8. Calculate Daily Aggregates ---
        
        # A. Basic Counts
        results['Total Shots (all)'] = len(daily_df)
        results['VALID SHOTS'] = len(df_valid)
        results['Invalid Shots (999.9 sec)'] = results['Total Shots (all)'] - results['VALID SHOTS']

        # B. Time Calculations
        results['Total Run Time (sec)'] = (daily_df['SHOT TIME'].max() - daily_df['SHOT TIME'].min()).total_seconds()
        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
        results['Downtime (sec)'] = results['Total Run Time (sec)'] - results['Actual Cycle Time Total (sec)']

        # C. Output Calculations
        results['Actual Output'] = df_valid['actual_output'].sum()
        max_cavities = daily_df['Working Cavities'].max()
        results['Optimal Output'] = (results['Total Run Time (sec)'] / APPROVED_CT) * max_cavities

        # D. Loss & Gap Calculations
        # --- LOGIC FIX: Availability Loss must be multiplied by max_cavities ---
        results['Availability Loss'] = (results['Downtime (sec)'] / APPROVED_CT) * max_cavities
        results['Slow Cycle Loss'] = df_valid['parts_loss'].sum()
        # REMOVED 'Efficiency Gain'
        
        # --- LOGIC FIX: Gap is now the sum of the two gross losses ---
        results['Gap'] = results['Availability Loss'] + results['Slow Cycle Loss']
        
        # F. Target
        results['Target Output'] = results['Optimal Output'] * (target_output_perc / 100.0)
        
        daily_results_list.append(results)

    # --- 9. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None

    final_df = pd.DataFrame(daily_results_list)
    # --- IMPORTANT: Convert Date to datetime for resampling ---
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date') # Set Date as the index
    
    # Re-order columns for clarity
    column_order = [
        'Total Shots (all)', 'VALID SHOTS', 'Invalid Shots (999.9 sec)',
        'Total Run Time (sec)', 'Actual Cycle Time Total (sec)', 'Downtime (sec)',
        'Actual Output', 'Availability Loss', 'Slow Cycle Loss', 'Gap',
        'Optimal Output', 'Target Output'
    ]
    # Filter for columns that actually exist (in case one was skipped)
    final_columns = [col for col in column_order if col in final_df.columns]
    final_df = final_df[final_columns]
    
    return final_df

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

# --- NEW: Data Frequency Selector ---
data_frequency = st.sidebar.radio(
    "Select Graph Frequency",
    ['Daily', 'Weekly', 'Monthly'],
    index=0, # Default to Daily
    horizontal=True
)

st.sidebar.markdown("---")

toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots", 
    value=True, # Default to ON, as per our test
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
    value=85.0, # More realistic default
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
            
            # The function now returns a DataFrame
            results_df = calculate_capacity_risk(
                df_raw, 
                toggle_filter, 
                default_cavities, 
                target_output_perc
            )
            
            if results_df is not None and not results_df.empty:
                
                # --- NEW: Aggregate data based on frequency selection ---
                if data_frequency == 'Weekly':
                    # Resample by week (summing all columns)
                    # 'W' stands for Weekly, ending on Sunday
                    agg_df = results_df.resample('W').sum()
                    chart_title = "Weekly Capacity Report"
                    xaxis_title = "Week"
                    display_df = agg_df
                elif data_frequency == 'Monthly':
                    # Resample by month end (summing all columns)
                    agg_df = results_df.resample('ME').sum()
                    chart_title = "Monthly Capacity Report"
                    xaxis_title = "Month"
                    display_df = agg_df
                else: # Daily
                    display_df = results_df # Use the original daily df
                    chart_title = "Daily Capacity Report"
                    xaxis_title = "Date"
                
                chart_df = display_df.reset_index()
                # --- END NEW LOGIC ---

                # --- Performance Breakdown Chart (using Plotly) ---
                st.header(f"{data_frequency} Performance Breakdown")
                
                # Create the figure
                fig = go.Figure()

                # --- NEW "GROSS LOSS" STACKED BAR CHART ---
                fig.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Actual Output'],
                    name='Actual Output',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Availability Loss'],
                    name='Availability Loss',
                    marker_color='red'
                ))
                fig.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Slow Cycle Loss'],
                    name='Slow Cycle Loss',
                    marker_color='gold'
                ))
                                
                # --- EFFICIENCY GAIN REMOVED ---

                # --- OVERLAY LINES (BOTH ON PRIMARY Y-AXIS) ---
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Target Output'], 
                    name='Target Output', 
                    mode='lines',
                    line=dict(color='blue', dash='dash'), 
                ))
                
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Optimal Output'], 
                    name='Optimal Output (100%)', 
                    mode='lines',
                    line=dict(color='darkgrey', dash='dot'), 
                ))
                
                # --- LAYOUT UPDATE ---
                fig.update_layout(
                    barmode='stack', # This is key
                    title=chart_title, 
                    xaxis_title=xaxis_title,
                    yaxis_title='Parts (Output & Loss)',
                    legend_title='Metric',
                    hovermode="x unified"
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # --- Full Data Table (Open by Default) ---
                st.header(f"Full {data_frequency} Report")
                
                # Display the aggregated dataframe
                st.dataframe(
                    display_df.style.format("{:,.2f}"),
                    use_container_width=True
                )

            elif results_df is not None:
                st.warning("No valid data was found after filtering. Cannot display results.")

else:
    st.info("Please upload a data file to begin.")

