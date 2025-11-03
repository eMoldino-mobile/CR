import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

def calculate_capacity_risk(df_raw, toggle_filter, default_cavities, slow_tol_perc, fast_tol_perc, target_oee_perc):
    """
    Main function to process the raw DataFrame and calculate all Capacity Risk fields
    using the final "Performance vs. Quality" logic (Test 5).
    """
    
    # --- 1. Standardize and Prepare Data ---
    df = df_raw.copy()
    # Define standard column names we expect
    col_map = {
        'SHOT TIME': 'SHOT TIME', 'APPROVED CT': 'Approved CT', 'ACTUAL CT': 'Actual CT',
        'Working Cavities': 'Working Cavities', 'Plant Area': 'Plant Area'
    }
    
    # Find and rename columns, ignoring case/spacing
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
        # Default missing cavity data to 1
        df['Working Cavities'].fillna(1, inplace=True)

    if 'Plant Area' not in df.columns:
        if toggle_filter:
            st.warning("'Plant Area' column not found. Cannot apply Maintenance/Warehouse filter.")
            toggle_filter = False # Disable the toggle if column is missing
        df['Plant Area'] = 'Production' # Assign a default value
    else:
        df['Plant Area'].fillna('Production', inplace=True)

    # Convert data types
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

    # Create the final 'valid' dataframe (for cycle calcs)
    df_valid = df_production_only[df_production_only['Actual CT'] < 999.9].copy()

    if df_valid.empty:
        st.warning("Warning: No valid shots found (all shots >= 999.9 sec).")
        # Create empty results to avoid crashing
        return pd.Series(dtype=float), pd.DataFrame()

    # --- 5. Get Configuration Values ---
    try:
        # Get the single, consistent Approved CT
        APPROVED_CT = df_valid['Approved CT'].mode().iloc[0]
    except IndexError:
        # Fallback if df_valid is empty
        APPROVED_CT = df_production_only['Approved CT'].mode().iloc[0]

    # --- 5a. Define Quality Boundaries (for 'Good Shots' count) ---
    quality_slow_limit = APPROVED_CT * (1 + (slow_tol_perc / 100.0))
    quality_fast_limit = APPROVED_CT * (1 - (fast_tol_perc / 100.0))
    
    # --- 5b. Define Performance Boundaries (for Gain/Loss calcs) ---
    # Performance is benchmarked *only* against the Approved CT
    PERFORMANCE_BENCHMARK = APPROVED_CT

    # --- 6. Calculate Per-Shot Metrics ---
    # We calculate these *before* aggregating for the hourly chart
    
    # Performance Calcs (relative to APPROVED_CT)
    df_valid['parts_gain'] = np.where(
        df_valid['Actual CT'] < PERFORMANCE_BENCHMARK,
        ((PERFORMANCE_BENCHMARK - df_valid['Actual CT']) / PERFORMANCE_BENCHMARK) * df_valid['Working Cavities'],
        0
    )
    df_valid['parts_loss'] = np.where(
        df_valid['Actual CT'] > PERFORMANCE_BENCHMARK,
        ((df_valid['Actual CT'] - PERFORMANCE_BENCHMARK) / PERFORMANCE_BENCHMARK) * df_valid['Working Cavities'],
        0
    )
    
    # Quality Calc (relative to tolerance band)
    df_valid['is_good'] = np.where(
        (df_valid['Actual CT'] >= quality_fast_limit) & (df_valid['Actual CT'] <= quality_slow_limit),
        1,
        0
    )
    
    # Output Calc
    df_valid['actual_output'] = df_valid['Working Cavities']

    # --- 7. Calculate Daily Aggregates (for Table & Metrics) ---
    results = {}
    
    # A. Basic Counts
    results['Total Shots (all)'] = len(df_production_only)
    results['VALID SHOTS'] = len(df_valid)
    results['Invalid Shots (999.9 sec)'] = results['Total Shots (all)'] - results['VALID SHOTS']

    # B. Time Calculations
    results['Total Run Time (sec)'] = (df_production_only['SHOT TIME'].max() - df_production_only['SHOT TIME'].min()).total_seconds()
    results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
    results['Downtime (sec)'] = results['Total Run Time (sec)'] - results['Actual Cycle Time Total (sec)']

    # C. Output Calculations
    results['Actual Output'] = df_valid['actual_output'].sum()
    max_cavities = df_production_only['Working Cavities'].max()
    results['Optimal Output'] = (results['Total Run Time (sec)'] / APPROVED_CT) * max_cavities

    # D. Loss & Gap Calculations (from our new logic)
    results['Availability Loss'] = results['Downtime (sec)'] / APPROVED_CT
    results['Slow Cycle Loss'] = df_valid['parts_loss'].sum()
    results['Efficiency Gain'] = df_valid['parts_gain'].sum()
    results['Gap'] = results['Optimal Output'] - results['Actual Output']

    # E. Quality
    results['Good Shots'] = df_valid['is_good'].sum()

    # F. Target
    results['Target OEE'] = target_oee_perc
    results['Target Output'] = results['Optimal Output'] * (target_oee_perc / 100.0)

    # G. OEE Percentage Metrics
    # Avoid division by zero if run time is 0
    if results['Total Run Time (sec)'] > 0:
        # Availability = Run Time / Total Time
        # Per our logic: Run Time = Actual Cycle Time Total
        # Total Time = Total Run Time
        results['Availability %'] = results['Actual Cycle Time Total (sec)'] / results['Total Run Time (sec)']
    else:
        results['Availability %'] = 0
        
    if results['Actual Cycle Time Total (sec)'] > 0:
        # Performance = (Total Parts * Ideal CT) / Run Time
        # Per our logic: Total Parts = VALID SHOTS
        # Ideal CT = APPROVED_CT
        # Run Time = Actual Cycle Time Total
        results['Performance %'] = (APPROVED_CT * results['VALID SHOTS']) / results['Actual Cycle Time Total (sec)']
    else:
        results['Performance %'] = 0

    if results['VALID SHOTS'] > 0:
        # Quality = Good Parts / Total Parts
        results['Quality %'] = results['Good Shots'] / results['VALID SHOTS']
    else:
        results['Quality %'] = 0
        
    results['OEE %'] = results['Availability %'] * results['Performance %'] * results['Quality %']

    # --- 8. Calculate Hourly Aggregates (for Chart) ---
    df_hourly = pd.DataFrame()
    df_valid_indexed = df_valid.set_index('SHOT TIME')
    df_prod_indexed = df_production_only.set_index('SHOT TIME')

    # Sum up hourly components
    df_hourly['Actual Output'] = df_valid_indexed['actual_output'].resample('H').sum()
    df_hourly['Slow Cycle Loss'] = df_valid_indexed['parts_loss'].resample('H').sum()
    df_hourly['Efficiency Gain'] = df_valid_indexed['parts_gain'].resample('H').sum()
    
    # Calculate hourly runtime (max-min)
    hourly_runtime = df_prod_indexed['Actual CT'].resample('H').apply(
        lambda x: (x.index.max() - x.index.min()).total_seconds() if not x.empty else 0
    )
    # Add back the last cycle time for a more accurate hourly "run"
    hourly_last_ct = df_prod_indexed['Actual CT'].resample('H').last().fillna(0)
    hourly_runtime = hourly_runtime + hourly_last_ct
    
    df_hourly['Hourly Optimal Output'] = (hourly_runtime / APPROVED_CT) * max_cavities

    # Reset index to make 'Hour' a column for Altair
    df_hourly = df_hourly.reset_index().rename(columns={'SHOT TIME': 'Hour'})
    
    # --- 9. Format and Return Results ---
    final_results = pd.Series(results)
    
    return final_results, df_hourly

# ==================================================================
#                       STREAMLIT APP LAYOUT
# ==================================================================

# --- Page Config ---
st.set_page_config(
    page_title="Capacity Risk Calculator",
    layout="wide"
)

st.title("Capacity Risk Dashboard (Phase 1)")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Raw Data File (CSV or Excel)", type=["csv", "xlsx", "xls"])

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
st.sidebar.subheader("Quality Tolerance & Targets")

slow_tol_perc = st.sidebar.slider(
    "Slow Tolerance % (for Quality)", 
    min_value=0.0, max_value=100.0, 
    value=5.0, # New Default
    step=0.5,
    format="%.1f%%",
    help="Defines the *upper* band for a 'Good Shot' (e.g., 5% over ACT)."
)

fast_tol_perc = st.sidebar.slider(
    "Fast Tolerance % (for Quality)", 
    min_value=0.0, max_value=100.0, 
    value=5.0, # New Default
    step=0.5,
    format="%.1f%%",
    help="Defines the *lower* band for a 'Good Shot' (e.g., 5% under ACT)."
)

target_oee_perc = st.sidebar.slider(
    "Target OEE", 
    min_value=0.0, max_value=100.0, 
    value=85.0, # More realistic default
    step=1.0,
    format="%.0f%%"
)


# --- Main Page Display ---
if uploaded_file is not None:
    
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")
        
        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk..."):
            
            results_series, df_hourly = calculate_capacity_risk(
                df_raw, 
                toggle_filter, 
                default_cavities, 
                slow_tol_perc, 
                fast_tol_perc, 
                target_oee_perc
            )
            
            if results_series is not None and not results_series.empty:
                
                # --- 1. Metrics Dashboard ---
                st.header("Daily Metrics Dashboard")
                
                # --- Top Level Metrics ---
                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Actual Output (Parts)", 
                    f"{results_series.get('Actual Output', 0):,.0f}"
                )
                col2.metric(
                    "Target Output (Parts)", 
                    f"{results_series.get('Target Output', 0):,.0f}",
                    f"{results_series.get('Target OEE', 0):.0f}% of Optimal"
                )
                col3.metric(
                    "Gap to Optimal (Parts)", 
                    f"{results_series.get('Gap', 0):,.0f}"
                )
                
                st.divider()

                # --- OEE Metrics ---
                colA, colB, colC, colD = st.columns(4)
                colA.metric(
                    "Availability", 
                    f"{results_series.get('Availability %', 0):.1%}"
                )
                colB.metric(
                    "Performance", 
                    f"{results_series.get('Performance %', 0):.1%}"
                )
                colC.metric(
                    "Quality (Good Shots)", 
                    f"{results_series.get('Quality %', 0):.1%}"
                )
                colD.metric(
                    "OEE", 
                    f"{results_series.get('OEE %', 0):.1%}",
                    f"{results_series.get('OEE %', 0) - (target_oee_perc/100.0):.1%}",
                    delta_color="inverse"
                )

                # --- 2. Hourly Chart ---
                st.header("Hourly Performance Breakdown")

                # Melt the dataframe for stacking
                df_melted = df_hourly.melt(
                    id_vars=['Hour', 'Hourly Optimal Output', 'Efficiency Gain'],
                    value_vars=['Actual Output', 'Slow Cycle Loss'],
                    var_name='Metric',
                    value_name='Value'
                )

                # Base chart for stacking
                base = alt.Chart(df_melted).encode(
                    x=alt.X('Hour:T', axis=alt.Axis(title='Hour of Day', format="%H:00")),
                    y=alt.Y('Value:Q', axis=alt.Axis(title='Parts')),
                    color=alt.Color('Metric:N', scale={'domain': ['Actual Output', 'Slow Cycle Loss'],
                                                      'range': ['#4e79a7', '#e15759']}), # Blue, Red
                    tooltip=['Hour:T', 'Metric:N', 'Value:Details', alt.Tooltip('Value', format=',.0f')]
                ).transform_calculate(
                    # Create a "Details" field for the tooltip to show gain
                    Value_Details=alt.datum.Value + alt.datum['Efficiency Gain']
                )


                # Stacked bars
                bar_chart = base.mark_bar().properties(
                    title="Hourly Production vs. Loss"
                )
                
                # Line chart for Optimal Output
                line_chart = alt.Chart(df_hourly).mark_line(color='green', strokeDash=[5,5]).encode(
                    x=alt.X('Hour:T'),
                    y=alt.Y('Hourly Optimal Output:Q', axis=alt.Axis(title='Hourly Optimal Output', titleColor='green')),
                    tooltip=[alt.Tooltip('Hour:T'), alt.Tooltip('Hourly Optimal Output:Q', title='Hourly Optimal', format=',.0f')]
                ).properties(
                    title="Hourly Optimal Output (Right Axis)"
                )

                # Combine the charts with independent Y-axes
                final_chart = alt.layer(bar_chart, line_chart).resolve_scale(
                    y='independent'
                ).interactive()
                
                st.altair_chart(final_chart, use_container_width=True)

                # --- 3. Full Data Table (Open by Default) ---
                st.header("Full Daily Report")
                
                # Format the Series into a nice DataFrame for display
                results_df = results_series.to_frame(name="Value")
                
                # --- NEW, CORRECTED CODE ---

                # 1. Define the metrics that should be percentages
                percent_metrics = [
                    'Availability %', 
                    'Performance %', 
                    'Quality %', 
                    'OEE %'
                ]

                # 2. Apply formatting using an IndexSlice for precision
                st.dataframe(
                    results_df.style.format("{:,.2f}", na_rep="N/A") # Default: format all as float
                    .format(
                        "{:,.1%}", # Override: format these specific rows as percent
                        subset=pd.IndexSlice[percent_metrics, "Value"]
                    ),
                    use_container_width=True
                )
                # --- END CORRECTED CODE ---

            elif results_series is not None:
                st.warning("No valid data was found after filtering. Cannot display results.")

else:
    st.info("Please upload a data file to begin.")
