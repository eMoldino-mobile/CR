import streamlit as st
import pandas as pd
import numpy as np
# Altair import removed as it's no longer needed

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
        return None

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
        return None
        
    # --- 4. Apply Filters (The Toggle) ---
    if toggle_filter:
        df_production_only = df[~df['Plant Area'].isin(['Maintenance', 'Warehouse'])].copy()
    else:
        df_production_only = df.copy()

    if df_production_only.empty:
        st.error("Error: No 'Production' data found after filtering.")
        return None

    # Create the final 'valid' dataframe (for cycle calcs)
    df_valid = df_production_only[df_production_only['Actual CT'] < 999.9].copy()

    if df_valid.empty:
        st.warning("Warning: No valid shots found (all shots >= 999.9 sec).")
        # Create empty results to avoid crashing
        return pd.Series(dtype=float)

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
    
    # --- 8. Format and Return Results ---
    # Removed the hourly aggregation logic
    final_results = pd.Series(results)
    
    return final_results

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
            
            # Removed df_hourly from the return
            results_series = calculate_capacity_risk(
                df_raw, 
                toggle_filter, 
                default_cavities, 
                slow_tol_perc, 
                fast_tol_perc, 
                target_oee_perc
            )
            
            if results_series is not None and not results_series.empty:
                
                # --- METRICS AND CHART REMOVED ---

                # --- Full Data Table (Open by Default) ---
                st.header("Full Daily Report")
                
                # Format the Series into a nice DataFrame for display
                results_df = results_series.to_frame(name="Value")
                
                # --- ROBUST V1.4 FORMATTING ---

                # 1. Define the metrics that should be percentages
                percent_metrics = [
                    'Availability %', 
                    'Performance %', 
                    'Quality %', 
                    'OEE %'
                ]
                
                # Create a copy to format and ensure it's object type for strings
                formatted_df = results_df.astype(object)

                # 2. Iterate and apply string formatting
                for metric_name in percent_metrics:
                    if metric_name in formatted_df.index:
                        value = formatted_df.loc[metric_name, 'Value']
                        # Check if it's a valid number (not NaN or None)
                        if pd.notna(value) and isinstance(value, (int, float)):
                            formatted_df.loc[metric_name, 'Value'] = f"{value:.1%}"
                        elif pd.notna(value):
                             # if it's not a number (e.g., already a string), just keep it
                             formatted_df.loc[metric_name, 'Value'] = str(value)
                        else:
                            formatted_df.loc[metric_name, 'Value'] = "N/A" # Handle NaN/None

                # 3. Format all other numeric values
                for idx in formatted_df.index:
                    if idx not in percent_metrics:
                        value = formatted_df.loc[idx, 'Value']
                        if pd.notna(value) and isinstance(value, (int, float)):
                            formatted_df.loc[idx, 'Value'] = f"{value:,.2f}"
                        elif pd.notna(value):
                             formatted_df.loc[idx, 'Value'] = str(value)
                        else:
                            formatted_df.loc[idx, 'Value'] = "N/A"

                # 4. Display the string-formatted DataFrame
                # We don't use .style.format() at all now.
                st.dataframe(
                    formatted_df,
                    use_container_width=True
                )
                # --- END NEW FORMATTING ---

            elif results_series is not None:
                st.warning("No valid data was found after filtering. Cannot display results.")

else:
    st.info("Please upload a data file to begin.")

