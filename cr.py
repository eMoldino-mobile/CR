import streamlit as st
import pandas as pd
import numpy as np

def load_data(uploaded_file):
    """Loads data from the uploaded file (Excel or CSV) into a DataFrame."""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
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
    Main function to process the raw DataFrame and calculate all 16 Capacity Risk fields.
    """
    
    # --- 1. Standardize and Prepare Data ---
    
    # Create a working copy
    df = df_raw.copy()

    # Define standard column names
    col_map = {
        'SHOT TIME': 'SHOT TIME',
        'APPROVED CT': 'Approved CT',
        'ACTUAL CT': 'Actual CT',
        'Working Cavities': 'Working Cavities',
        'Plant Area': 'Plant Area'
    }
    
    # Rename columns to standard names (case-insensitive find)
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
    
    # Handle 'Working Cavities' (as defined in our logic)
    if 'Working Cavities' not in df.columns:
        st.info(f"'Working Cavities' column not found. Using default value: {default_cavities}")
        df['Working Cavities'] = default_cavities
    else:
        # Fill any missing cavity data with the default (1)
        df['Working Cavities'].fillna(1, inplace=True)

    # Handle 'Plant Area' (as defined in our logic)
    if 'Plant Area' not in df.columns:
        if toggle_filter:
            st.warning("'Plant Area' column not found. Cannot apply Maintenance/Warehouse filter.")
            toggle_filter = False # Disable the toggle if column is missing
        df['Plant Area'] = 'Production' # Assign a default value

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

    # Filter for production shots based on the toggle
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
        # Allow calculations to proceed, but many will be 0.
    
    # --- 5. Get Configuration Values ---
    
    # Get the single, consistent Approved CT
    # We take the mode() in case there are multiple, but user said it's consistent.
    try:
        APPROVED_CT = df_valid['Approved CT'].mode().iloc[0]
    except IndexError:
        # Fallback if df_valid is empty
        APPROVED_CT = df_production_only['Approved CT'].mode().iloc[0]

    SLOW_TOLERANCE = slow_tol_perc / 100.0
    FAST_TOLERANCE = fast_tol_perc / 100.0
    TARGET_OEE = target_oee_perc / 100.0

    # --- 6. Calculate All 16 Fields ---
    
    results = {}
    
    # A. Basic Counts
    results['Total Shots (all)'] = len(df_production_only)
    results['VALID SHOTS'] = len(df_valid)
    results['Invalid Shots (999.9 sec)'] = results['Total Shots (all)'] - results['VALID SHOTS']
    
    # B. Time Calculations
    results['Total Run Time (sec)'] = (df_production_only['SHOT TIME'].max() - df_production_only['SHOT TIME'].min()).total_seconds()
    
    if not df_valid.empty:
        results['Mode CT'] = df_valid['Actual CT'].mode().iloc[0]
        results['Actual Cycle Time Total (sec)'] = df_valid['Actual CT'].sum()
    else:
        results['Mode CT'] = 0
        results['Actual Cycle Time Total (sec)'] = 0
    
    # *** NEW CONFIRMED LOGIC ***
    results['Downtime (sec)'] = results['Total Run Time (sec)'] - results['Actual Cycle Time Total (sec)']
    
    # C. Output Calculations
    results['Actual Output'] = df_valid['Working Cavities'].sum()
    
    if not df_valid.empty:
        max_cavities = df_valid['Working Cavities'].max()
    else:
        max_cavities = default_cavities # Fallback
        
    results['Optimal Output'] = (results['Total Run Time (sec)'] / APPROVED_CT) * max_cavities
    
    # D. Loss & Gap Calculations
    results['Availability Loss'] = results['Downtime (sec)'] / APPROVED_CT
    results['Gap'] = results['Optimal Output'] - results['Actual Output']
    
    # Slow Cycle Loss
    slow_limit = APPROVED_CT * (1 + SLOW_TOLERANCE)
    df_slow = df_valid[df_valid['Actual CT'] > slow_limit]
    if not df_slow.empty:
        slow_loss_series = ((df_slow['Actual CT'] - APPROVED_CT) / APPROVED_CT) * df_slow['Working Cavities']
        results['Slow Cycle Loss'] = slow_loss_series.sum()
    else:
        results['Slow Cycle Loss'] = 0
        
    # Efficiency Gain
    fast_limit = APPROVED_CT * (1 - FAST_TOLERANCE)
    df_fast = df_valid[df_valid['Actual CT'] < fast_limit]
    if not df_fast.empty:
        gain_series = ((APPROVED_CT - df_fast['Actual CT']) / APPROVED_CT) * df_fast['Working Cavities']
        results['Efficiency Gain'] = gain_series.sum()
    else:
        results['Efficiency Gain'] = 0

    # Good Shots
    df_good = df_valid[(df_valid['Actual CT'] >= fast_limit) & (df_valid['Actual CT'] <= slow_limit)]
    results['Good Shots'] = len(df_good)
    
    # E. Target Calculations
    results['Target OEE'] = target_oee_perc # Store as percentage
    results['Target Output'] = results['Optimal Output'] * TARGET_OEE
    
    # --- 7. Format and Return Results ---
    
    # Convert dictionary to a Series for a clean data table
    final_results = pd.Series(results, name="Value")
    final_results.index.name = "Metric"
    
    return final_results

# ==================================================================
#                       STREAMLIT APP LAYOUT
# ==================================================================

# --- Page Config ---
st.set_page_config(
    page_title="Capacity Risk Calculator",
    layout="wide"
)

st.title("Capacity Risk Calculator (Phase 1)")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Raw Data File (CSV or Excel)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")

toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots", 
    value=False,
    help="If ON, all calculations will exclude shots where 'Plant Area' is 'Maintenance' or 'Warehouse'."
)

default_cavities = st.sidebar.number_input(
    "Default Working Cavities", 
    min_value=1, 
    value=2,
    help="This value will be used if the 'Working Cavities' column is not found in the file."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Tolerance & Targets")

slow_tol_perc = st.sidebar.slider(
    "Slow Tolerance %", 
    min_value=0.0, 
    max_value=100.0, 
    value=5.0, 
    step=0.5,
    format="%.1f%%"
)

fast_tol_perc = st.sidebar.slider(
    "Fast Tolerance %", 
    min_value=0.0, 
    max_value=100.0, 
    value=10.0, 
    step=0.5,
    format="%.1f%%"
)

target_oee_perc = st.sidebar.slider(
    "Target OEE", 
    min_value=0.0, 
    max_value=100.0, 
    value=100.0, 
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
            try:
                results_series = calculate_capacity_risk(
                    df_raw, 
                    toggle_filter, 
                    default_cavities, 
                    slow_tol_perc, 
                    fast_tol_perc, 
                    target_oee_perc
                )
                
                if results_series is not None:
                    # --- Display Results Table ---
                    st.header("Capacity Risk Data Table")
                    
                    # Format the Series into a nice DataFrame for display
                    results_df = results_series.to_frame(name="Value")
                    
                    # Apply number formatting
                    st.dataframe(
                        results_df.style.format("{:,.2f}", na_rep="N/A"),
                        use_container_width=True
                    )
                    
                    # --- Display Raw Data (optional) ---
                    with st.expander("Show Loaded & Standardized Raw Data"):
                        # Re-load for a clean view, or show the processed df
                        # For simplicity, we'll just show the head() of the raw file
                        st.dataframe(df_raw.head(100))

            except Exception as e:
                st.error(f"An unexpected error occurred during calculation: {e}")

else:
    st.info("Please upload a data file to begin.")