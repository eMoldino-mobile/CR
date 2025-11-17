import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
from dateutil.relativedelta import relativedelta

# ==================================================================
#                            HELPER FUNCTIONS
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
    return " ".join(parts) if parts else "0m"

# Define all result columns globally
ALL_RESULT_COLUMNS = [
    'Date', 'Filtered Run Time (sec)', 'Optimal Output (parts)',
    'Capacity Loss (downtime) (sec)',
    'Capacity Loss (downtime) (parts)',
    'Actual Output (parts)', 'Actual Cycle Time Total (sec)',
    'Capacity Gain (fast cycle time) (sec)', 'Capacity Loss (slow cycle time) (sec)',
    'Capacity Loss (slow cycle time) (parts)', 'Capacity Gain (fast cycle time) (parts)',
    'Total Capacity Loss (parts)', 'Total Capacity Loss (sec)',
    'Target Output (parts)', 'Gap to Target (parts)',
    'Capacity Loss (vs Target) (parts)', 'Capacity Loss (vs Target) (sec)',
    'Total Shots (all)', 'Production Shots', 'Downtime Shots'
]

# ==================================================================
#                           DATA LOADING
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

@st.cache_data
def get_preprocessed_data(df_raw):
    """
    Standardizes columns and parses SHOT TIME for date filtering.
    This is run once before the main calculation.
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), None, None

    df = df_raw.copy()
    
    # --- Flexible Column Name Mapping ---
    column_variations = {
        'SHOT TIME': ['shot time', 'shot_time', 'timestamp', 'datetime'],
        'Plant Area': ['plant area', 'plant_area', 'area']
    }
    rename_dict = {}
    for standard_name, variations in column_variations.items():
        for col in df.columns:
            if str(col).strip().lower() in variations:
                rename_dict[col] = standard_name
                break
    df.rename(columns=rename_dict, inplace=True)

    if 'SHOT TIME' not in df.columns:
        st.error("Error: 'SHOT TIME' column not found.")
        return pd.DataFrame(), None, None
        
    try:
        df['SHOT TIME'] = pd.to_datetime(df['SHOT TIME'])
        df.dropna(subset=['SHOT TIME'], inplace=True)
        
        df['date'] = df['SHOT TIME'].dt.date
        df['week'] = df['SHOT TIME'].dt.to_period('W')
        df['month'] = df['SHOT TIME'].dt.to_period('M')
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        return df, min_date, max_date
        
    except Exception as e:
        st.error(f"Error parsing 'SHOT TIME' column: {e}")
        return pd.DataFrame(), None, None

# ==================================================================
#                           CORE CALCULATION
# ==================================================================

def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc_slider, mode_ct_tolerance, rr_downtime_gap, run_interval_hours):
    """
    Core function to process the raw DataFrame and calculate all Capacity Risk fields
    using the hybrid RR (downtime) + CR (inefficiency) logic.
    This function ALWAYS calculates vs Optimal (Approved CT).
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
        df['Working Cavities'] = pd.to_numeric(df['Working Cavities'], errors='coerce')
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
        df['Actual CT'] = pd.to_numeric(df['Actual CT'], errors='coerce')
        df['Approved CT'] = pd.to_numeric(df['Approved CT'], errors='coerce')
        
        # Drop rows where essential data could not be parsed
        df.dropna(subset=['SHOT TIME', 'Actual CT', 'Approved CT'], inplace=True)
        
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
    
    # 1. Sort all shots by time
    df_rr = df_production_only.sort_values("SHOT TIME").reset_index(drop=True)

    # 2. Calculate time differences
    is_hard_stop_code = df_rr["Actual CT"] >= 999.9
    
    # This finds the true gaps between *production runs*
    df_production_gaps = df_rr[~is_hard_stop_code]["SHOT TIME"].diff().dt.total_seconds()
    df_rr["run_break_time_diff"] = df_production_gaps.fillna(0.0)
    
    # This finds gaps between *all shots* (for RR stoppage logic)
    df_rr["rr_time_diff"] = df_rr["SHOT TIME"].diff().dt.total_seconds().fillna(0.0)

    # 4. Identify global "Run Breaks"
    run_break_threshold_sec = run_interval_hours * 3600
    is_run_break = df_rr["run_break_time_diff"] > run_break_threshold_sec
    df_rr['is_run_break'] = is_run_break
    
    # 5. Assign a *global* run_id (0-based index)
    df_rr['run_id'] = is_run_break.cumsum()

    # 6. Initialize all computed columns
    df_rr['mode_ct'] = 0.0
    df_rr['mode_lower_limit'] = 0.0
    df_rr['mode_upper_limit'] = 0.0
    df_rr['approved_ct_for_run'] = 0.0
    df_rr['reference_ct'] = 0.0
    df_rr['stop_flag'] = 0
    df_rr['adj_ct_sec'] = 0.0
    df_rr['parts_gain'] = 0.0
    df_rr['parts_loss'] = 0.0
    df_rr['time_gain_sec'] = 0.0
    df_rr['time_loss_sec'] = 0.0
    df_rr['Shot Type'] = 'N/A'
    df_rr['Mode CT Lower'] = 0.0
    df_rr['Mode CT Upper'] = 0.0
    
    # 7. Calculate Mode CT *per global run*
    df_for_mode = df_rr[df_rr["Actual CT"] < 999.9]
    run_modes = df_for_mode.groupby('run_id')['Actual CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['mode_ct'] = df_rr['run_id'].map(run_modes)
    df_rr['mode_lower_limit'] = df_rr['mode_ct'] * (1 - mode_ct_tolerance)
    df_rr['mode_upper_limit'] = df_rr['mode_ct'] * (1 + mode_ct_tolerance)

    # 8. Calculate Approved CT *per global run*
    run_approved_cts = df_rr.groupby('run_id')['Approved CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['approved_ct_for_run'] = df_rr['run_id'].map(run_approved_cts)
    
    # 9. Set REFERENCE_CT (always Approved CT in this function)
    df_rr['reference_ct'] = df_rr['approved_ct_for_run']

    # 10. Run Stop Detection on *all shots*
    prev_actual_ct = df_rr["Actual CT"].shift(1).fillna(0)
    in_mode_band = (df_rr["Actual CT"] >= df_rr['mode_lower_limit']) & (df_rr["Actual CT"] <= df_rr['mode_upper_limit'])
    
    # --- v8.5: This logic now *removes* the & ~is_run_break ---
    # A time gap is a stop
    is_time_gap = (df_rr["rr_time_diff"] > (prev_actual_ct + rr_downtime_gap))
    
    # An abnormal cycle is a stop
    is_abnormal_cycle = ~in_mode_band & ~is_hard_stop_code
    
    # Flag all three types of stops
    df_rr["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code, 1, 0)
    
    # --- v8.5: FIX to match run_rate_app.py chunk logic ---
    # Force the *first shot of every run* to be a production shot.
    if not df_rr.empty:
        first_shot_of_each_run_idx = df_rr.groupby('run_id').head(1).index
        df_rr.loc[first_shot_of_each_run_idx, "stop_flag"] = 0
    # --- End v8.5 Fix ---
    
    df_rr['adj_ct_sec'] = df_rr['Actual CT']
    # Use 'rr_time_diff' for the stoppage time
    df_rr.loc[is_time_gap, 'adj_ct_sec'] = df_rr['rr_time_diff']
    df_rr.loc[is_hard_stop_code, 'adj_ct_sec'] = 0 
    df_rr.loc[is_run_break, 'adj_ct_sec'] = 0 # Run break gaps are NOT downtime

    # 11. Separate all shots into Production and Downtime
    df_production = df_rr[df_rr['stop_flag'] == 0].copy()
    df_downtime   = df_rr[df_rr['stop_flag'] == 1].copy()

    # 12. Calculate per-shot losses/gains (with floating point fix)
    
    # A shot is "Slow" only if it's measurably greater than the reference
    is_slow = (df_production['Actual CT'] > df_production['reference_ct']) & \
              ~np.isclose(df_production['Actual CT'], df_production['reference_ct'])
    
    # A shot is "Fast" only if it's measurably less than the reference
    is_fast = (df_production['Actual CT'] < df_production['reference_ct']) & \
              ~np.isclose(df_production['Actual CT'], df_production['reference_ct'])
    
    # A shot is "On Target" only if it's "close enough"
    is_on_target = np.isclose(df_production['Actual CT'], df_production['reference_ct'])
    
    df_production['parts_gain'] = np.where(
        is_fast,
        ((df_production['reference_ct'] - df_production['Actual CT']) / df_production['reference_ct']) * df_production['Working Cavities'],
        0
    )
    df_production['parts_loss'] = np.where(
        is_slow,
        ((df_production['Actual CT'] - df_production['reference_ct']) / df_production['reference_ct']) * df_production['Working Cavities'],
        0
    )
    df_production['time_gain_sec'] = np.where(
        is_fast,
        (df_production['reference_ct'] - df_production['Actual CT']),
        0
    )
    df_production['time_loss_sec'] = np.where(
        is_slow,
        (df_production['Actual CT'] - df_production['reference_ct']),
        0
    )
    
    # Update df_rr with the values from df_production
    df_rr.update(df_production[['parts_gain', 'parts_loss', 'time_gain_sec', 'time_loss_sec']])

    # 13. Add Shot Type and date
    conditions = [is_slow, is_fast, is_on_target]
    choices = ['Slow', 'Fast', 'On Target']
    df_production['Shot Type'] = np.select(conditions, choices, default='N/A')
    
    # Update df_rr with the new 'Shot Type'
    df_rr['Shot Type'] = df_production['Shot Type'] 
    df_rr.loc[is_run_break, 'Shot Type'] = 'Run Break (Excluded)'
    df_rr['Shot Type'].fillna('RR Downtime (Stop)', inplace=True) 
    
    df_rr['date'] = df_rr['SHOT TIME'].dt.date
    df_production['date'] = df_production['SHOT TIME'].dt.date
    df_downtime['date'] = df_downtime['SHOT TIME'].dt.date
    
    # 14. Add Mode CT band columns for the chart
    df_rr['Mode CT Lower'] = df_rr['mode_lower_limit']
    df_rr['Mode CT Upper'] = df_rr['mode_upper_limit']

    all_shots_list = [df_rr] # Store the processed df
    
    # 15. Group by Day *after* all logic is applied
    daily_results_list = []
    
    if df_rr.empty:
        st.warning("No data found to process.")
        return None, None

    for date, daily_df in df_rr.groupby('date'):

        results = {col: 0 for col in ALL_RESULT_COLUMNS} # Pre-fill all with 0
        results['Date'] = date
        
        daily_prod = df_production[df_production['date'] == date]
        daily_down = df_downtime[df_downtime['date'] == date]

        # Get Wall Clock Time
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        results['Filtered Run Time (sec)'] = base_run_time_sec

        # Get Config (Max Cavities & Avg Reference CT)
        max_cavities = daily_df['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = daily_df['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct): avg_reference_ct = 1
            
        avg_approved_ct = daily_df['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct): avg_approved_ct = 1

        # Calculate The 4 Segments (in Parts)
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities
        results['Capacity Loss (downtime) (sec)'] = daily_down['adj_ct_sec'].sum()
        results['Actual Output (parts)'] = daily_prod['Working Cavities'].sum()
        
        # --- FIX for 'Actual Production Time' ---
        # Match run_rate_app.py: Sum the CT of production shots directly.
        results['Actual Cycle Time Total (sec)'] = daily_prod['Actual CT'].sum()
        
        # Inefficiency (CT Slow/Fast) Loss
        results['Capacity Gain (fast cycle time) (sec)'] = daily_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = daily_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = daily_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = daily_prod['parts_gain'].sum()
        
        # Reconciliation
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        
        # Final Aggregations
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        net_cycle_loss_sec = results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # Target Calculations
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        
        # This value is RE-CALCULATED after aggregation in the main app.
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities

        # New Shot Counts
        results['Total Shots (all)'] = len(daily_df)
        results['Production Shots'] = len(daily_prod)
        results['Downtime Shots'] = len(daily_down)

        daily_results_list.append(results)

    # 16. Format and Return Final DataFrame
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list).replace([np.inf, -np.inf], np.nan).fillna(0)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')

    if not all_shots_list:
        return final_df, pd.DataFrame()

    all_shots_df = pd.concat(all_shots_list, ignore_index=True)
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date
    
    return final_df, all_shots_df


def calculate_run_summaries(all_shots_df, target_output_perc_slider):
    """
    Takes the full, processed all_shots_df and aggregates it by run_id
    instead of by date.
    """
    
    if all_shots_df.empty or 'run_id' not in all_shots_df.columns:
        return pd.DataFrame()
    
    run_summary_list = []
    
    # Group by the global run_id
    for run_id, df_run in all_shots_df.groupby('run_id'):
        
        results = {col: 0 for col in ALL_RESULT_COLUMNS}
        results['run_id'] = run_id
        
        run_prod = df_run[df_run['stop_flag'] == 0]
        run_down = df_run[df_run['stop_flag'] == 1]

        # Get Wall Clock Time
        first_shot_time = df_run['SHOT TIME'].min()
        last_shot_time = df_run['SHOT TIME'].max()
        last_shot_ct = df_run.iloc[-1]['Actual CT'] if not df_run.empty else 0
        
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        results['Filtered Run Time (sec)'] = base_run_time_sec
        
        # Get Config
        max_cavities = df_run['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = df_run['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct): avg_reference_ct = 1
            
        avg_approved_ct = df_run['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct): avg_approved_ct = 1
            
        df_run_prod_for_mode = df_run[df_run["Actual CT"] < 999.9]
        if not df_run_prod_for_mode.empty:
            results['Mode CT'] = df_run_prod_for_mode['Actual CT'].mode().iloc[0] if not df_run_prod_for_mode['Actual CT'].mode().empty else 0.0
        else:
            results['Mode CT'] = 0.0

        # Calculate Segments
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities
        results['Capacity Loss (downtime) (sec)'] = run_down['adj_ct_sec'].sum()
        results['Actual Output (parts)'] = run_prod['Working Cavities'].sum()
        
        # --- FIX for 'Actual Production Time' ---
        # Match run_rate_app.py: Sum the CT of production shots directly.
        results['Actual Cycle Time Total (sec)'] = run_prod['Actual CT'].sum()

        results['Capacity Gain (fast cycle time) (sec)'] = run_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = run_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = run_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = run_prod['parts_gain'].sum()

        # Reconciliation
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        
        # Final Aggregations
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + net_cycle_loss_parts
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']

        # Target Calcs
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities

        # Shot Counts
        results['Total Shots (all)'] = len(df_run)
        results['Production Shots'] = len(run_prod)
        results['Downtime Shots'] = len(run_down)
        
        # Add start time for charting
        results['Start Time'] = first_shot_time

        run_summary_list.append(results)

    if not run_summary_list:
        return pd.DataFrame()
        
    run_summary_df = pd.DataFrame(run_summary_list).replace([np.inf, -np.inf], np.nan).fillna(0)
    run_summary_df = run_summary_df.set_index('run_id')
    
    return run_summary_df


# ==================================================================
#                       CACHING WRAPPER
# ==================================================================

@st.cache_data
def run_capacity_calculation_cached_v2(raw_data_df, toggle, cavities, target_output_perc_slider, mode_tol, rr_gap, run_interval, _cache_version=None):
    """Cached wrapper for the main calculation function."""
    if raw_data_df.empty:
        st.warning("No data found for the selected period.")
        return pd.DataFrame(), pd.DataFrame()
        
    return calculate_capacity_risk(
        raw_data_df,
        toggle,
        cavities,
        target_output_perc_slider,
        mode_tol,      
        rr_gap,        
        run_interval    
    )

# ==================================================================
#                       TABS 2 & 3 (COMMENTED)
# ==================================================================
# (All of Tab 2 and Tab 3 functions remain commented out)
