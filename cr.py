import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
# import plotly.express as px # v7.45: Import Plotly Express for Treemap
from datetime import datetime, timedelta # v7.40: Import timedelta
import io # v7.40: Import io for text parsing
from dateutil.relativedelta import relativedelta # v7.42: Import for monthly forecast

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
# v7.53: Refactored UI into modular functions
__version__ = "v7.53 (Modular Refactor)"
# ==================================================================

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

# --- v6.89: Define all_result_columns globally to fix NameError ---
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
#                           DATA CALCULATION
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
# --- v6.64: This function is now ONLY run vs Optimal (100%) ---
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc_slider, mode_ct_tolerance, rr_downtime_gap, run_interval_hours):
    """
    Core function to process the raw DataFrame and calculate all Capacity Risk fields
    using the new hybrid RR (downtime) + CR (inefficiency) logic.
    
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

    # --- v6.88: NEW LOGIC - Process *entire* dataframe first ---
    
    # 1. Sort all shots by time
    df_rr = df_production_only.sort_values("SHOT TIME").reset_index(drop=True)

    # --- v7.06: FIX for Run Breaks vs RR Stoppages ---
    is_hard_stop_code = df_rr["Actual CT"] >= 999.9
    
    # 2. Calculate `run_break_time_diff` (for Run Interval)
    # This finds the true gaps between *production runs*
    df_production_gaps = df_rr[~is_hard_stop_code]["SHOT TIME"].diff().dt.total_seconds()
    df_rr["run_break_time_diff"] = df_production_gaps
    df_rr["run_break_time_diff"].fillna(0.0, inplace=True)
    df_rr.loc[0, "run_break_time_diff"] = 0.0 # First shot has no gap
    
    # 3. Calculate `rr_time_diff` (for RR Downtime)
    # This finds gaps between *all shots* (for RR stoppage logic)
    df_rr["rr_time_diff"] = df_rr["SHOT TIME"].diff().dt.total_seconds()
    df_rr["rr_time_diff"].fillna(0.0, inplace=True)
    df_rr.loc[0, "rr_time_diff"] = 0.0 # First shot has no gap
    # --- End v7.06 Fix ---

    # 4. Identify global "Run Breaks"
    run_break_threshold_sec = run_interval_hours * 3600
    # Use the production-only gap calculation for this
    is_run_break = df_rr["run_break_time_diff"] > run_break_threshold_sec
    df_rr['is_run_break'] = is_run_break # Store this for later
    
    # 5. Assign a *global* run_id
    df_rr['run_id'] = is_run_break.cumsum()

    # ==================================================================
    # --- v7.11: KEY BUG FIX ---
    # Initialize *all* computed columns on df_rr first.
    # This ensures that all_shots_df (which is made from df_rr)
    # has these columns, even if logic fails or edge cases occur.
    # This fixes KeyErrors in 'by Run' mode and the Shot Chart.
    # ==================================================================
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
    # ==================================================================
    
    # 6. Calculate Mode CT *per global run*
    df_for_mode = df_rr[df_rr["Actual CT"] < 999.9]
    # --- v7.13: Reverted .median() back to .mode() to match run_rate_app ---
    run_modes = df_for_mode.groupby('run_id')['Actual CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['mode_ct'] = df_rr['run_id'].map(run_modes)
    df_rr['mode_lower_limit'] = df_rr['mode_ct'] * (1 - mode_ct_tolerance)
    df_rr['mode_upper_limit'] = df_rr['mode_ct'] * (1 + mode_ct_tolerance)

    # 7. Calculate Approved CT *per global run*
    run_approved_cts = df_rr.groupby('run_id')['Approved CT'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 0
    )
    df_rr['approved_ct_for_run'] = df_rr['run_id'].map(run_approved_cts)
    
    # 8. Set REFERENCE_CT (always Approved CT in this function)
    df_rr['reference_ct'] = df_rr['approved_ct_for_run']

    # 9. Run Stop Detection on *all shots*
    prev_actual_ct = df_rr["Actual CT"].shift(1).fillna(0)
    
    in_mode_band = (df_rr["Actual CT"] >= df_rr['mode_lower_limit']) & (df_rr["Actual CT"] <= df_rr['mode_upper_limit'])
    
    # --- v7.08: Remove '& ~in_mode_band' to fix downtime calc ---
    # Use the 'rr_time_diff' for this logic
    is_time_gap = (df_rr["rr_time_diff"] > (prev_actual_ct + rr_downtime_gap)) & ~is_run_break
    
    # --- v7.15: FINAL FIX - Remove '& ~is_run_break' ---
    # This correctly flags the first shot of a run as downtime
    # if it is an abnormal cycle.
    is_abnormal_cycle = ~in_mode_band & ~is_hard_stop_code
    
    # --- v7.51: Reverted v7.50 fix. ---
    # is_run_break is NOT a stop condition on its own.
    # A run break shot is only a stop if it's also an is_abnormal_cycle or is_time_gap.
    # This is the correct v7.49 logic.
    df_rr["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code, 1, 0)
    
    df_rr['adj_ct_sec'] = df_rr['Actual CT']
    # Use 'rr_time_diff' for the stoppage time
    df_rr.loc[is_time_gap, 'adj_ct_sec'] = df_rr['rr_time_diff']
    df_rr.loc[is_hard_stop_code, 'adj_ct_sec'] = 0 
    df_rr.loc[is_run_break, 'adj_ct_sec'] = 0 # Run break gaps are NOT downtime

    # 10. Separate all shots into Production and Downtime
    df_production = df_rr[df_rr['stop_flag'] == 0].copy()
    df_downtime   = df_rr[df_rr['stop_flag'] == 1].copy()

    # 11. Calculate per-shot losses/gains
    
    # --- v7.46: FIX for floating point bug ---
    # Use np.isclose to ensure logic is 100% consistent
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
    # --- End v7.46 Fix ---
    
    # Update df_rr with the values from df_production
    df_rr.update(df_production[['parts_gain', 'parts_loss', 'time_gain_sec', 'time_loss_sec']])

    # 12. Add Shot Type and date
    
    # --- v7.46: Use the same robust logic for Shot Type ---
    conditions = [
        is_slow,
        is_fast,
        is_on_target
    ]
    choices = ['Slow', 'Fast', 'On Target']
    df_production['Shot Type'] = np.select(conditions, choices, default='N/A')
    
    # Update df_rr with the new 'Shot Type'
    df_rr['Shot Type'] = df_production['Shot Type'] 
    df_rr.loc[is_run_break, 'Shot Type'] = 'Run Break (Excluded)'
    df_rr['Shot Type'].fillna('RR Downtime (Stop)', inplace=True) 
    # --- End v7.46 Fix ---
    
    df_rr['date'] = df_rr['SHOT TIME'].dt.date
    df_production['date'] = df_production['SHOT TIME'].dt.date # df_production is still used for daily calcs
    df_downtime['date'] = df_downtime['SHOT TIME'].dt.date   # df_downtime is still used for daily calcs
    
    # 13. Add Mode CT band columns for the chart
    df_rr['Mode CT Lower'] = df_rr['mode_lower_limit']
    df_rr['Mode CT Upper'] = df_rr['mode_upper_limit']

    all_shots_list = [df_rr] # Store the processed df
    
    # --- End v6.88 Global Processing ---

    # --- v6.88: NEW - Group by Day *after* all logic is applied ---
    daily_results_list = []
    
    if df_rr.empty:
        st.warning("No data found to process.")
        return None, None

    for date, daily_df in df_rr.groupby('date'):

        results = {col: 0 for col in ALL_RESULT_COLUMNS} # Pre-fill all with 0
        results['Date'] = date
        
        # Get the day's subsets from the pre-processed dataframes
        daily_prod = df_production[df_production['date'] == date]
        daily_down = df_downtime[df_downtime['date'] == date]

        # --- 6. Get Wall Clock Time (Basis for Segment 4) ---
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        # --- v6.90: BUG FIX ---
        results['Filtered Run Time (sec)'] = base_run_time_sec
        # --- End v6.90 Bug Fix ---

        # --- 9. Get Config (Max Cavities & Avg Reference CT) ---
        max_cavities = daily_df['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = daily_df['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct):
            avg_reference_ct = 1
            
        avg_approved_ct = daily_df['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct):
            avg_approved_ct = 1


        # --- 10. Calculate The 4 Segments (in Parts) ---

        # SEGMENT 4: Optimal Production (Benchmark)
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities

        # SEGMENT 3: RR Downtime Loss
        results['Capacity Loss (downtime) (sec)'] = daily_down['adj_ct_sec'].sum()

        # SEGMENT 1: Actual Production
        results['Actual Output (parts)'] = daily_prod['Working Cavities'].sum()
        
        # --- v7.16: RECONCILIATION ---
        # Force Actual Prod Time to be Run Time - Downtime to match RR app
        results['Actual Cycle Time Total (sec)'] = results['Filtered Run Time (sec)'] - results['Capacity Loss (downtime) (sec)']
        # --- END v7.16 ---

        # SEGMENT 2: Inefficiency (CT Slow/Fast) Loss
        results['Capacity Gain (fast cycle time) (sec)'] = daily_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = daily_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = daily_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = daily_prod['parts_gain'].sum()
        
        # --- v6.56: RECONCILIATION LOGIC ---
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        # --- END v.6.56 RECONCILIATION ---

        # --- 11. Final Aggregations ---
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        net_cycle_loss_sec = results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # --- v6.64: This function now ALSO calculates the Target values ---
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        
        # --- v7.39: This is the daily loss, which gets summed.
        # This value is RE-CALCULATED after aggregation in the main app.
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities


        # New Shot Counts
        results['Total Shots (all)'] = len(daily_df)
        results['Production Shots'] = len(daily_prod)
        results['Downtime Shots'] = len(daily_down)

        daily_results_list.append(results)

    # --- 12. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    final_df = pd.DataFrame(daily_results_list).replace([np.inf, -np.inf], np.nan).fillna(0)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')

    if not all_shots_list:
        return final_df, pd.DataFrame()

    all_shots_df = pd.concat(all_shots_list, ignore_index=True)
    
    # --- v7.03: 'run_id' increment moved to main app scope ---
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date
    
    return final_df, all_shots_df

# ==================================================================
# --- v6.89: NEW HELPER FUNCTION FOR 'BY RUN' AGGREGATION ---
# ==================================================================

def calculate_run_summaries(all_shots_df, target_output_perc_slider):
    """
    Takes the full, processed all_shots_df and aggregates it by run_id
    instead of by date.
    """
    
    # --- v6.97: Fix KeyError ---
    if all_shots_df.empty or 'run_id' not in all_shots_df.columns:
        return pd.DataFrame()
    # --- End Fix ---
    
    run_summary_list = []
    
    # Group by the global run_id
    for run_id, df_run in all_shots_df.groupby('run_id'):
        
        results = {col: 0 for col in ALL_RESULT_COLUMNS}
        results['run_id'] = run_id
        
        run_prod = df_run[df_run['stop_flag'] == 0]
        run_down = df_run[df_run['stop_flag'] == 1]

        # 1. Get Wall Clock Time for the run
        first_shot_time = df_run['SHOT TIME'].min()
        last_shot_time = df_run['SHOT TIME'].max()
        last_shot_ct = df_run.iloc[-1]['Actual CT'] if not df_run.empty else 0
        
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        base_run_time_sec = time_span_sec + last_shot_ct

        # --- v6.90: BUG FIX ---
        results['Filtered Run Time (sec)'] = base_run_time_sec
        # --- End v6.90 Bug Fix ---
        
        # 2. Get Config (Max Cavities & Avg Reference CT)
        max_cavities = df_run['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        avg_reference_ct = df_run['reference_ct'].mean()
        if avg_reference_ct == 0 or pd.isna(avg_reference_ct):
            avg_reference_ct = 1
            
        avg_approved_ct = df_run['approved_ct_for_run'].mean()
        if avg_approved_ct == 0 or pd.isna(avg_approved_ct):
            avg_approved_ct = 1
            
        # --- v7.11: Add Mode CT for the run ---
        df_run_prod_for_mode = df_run[df_run["Actual CT"] < 999.9]
        if not df_run_prod_for_mode.empty:
            # --- v7.13: Reverted .median() back to .mode() ---
            results['Mode CT'] = df_run_prod_for_mode['Actual CT'].mode().iloc[0] if not df_run_prod_for_mode['Actual CT'].mode().empty else 0.0
        else:
            results['Mode CT'] = 0.0

        # --- v7.21: Add Avg Actual CT and Std/Approved CT for table ---
        # --- v7.22: REMOVED Avg Actual CT and Std/Approved CT ---
        # results['Avg Actual CT'] = run_prod['Actual CT'].mean() if not run_prod.empty else 0.0
        # results['Std/Approved CT'] = avg_approved_ct

        # 3. Calculate Segments
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / avg_reference_ct) * max_cavities
        results['Capacity Loss (downtime) (sec)'] = run_down['adj_ct_sec'].sum()
        results['Actual Output (parts)'] = run_prod['Working Cavities'].sum()
        
        # --- v7.16: RECONCILIATION ---
        # Force Actual Prod Time to be Run Time - Downtime to match RR app
        results['Actual Cycle Time Total (sec)'] = results['Filtered Run Time (sec)'] - results['Capacity Loss (downtime) (sec)']
        # --- END v7.16 ---

        # --- v6.95: Fix KeyError ---
        results['Capacity Gain (fast cycle time) (sec)'] = run_prod['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = run_prod['time_loss_sec'].sum()
        results['Capacity Loss (slow cycle time) (parts)'] = run_prod['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = run_prod['parts_gain'].sum()
        # --- End v6.95 Fix ---

        # 4. Reconciliation
        true_capacity_loss_parts = results['Optimal Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        
        # 5. Final Aggregations
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + net_cycle_loss_parts
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']

        # 6. Target Calcs
        target_perc_ratio = target_output_perc_slider / 100.0
        optimal_100_parts = (results['Filtered Run Time (sec)'] / avg_approved_ct) * max_cavities
        results['Target Output (parts)'] = optimal_100_parts * target_perc_ratio
        
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)']
        
        # --- v7.39: This is the run-level loss, which gets summed.
        # This value is RE-CALCULATED after aggregation in the main app.
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)'])
        
        results['Capacity Loss (vs Target) (sec)'] = (results['Capacity Loss (vs Target) (parts)'] * avg_reference_ct) / max_cavities

        # 7. Shot Counts
        results['Total Shots (all)'] = len(df_run)
        results['Production Shots'] = len(run_prod)
        results['Downtime Shots'] = len(run_down)
        
        # 8. Add start time for charting
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
# --- v7.48: Renamed function to bust cache ---
def run_capacity_calculation_cached_v2(raw_data_df, toggle, cavities, target_output_perc_slider, mode_tol, rr_gap, run_interval, _cache_version=None):
    """Cached wrapper for the main calculation function."""
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
#                       NEW TAB 2 FUNCTION (v7.45)
# ==================================================================

# --- v7.52: Commented out Tab 2 ---
# def render_automated_risk_tab(all_time_summary_df, all_time_totals):
#     """
#     Renders the new "Automated Risk Analysis" tab (Tab 2).
#     This tab shows the historical gap to optimal and its causes.
#     """
#     st.header("Automated Risk Analysis")
#     st.info("This tab analyzes your entire historical dataset to show the gap between your actual performance and your 100% optimal potential, highlighting the key drivers of that loss.")
    
#     # Extract totals calculated in the main app logic
#     total_optimal_100 = all_time_totals['total_optimal_100']
#     total_produced = all_time_totals['total_produced']
#     total_true_net_loss_parts = all_time_totals['total_true_net_loss_parts']
#     total_downtime_loss_parts = all_time_totals['total_downtime_loss_parts']
#     total_slow_loss_parts = all_time_totals['total_slow_loss_parts']
#     total_fast_gain_parts = all_time_totals['total_fast_gain_parts']
#     total_net_cycle_loss_parts = all_time_totals['total_net_cycle_loss_parts']

#     total_prod_sec_hist = all_time_summary_df['Actual Cycle Time Total (sec)'].sum()
#     total_run_time_sec_hist = all_time_summary_df['Filtered Run Time (sec)'].sum()
#     total_downtime_sec_hist = all_time_summary_df['Capacity Loss (downtime) (sec)'].sum()
#     total_stops_hist = all_time_summary_df['Downtime Shots'].sum() # Note: This is a proxy. A better 'stops' sum may be needed.
    
#     # --- 1. Top-Level KPIs ---
#     st.subheader("Historical Performance vs. Optimal")
    
#     c1, c2, c3 = st.columns(3)
#     c1.metric("Optimal Output (100%)", f"{total_optimal_100:,.0f} parts")
#     c2.metric("Actual Output", f"{total_produced:,.0f} parts", f"{((total_produced / total_optimal_100) * 100):.1f}% of Optimal")
#     c3.metric("Total Capacity Loss", f"{total_true_net_loss_parts:,.0f} parts", delta_color="inverse")
    
#     st.divider()
    
#     # --- 2. Loss Driver Analysis ---
#     st.subheader("Historical Loss Driver Analysis")
    
#     c1, c2 = st.columns([1, 1])
    
#     with c1:
#         st.markdown("##### Loss Breakdown (Treemap)")
        
#         # Prepare data for the treemap
#         loss_data = {
#             'Category': ['Total Capacity Loss', 'Total Capacity Loss', 'Total Capacity Loss'],
#             'Loss Type': ['Loss (RR Downtime)', 'Loss (Slow Cycles)', 'Gain (Fast Cycles)'],
#             'Parts': [total_downtime_loss_parts, total_slow_loss_parts, total_fast_gain_parts]
#         }
#         df_loss_treemap = pd.DataFrame(loss_data)
        
#         # Create the treemap
#         fig_treemap = px.treemap(
#             df_loss_treemap,
#             path=['Category', 'Loss Type'],
#             values='Parts',
#             color='Loss Type',
#             color_discrete_map={
#                 'Loss (RR Downtime)': '#ff6961', # Red
#                 'Loss (Slow Cycles)': '#ffb347', # Orange
#                 'Gain (Fast Cycles)': '#77dd77'  # Green
#             },
#             title="Root Causes of Capacity Loss (vs. Optimal)"
#         )
#         fig_treemap.update_traces(
#             textinfo="label+value+percent root",
#             hovertemplate='<b>%{label}</b><br>Parts: %{value:,.0f}<br>%{percentParent:.1%} of Total Loss'
#         )
#         fig_treemap.update_layout(margin = dict(t=50, l=0, r=0, b=0))
#         st.plotly_chart(fig_treemap, use_container_width=True)

#     with c2:
#         st.markdown("##### Key Performance Levers")
#         st.info("These are your historical averages. You can use these as levers to model scenarios in the 'Demand & Capacity Planning' tab.")
        
#         # Calculate historical metrics
#         hist_uptime_perc = (total_prod_sec_hist / total_run_time_sec_hist) * 100 if total_run_time_sec_hist > 0 else 0
#         hist_parts_per_uptime_hour = (total_produced / (total_prod_sec_hist / 3600)) if total_prod_sec_hist > 0 else 0
#         hist_mttr_min = (total_downtime_sec_hist / 60 / total_stops_hist) if total_stops_hist > 0 else 0
#         hist_mtbf_min = (total_prod_sec_hist / 60 / total_stops_hist) if total_stops_hist > 0 else (total_prod_sec_hist / 60)

#         st.metric("Historical Uptime %", f"{hist_uptime_perc:.1f}%", help="The percentage of your total run time that was actual production (Uptime).")
#         st.metric("Historical Production Rate", f"{hist_parts_per_uptime_hour:,.0f} parts / hr", help="Your average production rate *during Uptime*.")
#         st.metric("Historical MTTR", f"{format_seconds_to_dhm(hist_mttr_min * 60)}", help="Mean Time To Repair: The average duration of a stop event.")
#         st.metric("Historical MTBF", f"{format_seconds_to_dhm(hist_mtbf_min * 60)}", help="Mean Time Between Failures: The average duration of an Uptime period *between* stops.")


# ==================================================================
#                       NEW TAB 3 FUNCTION (v7.45)
# ==================================================================

# --- v7.52: Commented out Tab 3 ---
# def render_demand_planning_tab(daily_summary_df, all_shots_df, all_time_summary_df):
#     st.header("Capacity & Demand Planning")
    
#     # --- 1. Calculate Historical Baselines ---
#     total_parts_hist = all_time_summary_df['Actual Output (parts)'].sum()
#     total_prod_sec_hist = all_time_summary_df['Actual Cycle Time Total (sec)'].sum()
#     total_run_time_sec_hist = all_time_summary_df['Filtered Run Time (sec)'].sum()

#     # --- v7.43: Calculate "Current Trend" (Uptime Rate) ---
#     # --- v7.44: Renamed for clarity ---
#     default_parts_per_uptime_hour = (total_parts_hist / (total_prod_sec_hist / 3600)) if total_prod_sec_hist > 0 else 0
#     default_uptime_perc = (total_prod_sec_hist / total_run_time_sec_hist) * 100 if total_run_time_sec_hist > 0 else 0
    
#     # --- v7.43: Calculate "Max Possible" (Optimal Rate) ---
#     total_optimal_parts_hist = all_time_summary_df['Optimal Output (parts)'].sum()
#     default_parts_per_total_hour = (total_optimal_parts_hist / (total_run_time_sec_hist / 3600)) if total_run_time_sec_hist > 0 else 0


#     st.info(f"""
#     This tool forecasts your ability to meet future demand. It's pre-filled with your historical averages from the 'Automated Risk Analysis' tab:
#     - **Baseline Uptime %:** `{default_uptime_perc:.1f}%`
#     - **Baseline Production Rate:** `{default_parts_per_uptime_hour:,.0f} parts / hour` (during uptime)
#     - **Max Possible Rate:** `{default_parts_per_total_hour:,.0f} parts / hour` (at 100% optimal)
#     """)
    
#     # --- 2. 12-Month Strategic Model ---
#     st.subheader("1. 12-Month Strategic Forecast")
    
#     with st.container(border=True):
#         st.markdown("##### Scenario Inputs")
        
#         c1, c2 = st.columns(2)
#         with c1:
#             working_hours_per_month = st.number_input(
#                 "Total Working Hours / Month",
#                 min_value=1, max_value=744, value=352, # 744=24*31
#                 help="Your total available operating hours per month (e.g., 22 days * 16 hours = 352)"
#             )
#         with c2:
#             # --- v7.44: Renamed "Stability" to "Uptime %" ---
#             baseline_uptime_perc = st.slider(
#                 "Projected Uptime %",
#                 min_value=0.0, max_value=100.0, value=default_uptime_perc, step=0.5, format="%.1f%%",
#                 help="Your expected uptime (e.g., Uptime / (Uptime + Downtime)). Defaulted from your historical average."
#             )
            
#         # --- v7.43: New Input Sliders ---
#         st.markdown("###### Production Rate Scenarios")
#         c1, c2 = st.columns(2)
#         with c1:
#             projected_uptime_rate = st.slider(
#                 "Projected Uptime Rate (Parts/hr)",
#                 min_value=0.0, 
#                 max_value=default_parts_per_uptime_hour * 2 if default_parts_per_uptime_hour > 0 else 1000.0, 
#                 value=default_parts_per_uptime_hour, 
#                 step=max(1.0, default_parts_per_uptime_hour * 0.01), # 1% step
#                 format="%.0f parts/hr",
#                 help="Your expected production rate *during uptime*. (DEFAULT = CURRENT TREND)"
#             )
#         with c2:
#             projected_max_rate = st.slider(
#                 "Maximum Possible Rate (Parts/hr)",
#                 min_value=0.0, 
#                 max_value=default_parts_per_total_hour * 1.5 if default_parts_per_total_hour > 0 else 1000.0, 
#                 value=default_parts_per_total_hour, 
#                 step=max(1.0, default_parts_per_total_hour * 0.01), # 1% step
#                 format="%.0f parts/hr",
#                 help="Your theoretical max production rate at 100% stability. (DEFAULT = 100% OPTIMAL)"
#             )
            
#         # --- v7.43: Updated Capacity Calculations ---
#         # --- v7.44: Renamed "baseline_stability" to "baseline_uptime_perc" ---
#         projected_uptime_hours = working_hours_per_month * (baseline_uptime_perc / 100.0)
#         projected_monthly_capacity = projected_uptime_hours * projected_uptime_rate
#         projected_max_monthly_capacity = working_hours_per_month * projected_max_rate

#         c1, c2 = st.columns(2)
#         with c1:
#             st.metric(
#                 "Forecasted Capacity (at Current Trend)",
#                 f"{projected_monthly_capacity:,.0f} parts / month"
#             )
#             st.caption(f"{working_hours_per_month:,.0f} hr * {baseline_uptime_perc:.1f}% Uptime * {projected_uptime_rate:,.0f} parts/hr")
#         with c2:
#             st.metric(
#                 "Maximum Capacity (at 100% Optimal)",
#                 f"{projected_max_monthly_capacity:,.0f} parts / month"
#             )
#             st.caption(f"{working_hours_per_month:,.0f} hr * {projected_max_rate:,.0f} parts/hr")
        
#         st.markdown("##### Demand Inputs")
        
#         start_inventory_monthly = st.number_input("Current Inventory (Starting Stock)", value=0, min_value=0, step=1000, key="monthly_stock")
        
#         # --- v7.44: New Demand Input UI ---
#         demand_input_type = st.radio(
#             "Demand Input Method",
#             ["Monthly", "Quarterly", "Yearly"],
#             horizontal=True,
#             key="demand_input_type"
#         )
        
#         demand_values = {}
        
#         if demand_input_type == "Monthly":
#             st.markdown("###### Enter Demand for Each Month")
#             cols = st.columns(4)
#             months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#             for i, month in enumerate(months):
#                 with cols[i % 4]:
#                     demand_values[i] = st.number_input(f"Month {i+1} ({month})", value=50000, min_value=0, step=100, key=f"month_{i}")
        
#         elif demand_input_type == "Quarterly":
#             st.markdown("###### Enter Demand for Each Quarter")
#             cols = st.columns(4)
#             for i in range(4):
#                 with cols[i]:
#                     demand_values[i] = st.number_input(f"Q{i+1} Demand (per month)", value=50000, min_value=0, step=100, key=f"q_{i}")
        
#         elif demand_input_type == "Yearly":
#             st.markdown("###### Enter Total Annual Demand")
#             demand_values[0] = st.number_input("Total Yearly Demand", value=600000, min_value=0, step=1000, key="year_0")

#         if st.button("Run 12-Month Forecast"):
#             try:
#                 # --- v7.44: Process new demand inputs ---
#                 demand_list = []
#                 if demand_input_type == "Monthly":
#                     demand_list = [demand_values[i] for i in range(12)]
#                 elif demand_input_type == "Quarterly":
#                     demand_list.extend([demand_values[0]] * 3) # Q1
#                     demand_list.extend([demand_values[1]] * 3) # Q2
#                     demand_list.extend([demand_values[2]] * 3) # Q3
#                     demand_list.extend([demand_values[3]] * 3) # Q4
#                 elif demand_input_type == "Yearly":
#                     monthly_avg = demand_values[0] / 12.0
#                     demand_list = [monthly_avg] * 12
                
#                 if len(demand_list) != 12:
#                     st.error(f"Error: Could not generate 12 months of demand. Please check inputs.")
#                 else:
#                     # --- v7.43: Get historical monthly actuals ---
#                     hist_actuals_monthly = daily_summary_df['Actual Output (parts)'].resample('ME').sum()
#                     hist_actuals_monthly.index = hist_actuals_monthly.index.strftime('%Y-%m')
                    
#                     forecast_data = []
#                     current_inventory = start_inventory_monthly
#                     start_month = datetime.now()
                    
#                     for i, demand in enumerate(demand_list):
#                         current_month_dt = start_month + relativedelta(months=i)
#                         month_name = current_month_dt.strftime('%Y-%m')
                        
#                         # Check if we have historical data for this month
#                         if month_name in hist_actuals_monthly.index:
#                             actual_production = hist_actuals_monthly.loc[month_name]
#                             status = "Actual"
#                             monthly_production = actual_production
#                         else:
#                             status = "Forecast"
#                             monthly_production = projected_monthly_capacity
                        
#                         monthly_net = monthly_production - demand
#                         ending_inventory = current_inventory + monthly_net
                        
#                         forecast_data.append({
#                             "Month": month_name,
#                             "Status": status,
#                             "Demand": demand,
#                             "Production": monthly_production,
#                             "Monthly Net": monthly_net,
#                             "Projected End Inventory": ending_inventory,
#                             "Forecasted Capacity": projected_monthly_capacity, # Keep for chart line
#                             "Max Capacity": projected_max_monthly_capacity # Keep for chart line
#                         })
                        
#                         current_inventory = ending_inventory # Set for next loop
                    
#                     df_forecast_monthly = pd.DataFrame(forecast_data)
                    
#                     # --- KPIs for Monthly Forecast ---
#                     end_inventory_monthly = df_forecast_monthly['Projected End Inventory'].iloc[-1]
#                     min_inventory_monthly = df_forecast_monthly['Projected End Inventory'].min()
#                     shortfall_month = None
#                     if min_inventory_monthly < 0:
#                         shortfall_month = df_forecast_monthly[df_forecast_monthly['Projected End Inventory'] < 0]['Month'].iloc[0]

#                     st.markdown("##### 12-Month Forecast Results")
#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("Projected Year-End Inventory", f"{end_inventory_monthly:,.0f} parts")
#                     c2.metric("Projected Max Shortfall", f"{min_inventory_monthly:,.0f} parts" if min_inventory_monthly < 0 else "No Shortfall", delta_color="inverse")
#                     c3.metric("Projected Shortfall Month", shortfall_month if shortfall_month else "None")

#                     # --- v7.45: New Risk Driver Analysis ---
#                     if min_inventory_monthly < 0:
#                         st.markdown("##### Capacity Risk Drivers")
#                         with st.container(border=True):
#                             st.error(f"**Action Required:** Your forecast shows a maximum shortfall of **{min_inventory_monthly:,.0f} parts**.")
                            
#                             # Calculate what's needed to fix it
#                             total_demand = df_forecast_monthly['Demand'].sum()
#                             total_actual_prod = df_forecast_monthly[df_forecast_monthly['Status'] == 'Actual']['Production'].sum()
                            
#                             # Find demand just for forecasted months
#                             forecast_demand = df_forecast_monthly[df_forecast_monthly['Status'] == 'Forecast']['Demand'].sum()
#                             num_forecast_months = len(df_forecast_monthly[df_forecast_monthly['Status'] == 'Forecast'])
                            
#                             if num_forecast_months > 0:
#                                 # We need to produce enough to cover forecast demand + starting inventory + shortfall
#                                 required_total_forecast_prod = forecast_demand - (start_inventory_monthly + total_actual_prod) - min_inventory_monthly
#                                 required_monthly_prod = required_total_forecast_prod / num_forecast_months
                                
#                                 # 1. How much more Uptime %?
#                                 required_uptime_hours = required_monthly_prod / projected_uptime_rate if projected_uptime_rate > 0 else 0
#                                 required_uptime_perc = (required_uptime_hours / working_hours_per_month) * 100 if working_hours_per_month > 0 else 0
                                
#                                 # 2. How much more Uptime Rate?
#                                 required_uptime_rate = required_monthly_prod / projected_uptime_hours if projected_uptime_hours > 0 else 0
                                
#                                 st.markdown("To close this gap (break-even), you would need to achieve one of the following for all future months:")
#                                 c1, c2 = st.columns(2)
#                                 c1.metric(
#                                     label=f"Increase Projected Uptime % to:",
#                                     value=f"{required_uptime_perc:.1f}%",
#                                     delta=f"{required_uptime_perc - baseline_uptime_perc:.1f}%"
#                                 )
#                                 c2.metric(
#                                     label=f"Increase Projected Uptime Rate to:",
#                                     value=f"{required_uptime_rate:,.0f} parts/hr",
#                                     delta=f"{required_uptime_rate - projected_uptime_rate:,.0f} parts/hr"
#                                 )
                            

#                     # --- v7.43: Updated Monthly Chart ---
#                     # --- v7.44: Changed demand bar color to blue ---
#                     fig_monthly = go.Figure()

#                     # Add Demand bars (Blue)
#                     fig_monthly.add_trace(go.Bar(
#                         x=df_forecast_monthly['Month'],
#                         y=df_forecast_monthly['Demand'],
#                         name='Monthly Demand',
#                         marker_color='#1f77b4' # Changed from red to blue
#                     ))
                    
#                     # Add Actual Production bars (Green) - will only show for "Actual" months
#                     df_actuals_chart = df_forecast_monthly[df_forecast_monthly['Status'] == 'Actual']
#                     if not df_actuals_chart.empty:
#                         fig_monthly.add_trace(go.Bar(
#                             x=df_actuals_chart['Month'],
#                             y=df_actuals_chart['Production'],
#                             name='Actual Production',
#                             marker_color='green'
#                         ))

#                     # Add Forecasted Capacity line (Dotted Blue)
#                     fig_monthly.add_trace(go.Scatter(
#                         x=df_forecast_monthly['Month'],
#                         y=df_forecast_monthly['Forecasted Capacity'],
#                         name='Forecasted Capacity (Trend)',
#                         mode='lines',
#                         line=dict(color='blue', dash='dot')
#                     ))
                    
#                     # Add Max Capacity line (Dotted Grey)
#                     fig_monthly.add_trace(go.Scatter(
#                         x=df_forecast_monthly['Month'],
#                         y=df_forecast_monthly['Max Capacity'],
#                         name='Max Possible Capacity',
#                         mode='lines',
#                         line=dict(color='grey', dash='dot')
#                     ))
                    
#                     # Add Inventory line on secondary axis (Solid Green)
#                     fig_monthly.add_trace(go.Scatter(
#                         x=df_forecast_monthly['Month'],
#                         y=df_forecast_monthly['Projected End Inventory'],
#                         name='Projected End Inventory',
#                         mode='lines+markers',
#                         line=dict(color='green'),
#                         yaxis="y2"
#                     ))
                    
#                     # Set barmode to group to see Demand vs Actual side-by-side
#                     fig_monthly.update_layout(barmode='group')

#                     fig_monthly.update_layout(
#                         title="12-Month Strategic Forecast",
#                         xaxis_title="Month",
#                         yaxis_title="Parts (Demand vs. Capacity)",
#                         yaxis2=dict(
#                             title="Parts (Projected Inventory)",
#                             overlaying="y",
#                             side="right"
#                         ),
#                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
#                     )
#                     st.plotly_chart(fig_monthly, use_container_width=True)
                    
#                     # --- Data Table ---
#                     # --- v7.43: Re-order and format table ---
#                     display_cols_monthly = [
#                         "Month", "Status", "Demand", "Production", 
#                         "Monthly Net", "Projected End Inventory"
#                     ]
#                     st.dataframe(df_forecast_monthly[display_cols_monthly].style.format({
#                         'Demand': '{:,.0f}',
#                         'Production': '{:,.0f}',
#                         'Monthly Net': '{:+,.0f}',
#                         'Projected End Inventory': '{:,.0f}'
#                     }).applymap(
#                         lambda v: 'color: red;' if v < 0 else ('color: green;' if v > 0 else None),
#                         subset=['Monthly Net']
#                     ).applymap(
#                         lambda v: 'color: red;' if v < 0 else None,
#                         subset=['Projected End Inventory']
#                     ), use_container_width=True)

#             except Exception as e:
#                 st.error(f"Error parsing demand data: {e}. Please ensure you have 12 valid numbers.")

#     # --- 3. Short-Term Daily Model (Original) ---
#     st.subheader("2. Short-Term Daily Forecast")
    
#     with st.expander("Expand for short-term daily planning"):
#         # --- Use the already-calculated dataframes from tab 1 ---
#         if 'daily_summary_df' not in locals():
#             # This is a fallback in case the function is called in a weird context
#             st.error("Daily summary data not found. Please reload the app.")
#             st.stop()
#         if 'all_shots_df' not in locals():
#             st.error("Shot data not loaded. Please ensure main report runs.")
#             st.stop()
            
#         min_data_date = all_shots_df['SHOT TIME'].dt.date.min()
#         max_data_date = all_shots_df['SHOT TIME'].dt.date.max()

#         st.markdown("##### Calculate Current Run Rate")
        
#         c1, c2 = st.columns(2)
#         with c1:
#             # --- v7.41: FIX for date_input crash ---
#             # Ensure default value is never less than min_value
#             default_start_date = max(min_data_date, max_data_date - timedelta(days=6))
            
#             analysis_start = st.date_input(
#                 "Analysis Start Date", 
#                 value=default_start_date, # Use safe default
#                 min_value=min_data_date, 
#                 max_value=max_data_date,
#                 key="forecast_start"
#             )
#         with c2:
#             analysis_end = st.date_input(
#                 "Analysis End Date", 
#                 value=max_data_date, 
#                 min_value=min_data_date, 
#                 max_value=max_data_date,
#                 key="forecast_end"
#             )
        
#         if analysis_start > analysis_end:
#             st.error("Analysis Start Date must be before End Date.")
#         else:
#             analysis_df = daily_summary_df.loc[analysis_start:analysis_end]
            
#             total_parts = analysis_df['Actual Output (parts)'].sum()
#             total_sec = analysis_df['Filtered Run Time (sec)'].sum()
            
#             parts_per_day = (total_parts / total_sec) * 86400 if total_sec > 0 else 0
            
#             st.metric(
#                 f"Calculated Run Rate (from {analysis_start} to {analysis_end})", 
#                 f"{parts_per_day:,.0f} Parts / Day"
#             )
#             st.caption(f"Based on {total_parts:,.0f} parts produced in {format_seconds_to_dhm(total_sec)} of run time.")

#             st.markdown("##### Input Future Demand Plan")
            
#             c1, c2 = st.columns([1, 2])
#             with c1:
#                 start_inventory = st.number_input("Current Inventory (Starting Stock)", value=0, min_value=0, step=100, key="daily_stock")
            
#             with c2:
#                 demand_sample = f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')},500\n{(datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')},500\n{(datetime.now() + timedelta(days=3)).strftime('%Y-%m-%d')},550"
#                 demand_input = st.text_area(
#                     "Paste Demand Data (Date,Demand)", 
#                     value=demand_sample, 
#                     height=150,
#                     help="Paste CSV data with two columns: `Date` and `Demand`. e.g., '2025-11-16,500'",
#                     key="daily_demand"
#                 )
            
#             if st.button("Run Daily Forecast"):
#                 if parts_per_day == 0:
#                     st.error("Cannot forecast with a zero production rate. Select a different analysis period.")
#                 elif not demand_input:
#                     st.error("Please paste demand data to run the forecast.")
#                 else:
#                     try:
#                         demand_data = io.StringIO(demand_input)
#                         df_demand = pd.read_csv(demand_data, header=None, names=['Date', 'Demand'])
#                         df_demand['Date'] = pd.to_datetime(df_demand['Date'])
#                         df_demand['Demand'] = pd.to_numeric(df_demand['Demand'])
                        
#                         # Create a full date range from the demand data
#                         forecast_start_date = df_demand['Date'].min()
#                         forecast_end_date = df_demand['Date'].max()
#                         all_forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
                        
#                         forecast_df = pd.DataFrame(all_forecast_dates, columns=['Date'])
                        
#                         # Merge demand data, filling non-demand days with 0
#                         forecast_df = pd.merge(forecast_df, df_demand, on='Date', how='left').fillna(0)
                        
#                         forecast_df['Projected Production'] = parts_per_day
#                         forecast_df['Cumulative Demand'] = forecast_df['Demand'].cumsum()
#                         forecast_df['Cumulative Production'] = forecast_df['Projected Production'].cumsum()
#                         forecast_df['Projected Inventory'] = start_inventory + forecast_df['Cumulative Production'] - forecast_df['Cumulative Demand']
                        
#                         # --- Calculate KPIs ---
#                         end_inventory = forecast_df['Projected Inventory'].iloc[-1]
#                         min_inventory = forecast_df['Projected Inventory'].min()
#                         shortfall_date = None
#                         if min_inventory < 0:
#                             shortfall_date = forecast_df[forecast_df['Projected Inventory'] < 0]['Date'].min()

#                         c1, c2, c3 = st.columns(3)
#                         with c1:
#                             st.metric(
#                                 "Projected End Inventory", 
#                                 f"{end_inventory:,.0f} parts",
#                                 delta=f"{end_inventory - start_inventory:,.0f}",
#                                 delta_color="normal"
#                             )
#                         with c2:
#                             st.metric(
#                                 "Projected Max Shortfall",
#                                 f"{min_inventory:,.0f} parts" if min_inventory < 0 else "No Shortfall",
#                                 delta_color="inverse"
#                             )
#                         with c3:
#                             if shortfall_date:
#                                 st.metric("Projected Shortfall Date", shortfall_date.strftime('%Y-%m-%d'))
#                             else:
#                                 st.metric("Projected Shortfall Date", "None")
                        
#                         # --- Plot Chart ---
#                         fig_forecast = go.Figure()

#                         # Add Cumulative Demand
#                         fig_forecast.add_trace(go.Scatter(
#                             x=forecast_df['Date'],
#                             y=forecast_df['Cumulative Demand'],
#                             name='Cumulative Demand',
#                             mode='lines',
#                             line=dict(color='red', width=2, dash='dot')
#                         ))
                        
#                         # Add Cumulative Production (with starting inventory)
#                         fig_forecast.add_trace(go.Scatter(
#                             x=forecast_df['Date'],
#                             y=forecast_df['Cumulative Production'] + start_inventory,
#                             name='Projected Cumulative Production (incl. stock)',
#                             mode='lines',
#                             line=dict(color='blue', width=2)
#                         ))
                        
#                         fig_forecast.update_layout(
#                             title="Projected Production vs. Demand",
#                             xaxis_title="Date",
#                             yaxis_title="Cumulative Parts",
#                             hovermode="x unified"
#                         )
#                         st.plotly_chart(fig_forecast, use_container_width=True)
                        
#                         # --- Show Data Table ---
#                         with st.expander("View Forecast Data Table"):
#                             st.dataframe(forecast_df.style.format({
#                                 'Demand': '{:,.0f}',
#                                 'Projected Production': '{:,.0f}',
#                                 'Cumulative Demand': '{:,.0f}',
#                                 'Cumulative Production': '{:,.0f}',
#                                 'Projected Inventory': '{:,.0f}'
#                             }), use_container_width=True)

#                     except Exception as e:
#                         st.error(f"Error parsing demand data: {e}")
#                         st.info("Please ensure your data is in the format 'YYYY-MM-DD,500' with one entry per line.")


# ==================================================================
#                       MAIN APP LOGIC (v7.45)
# ==================================================================

# --- Page Config ---
st.set_page_config(
    page_title=f"Capacity Risk Calculator (v{__version__})",
    layout="wide"
)

st.title("Capacity Risk Report")
st.markdown(f"**App Version:** `{__version__}` (RR-Downtime + CR-Inefficiency)")

# --- Sidebar for Inputs ---
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Upload Raw Data File (CSV or Excel)", type=["csv", "xlsx", "xls"])

st.sidebar.markdown("---")
st.sidebar.subheader("Run Rate Logic (for Downtime)")
st.sidebar.info("These settings define 'Downtime'.")

# --- v6.57: Restored Mode CT Tolerance slider ---
mode_ct_tolerance = st.sidebar.slider(
    "Mode CT Tolerance (%)", 0.01, 0.50, 0.25, 0.01,  
    help="Tolerance band (±) around the **Actual Mode CT**. Shots outside this band are flagged as 'Abnormal Cycle' (Downtime)."
)

# --- v6.61: Removed Approved CT Tolerance slider ---

rr_downtime_gap = st.sidebar.slider(
    "RR Downtime Gap (sec)", 0.0, 10.0, 2.0, 0.5, 
    help="Minimum idle time between shots to be considered a stop."
)

# --- v6.27: Add Run Interval Threshold ---
run_interval_hours = st.sidebar.slider(
    "Run Interval Threshold (hours)", 1.0, 24.0, 8.0, 0.5,
    help="Gaps between shots *longer* than this will be excluded from all calculations (e.g., weekends)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("CR Logic (for Inefficiency)")
st.sidebar.info("These settings define 'Inefficiency' during Uptime.")

data_frequency = st.sidebar.radio(
    "Select Graph Frequency",
    ['Daily', 'Weekly', 'Monthly', 'by Run'], # <-- v6.89: Added 'by Run'
    index=0,
    horizontal=True
)

toggle_filter = st.sidebar.toggle(
    "Remove Maintenance/Warehouse Shots",
    value=False, # Default OFF
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

benchmark_view = st.sidebar.radio(
    "Select Report Benchmark",
    ['Optimal Output', 'Target Output'],
    index=0, # Default to Optimal
    horizontal=False,
    help="Select the benchmark to compare against (e.g., 'Total Capacity Loss' vs 'Optimal' or 'Target')."
)

if benchmark_view == "Target Output":
    target_output_perc = st.sidebar.slider(
        "Target Output % (of Optimal)",
        min_value=0.0, max_value=100.0,
        value=90.0, # Default 90%
        step=1.0,
        format="%.0f%%",
        help="Sets the 'Target Output (parts)' goal as a percentage of 'Optimal Output (parts)'."
    )
else:
    # --- v6.64: Set to 100.0 for the function ---
    target_output_perc = 100.0 
    
st.sidebar.caption(f"App Version: **{__version__}**")


# --- Main Page Display ---
if uploaded_file is not None:

    df_raw = load_data(uploaded_file)

    if df_raw is not None:
        st.success(f"Successfully loaded file: **{uploaded_file.name}**")

        # --- v6.5: Removed CSS ---

        # --- Run Calculation ---
        with st.spinner("Calculating Capacity Risk... (Using new hybrid logic)"):
            
            # --- v6.64: Single Calculation Logic ---
            
            # 1. Always calculate vs. Optimal (100%)
            cache_key = f"{__version__}_{uploaded_file.name}_{target_output_perc}_{mode_ct_tolerance}_{rr_downtime_gap}_{run_interval_hours}"
            # --- v7.48: Renamed function call to bust cache ---
            results_df, all_shots_df = run_capacity_calculation_cached_v2(
                df_raw,
                toggle_filter,
                default_cavities,
                target_output_perc, # Pass the slider value
                mode_ct_tolerance,  
                rr_downtime_gap,        
                run_interval_hours,      
                _cache_version=cache_key
            )

            # --- v7.51: Removed the crash-loop block, and also the 1-based run_id increment.
            # all_shots_df['run_id'] is now 0-based (0, 1, 2...)

            if results_df is None or results_df.empty or all_shots_df.empty:
                st.error("No valid data found in file. Cannot proceed.")
            else:
                # --- End v6.64 ---
                
                # --- v7.42: CRITICAL REFACTOR ---
                # Move all calculations *outside* the tabs so both tabs can access the data
                
                # 1. Calculate all dataframes ONCE at the top.
                daily_summary_df = results_df.copy()
                run_summary_df = calculate_run_summaries(all_shots_df, target_output_perc)
                run_summary_df_for_total = run_summary_df.copy()
                
                # 2. Get All-Time totals (for both tabs)
                # --- v7.45: Create a totals_dict to pass to tabs ---
                all_time_totals = {}
                
                if run_summary_df_for_total.empty:
                    st.error("Failed to calculate 'by Run' summary for All-Time totals.")
                    # Set defaults to avoid crashing dashboard
                    all_time_totals = {
                        'total_produced': 0, 'total_downtime_loss_parts': 0,
                        'total_slow_loss_parts': 0, 'total_fast_gain_parts': 0,
                        'total_net_cycle_loss_parts': 0, 'total_optimal_100': 0,
                        'total_target': 0, 'total_downtime_loss_sec': 0,
                        'total_slow_loss_sec': 0, 'total_fast_gain_sec': 0,
                        'total_net_cycle_loss_sec': 0, 'run_time_sec_total': 0,
                        'run_time_dhm_total': "0m", 'total_actual_ct_sec': 0,
                        'total_actual_ct_dhm': "0m", 'total_true_net_loss_parts': 0,
                        'total_true_net_loss_sec': 0, 'total_calculated_net_loss_parts': 0,
                        'total_calculated_net_loss_sec': 0
                    }
                else:
                    # Sum the 'by Run' totals to get the All-Time totals
                    total_produced = run_summary_df_for_total['Actual Output (parts)'].sum()
                    total_downtime_loss_parts = run_summary_df_for_total['Capacity Loss (downtime) (parts)'].sum()
                    total_slow_loss_parts = run_summary_df_for_total['Capacity Loss (slow cycle time) (parts)'].sum()
                    total_fast_gain_parts = run_summary_df_for_total['Capacity Gain (fast cycle time) (parts)'].sum()
                    total_net_cycle_loss_parts = total_slow_loss_parts - total_fast_gain_parts
                    
                    total_optimal_100 = run_summary_df_for_total['Optimal Output (parts)'].sum()
                    total_target = run_summary_df_for_total['Target Output (parts)'].sum()
                    
                    total_downtime_loss_sec = run_summary_df_for_total['Capacity Loss (downtime) (sec)'].sum()
                    total_slow_loss_sec = run_summary_df_for_total['Capacity Loss (slow cycle time) (sec)'].sum()
                    total_fast_gain_sec = run_summary_df_for_total['Capacity Gain (fast cycle time) (sec)'].sum()
                    total_net_cycle_loss_sec = total_slow_loss_sec - total_fast_gain_sec
                    
                    run_time_sec_total = run_summary_df_for_total['Filtered Run Time (sec)'].sum()
                    run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)
                    
                    total_actual_ct_sec = run_summary_df_for_total['Actual Cycle Time Total (sec)'].sum()
                    total_actual_ct_dhm = format_seconds_to_dhm(total_actual_ct_sec)
                    
                    total_calculated_net_loss_parts = total_downtime_loss_parts + total_net_cycle_loss_parts
                    total_calculated_net_loss_sec = total_downtime_loss_sec + total_net_cycle_loss_sec
                    
                    total_true_net_loss_parts = total_optimal_100 - total_produced
                    total_true_net_loss_sec = total_calculated_net_loss_sec # v6.80 fix
                    
                    all_time_totals = {
                        'total_produced': total_produced, 'total_downtime_loss_parts': total_downtime_loss_parts,
                        'total_slow_loss_parts': total_slow_loss_parts, 'total_fast_gain_parts': total_fast_gain_parts,
                        'total_net_cycle_loss_parts': total_net_cycle_loss_parts, 'total_optimal_100': total_optimal_100,
                        'total_target': total_target, 'total_downtime_loss_sec': total_downtime_loss_sec,
                        'total_slow_loss_sec': total_slow_loss_sec, 'total_fast_gain_sec': total_fast_gain_sec,
                        'total_net_cycle_loss_sec': total_net_cycle_loss_sec, 'run_time_sec_total': run_time_sec_total,
                        'run_time_dhm_total': run_time_dhm_total, 'total_actual_ct_sec': total_actual_ct_sec,
                        'total_actual_ct_dhm': total_actual_ct_dhm, 'total_true_net_loss_parts': total_true_net_loss_parts,
                        'total_true_net_loss_sec': total_true_net_loss_sec, 'total_calculated_net_loss_parts': total_calculated_net_loss_parts,
                        'total_calculated_net_loss_sec': total_calculated_net_loss_sec
                    }
                
                # --- v7.52: Commented out tabs 2 & 3 ---
                # tab1, tab2, tab3 = st.tabs(["Capacity Risk Report", "Automated Risk Analysis", "Demand & Capacity Planning"])
                tab1, = st.tabs(["Capacity Risk Report"]) # Show only one tab

                with tab1:
                    # --- This is the original, detailed report ---
                    st.header("All-Time Summary")
                    
                    run_time_label = "Overall Run Time" if not toggle_filter else "Filtered Run Time"
                    actual_output_perc_val = (all_time_totals['total_produced'] / all_time_totals['total_optimal_100']) if all_time_totals['total_optimal_100'] > 0 else 0
                    benchmark_title = "Optimal Output"

                    # --- Box 1: Overall Summary ---
                    st.subheader(f"Overall Summary")
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns(4)
                        
                        with c1:
                            st.metric(run_time_label, all_time_totals['run_time_dhm_total'])
                        
                        with c2:
                            if benchmark_view == "Target Output":
                                st.metric(f"Target Output ({target_output_perc:.0f}%)", f"{all_time_totals['total_target']:,.0f}")
                                st.caption(f"Optimal (100%): {all_time_totals['total_optimal_100']:,.0f}")
                            else:
                                st.metric("Optimal Output (100%)", f"{all_time_totals['total_optimal_100']:,.0f}")
                        
                        with c3:
                            st.metric(f"Actual Output ({actual_output_perc_val:.1%})", f"{all_time_totals['total_produced']:,.0f} parts")
                            st.caption(f"Actual Production Time: {all_time_totals['total_actual_ct_dhm']}")
                            
                            if benchmark_view == "Target Output":
                                gap_to_target = all_time_totals['total_produced'] - all_time_totals['total_target']
                                gap_perc = (gap_to_target / all_time_totals['total_target']) if all_time_totals['total_target'] > 0 else 0
                                gap_color = "green" if gap_to_target > 0 else "red"
                                st.caption(f"Gap to Target: <span style='color:{gap_color};'>{gap_to_target:+,.0f} ({gap_perc:+.1%})</span>", unsafe_allow_html=True)
                        
                        with c4:
                            if benchmark_view == "Target Output":
                                total_loss_vs_target_parts = np.maximum(0, all_time_totals['total_target'] - all_time_totals['total_produced'])
                                total_loss_vs_target_sec = run_summary_df_for_total['Capacity Loss (vs Target) (sec)'].sum()
                                
                                st.markdown(f"**Capacity Loss (vs Target)**")
                                st.markdown(f"<h3><span style='color:red;'>{total_loss_vs_target_parts:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                                st.caption(f"Total Time Lost vs Target: {format_seconds_to_dhm(total_loss_vs_target_sec)}")
                            else:
                                st.markdown(f"**Total Capacity Loss (True)**")
                                st.markdown(f"<h3><span style='color:red;'>{all_time_totals['total_true_net_loss_parts']:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                                st.caption(f"Total Time Lost: {format_seconds_to_dhm(all_time_totals['total_true_net_loss_sec'])}")
                                
                    # --- Waterfall Chart Layout ---
                    st.subheader(f"Capacity Loss Breakdown (vs {benchmark_title})")
                    st.info(f"These values are calculated based on the *time-based* logic (Downtime + Slow/Fast Cycles) using **{benchmark_title}** as the benchmark.")
                    
                    c1, c2 = st.columns([1, 1])

                    with c1:
                        st.markdown("<h6 style='text-align: center;'>Overall Performance Breakdown</h6>", unsafe_allow_html=True)
                        
                        waterfall_x = [f"<b>Optimal Output (100%)</b>", "Loss (RR Downtime)"]
                        waterfall_y = [all_time_totals['total_optimal_100'], -all_time_totals['total_downtime_loss_parts']]
                        waterfall_measure = ["absolute", "relative"]
                        waterfall_text = [f"{all_time_totals['total_optimal_100']:,.0f}", f"{-all_time_totals['total_downtime_loss_parts']:,.0f}"]

                        if all_time_totals['total_net_cycle_loss_parts'] >= 0:
                            waterfall_x.append("Net Loss (Cycle Time)")
                            waterfall_y.append(-all_time_totals['total_net_cycle_loss_parts'])
                            waterfall_measure.append("relative")
                            waterfall_text.append(f"{-all_time_totals['total_net_cycle_loss_parts']:,.0f}")
                        else:
                            waterfall_x.append("Net Gain (Cycle Time)")
                            waterfall_y.append(abs(all_time_totals['total_net_cycle_loss_parts']))
                            waterfall_measure.append("relative")
                            waterfall_text.append(f"{abs(all_time_totals['total_net_cycle_loss_parts']):+,.0f}")
                        
                        waterfall_x.append("<b>Actual Output</b>")
                        waterfall_y.append(all_time_totals['total_produced'])
                        waterfall_measure.append("total")
                        waterfall_text.append(f"{all_time_totals['total_produced']:,.0f}")
                        
                        fig_waterfall = go.Figure(go.Waterfall(
                            name = "Breakdown", orientation = "v", measure = waterfall_measure,
                            x = waterfall_x, y = waterfall_y, text = waterfall_text,
                            textposition = "outside", connector = {"line":{"color":"rgb(63, 63, 63)"}},
                            increasing = {"marker":{"color":"#2ca02c"}}, decreasing = {"marker":{"color":"#ff6961"}},
                            totals = {"marker":{"color":"#1f77b4"}}
                        ))
                        
                        fig_waterfall.update_layout(
                            showlegend=False, margin=dict(t=0, b=0, l=0, r=0),
                            height=400, yaxis_title='Parts'
                        )
                        
                        if benchmark_view == "Target Output":
                            fig_waterfall.add_shape(
                                type='line', x0=-0.5, x1=len(waterfall_x)-0.5,
                                y0=all_time_totals['total_target'], y1=all_time_totals['total_target'],
                                line=dict(color='deepskyblue', dash='dash', width=2)
                            )
                            fig_waterfall.add_annotation(
                                x=0, y=all_time_totals['total_target'], text=f"Target: {all_time_totals['total_target']:,.0f}",
                                showarrow=True, arrowhead=1, ax=-40, ay=-20
                            )
                            fig_waterfall.add_annotation(
                                x=len(waterfall_x)-0.5, y=all_time_totals['total_optimal_100'],
                                text=f"Optimal (100%): {all_time_totals['total_optimal_100']:,.0f}",
                                showarrow=True, arrowhead=1, ax=40, ay=-20
                            )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
                        

                    with c2:
                        def get_color_css(val):
                            if val > 0: return "color: red;"
                            if val < 0: return "color: green;"
                            return "color: black;"

                        net_loss_val = all_time_totals['total_calculated_net_loss_parts']
                        net_loss_color = get_color_css(net_loss_val)
                        with st.container(border=True):
                            st.markdown(f"**Total Net Impact**")
                            st.markdown(f"<h3><span style='{net_loss_color}'>{net_loss_val:,.0f} parts</span></h3>", unsafe_allow_html=True)
                            st.caption(f"Net Time Lost: {format_seconds_to_dhm(all_time_totals['total_calculated_net_loss_sec'])}")
                        
                        table_data = {
                            "Metric": [
                                "Loss (RR Downtime)", 
                                "Net Loss (Cycle Time)", 
                                "\u00A0\u00A0\u00A0 └ Loss (Slow Cycles)", 
                                "\u00A0\u00A0\u00A0 └ Gain (Fast Cycles)"
                            ],
                            "Parts": [
                                all_time_totals['total_downtime_loss_parts'],
                                all_time_totals['total_net_cycle_loss_parts'],
                                all_time_totals['total_slow_loss_parts'],
                                all_time_totals['total_fast_gain_parts']
                            ],
                            "Time": [
                                format_seconds_to_dhm(all_time_totals['total_downtime_loss_sec']),
                                format_seconds_to_dhm(all_time_totals['total_net_cycle_loss_sec']),
                                format_seconds_to_dhm(all_time_totals['total_slow_loss_sec']),
                                format_seconds_to_dhm(all_time_totals['total_fast_gain_sec'])
                            ]
                        }
                        df_table = pd.DataFrame(table_data)

                        def style_parts_col(val, row_index):
                            if row_index == 0: color_style = get_color_css(val)
                            elif row_index == 1: color_style = get_color_css(val)
                            elif row_index == 2: color_style = get_color_css(val)
                            elif row_index == 3: color_style = get_color_css(val * -1)
                            else: color_style = "color: black;"
                            return color_style

                        styled_df = df_table.style.apply(
                            lambda row: [style_parts_col(row['Parts'], row.name) if col == 'Parts' else '' for col in row.index],
                            axis=1
                        ).format(
                            {"Parts": "{:,.0f}"}
                        ).set_properties(
                            **{'text-align': 'left'}, subset=['Metric', 'Time']
                        ).set_properties(
                            **{'text-align': 'right'}, subset=['Parts']
                        ).hide(axis='index')
                        
                        st.dataframe(styled_df, use_container_width=True)

                    # --- Collapsible Daily Summary Table ---
                    with st.expander("View Daily Summary Data"):
                        
                        daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                        daily_summary_df['Actual Output (parts %)'] = np.where( results_df['Optimal Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / results_df['Optimal Output (parts)'], 0 )
                        perc_base_parts = daily_summary_df['Optimal Output (parts)']
                        perc_base_sec = daily_summary_df['Filtered Run Time (sec)']
                        daily_summary_df['Total Capacity Loss (time %)'] = np.where( perc_base_sec > 0, daily_summary_df['Total Capacity Loss (sec)'] / perc_base_sec, 0 )
                        daily_summary_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, daily_summary_df['Total Capacity Loss (parts)'] / perc_base_parts, 0 )
                        daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)
                        daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( daily_summary_df['Target Output (parts)'] > 0, daily_summary_df['Capacity Loss (vs Target) (parts)'] / daily_summary_df['Target Output (parts)'], 0 )
                        daily_summary_df['Capacity Loss (vs Target) (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Capacity Loss (vs Target) (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                        daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = daily_summary_df['Capacity Loss (vs Target) (sec)'].apply(format_seconds_to_dhm)
                        daily_summary_df['Filtered Run Time (d/h/m)'] = daily_summary_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                        daily_summary_df['Actual Cycle Time Total (d/h/m)'] = daily_summary_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                        daily_kpi_table = pd.DataFrame(index=daily_summary_df.index)
                        daily_kpi_table[run_time_label] = daily_summary_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                        daily_kpi_table['Actual Production Time'] = daily_summary_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)
                        daily_kpi_table['Actual Output (parts)'] = daily_summary_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (parts %)']:.1%})", axis=1)

                        if benchmark_view == "Optimal Output":
                            daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                            daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                            st.dataframe(daily_kpi_table, use_container_width=True)
                        else: # Target Output
                            daily_summary_df['Gap to Target (parts)'] = pd.to_numeric(daily_summary_df['Gap to Target (parts)'], errors='coerce').fillna(0)
                            daily_kpi_table['Gap to Target (parts)'] = daily_summary_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                            daily_kpi_table['Capacity Loss (vs Target) (Time)'] = daily_summary_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (d/h/m)']} ({r['Capacity Loss (vs Target) (time %)']:.1%})", axis=1)
                            st.dataframe(daily_kpi_table.style.applymap(
                                lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                                subset=['Gap to Target (parts)']
                            ), use_container_width=True)

                    st.divider()

                    # --- 3. AGGREGATED REPORT (Chart & Table) ---
                    
                    if data_frequency == 'by Run':
                        agg_df = run_summary_df.copy()
                        chart_title_prefix = "Run-by-Run"
                    elif data_frequency == 'Weekly':
                        agg_df = daily_summary_df.resample('W').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Weekly"
                    elif data_frequency == 'Monthly':
                        agg_df = daily_summary_df.resample('ME').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Monthly"
                    else: # Daily
                        agg_df = daily_summary_df.copy()
                        chart_title_prefix = "Daily"
                    
                    display_df = agg_df
                    
                    if display_df.empty:
                        st.warning(f"No data to display for the '{data_frequency}' frequency.")
                    else:
                        display_df['Capacity Loss (vs Target) (parts)'] = np.maximum(0, -display_df['Gap to Target (parts)'])
                        perc_base_parts = display_df['Optimal Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Optimal)"
                        optimal_100_base = display_df['Optimal Output (parts)']
                        display_df['Actual Output (%)'] = np.where( optimal_100_base > 0, display_df['Actual Output (parts)'] / optimal_100_base, 0)
                        display_df['Production Shots (%)'] = np.where( display_df['Total Shots (all)'] > 0, display_df['Production Shots'] / display_df['Total Shots (all)'], 0)
                        display_df['Actual Cycle Time Total (time %)'] = np.where( display_df['Filtered Run Time (sec)'] > 0, display_df['Actual Cycle Time Total (sec)'] / display_df['Filtered Run Time (sec)'], 0)
                        display_df['Capacity Loss (downtime) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Loss (downtime) (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Loss (slow cycle time) (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( perc_base_parts > 0, display_df['Capacity Gain (fast cycle time) (parts)'] / perc_base_parts, 0)
                        display_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, display_df['Total Capacity Loss (parts)'] / perc_base_parts, 0)
                        display_df['Capacity Loss (vs Target) (parts %)'] = np.where( display_df['Target Output (parts)'] > 0, display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0)
                        display_df['Total Capacity Loss (cycle time) (parts)'] = display_df['Capacity Loss (slow cycle time) (parts)'] - display_df['Capacity Gain (fast cycle time) (parts)']
                        display_df['(Ref) Net Loss (RR)'] = display_df['Capacity Loss (downtime) (parts)']
                        display_df['(Ref) Net Loss (Slow)'] = display_df['Capacity Loss (slow cycle time) (parts)']
                        display_df['(Ref) Net Gain (Fast)'] = display_df['Capacity Gain (fast cycle time) (parts)']
                        display_df['(Ref) Total Net Loss'] = display_df['(Ref) Net Loss (RR)'] + display_df['(Ref) Net Loss (Slow)'] - display_df['(Ref) Net Gain (Fast)']
                        display_df['loss_downtime_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, display_df['(Ref) Net Loss (RR)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['loss_slow_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, display_df['(Ref) Net Loss (Slow)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['gain_fast_ratio'] = np.where(display_df['(Ref) Total Net Loss'] != 0, -display_df['(Ref) Net Gain (Fast)'] / display_df['(Ref) Total Net Loss'], 0)
                        display_df['Allocated Loss (RR Downtime)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['loss_downtime_ratio']
                        display_df['Allocated Loss (Slow Cycles)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['loss_slow_ratio']
                        display_df['Allocated Gain (Fast Cycles)'] = display_df['Capacity Loss (vs Target) (parts)'] * display_df['gain_fast_ratio']
                        display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                        display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                        
                        if 'Start Time' in display_df.columns:
                            display_df['Start Time_str'] = pd.to_datetime(display_df['Start Time']).dt.strftime('%Y-%m-%d %H:%M')
                            
                        if data_frequency == 'Weekly': xaxis_title = "Week"
                        elif data_frequency == 'Monthly': xaxis_title = "Month"
                        # --- v7.53: Fix SyntaxError by changing 'elif' to 'if' ---
                        if data_frequency == 'by Run': xaxis_title = "Run ID"
                        else: xaxis_title = "Date"
                        
                        if data_frequency == 'by Run':
                            chart_df = display_df.reset_index().rename(columns={'run_id': 'X-Axis'})
                            # --- v7.51: Add 1 to run_id for display ---
                            chart_df['X-Axis'] = 'Run ' + (chart_df['X-Axis'] + 1).astype(str)
                        else:
                            chart_df = display_df.reset_index().rename(columns={'Date': 'X-Axis'})
                        

                        # --- Unified Performance Breakdown Chart ---
                        st.header(f"{data_frequency} Performance Breakdown (vs {benchmark_title})")
                        fig_ts = go.Figure()
                        
                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Actual Output (parts)'], name='Actual Output',
                            marker_color='#3498DB', customdata=chart_df['Actual Output (%)'],
                            hovertemplate='Actual Output: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                        ))
                        
                        chart_df['Net Cycle Time Loss (parts)'] = chart_df['Total Capacity Loss (cycle time) (parts)']
                        chart_df['Net Cycle Time Loss (positive)'] = np.maximum(0, chart_df['Net Cycle Time Loss (parts)'])

                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Net Cycle Time Loss (positive)'], name='Capacity Loss (cycle time)',
                            marker_color='#ffb347',
                            customdata=np.stack((
                                chart_df['Net Cycle Time Loss (parts)'],
                                chart_df['Capacity Loss (slow cycle time) (parts)'],
                                chart_df['Capacity Gain (fast cycle time) (parts)']
                            ), axis=-1),
                            hovertemplate=
                                '<b>Net Cycle Time Loss: %{customdata[0]:,.0f}</b><br>' +
                                'Slow Cycle Loss: %{customdata[1]:,.0f}<br>' +
                                'Fast Cycle Gain: -%{customdata[2]:,.0f}<br>' + 
                                '<extra></extra>'
                        ))
                        
                        fig_ts.add_trace(go.Bar(
                            x=chart_df['X-Axis'], y=chart_df['Capacity Loss (downtime) (parts)'], name='Run Rate Downtime (Stops)',
                            marker_color='#808080', customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                            hovertemplate='Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                        ))
                        
                        fig_ts.update_layout(barmode='stack')

                        if benchmark_view == "Target Output":
                            fig_ts.add_trace(go.Scatter(
                                x=chart_df['X-Axis'], y=chart_df['Target Output (parts)'],
                                name=f'Target Output ({target_output_perc:.0f}%)', mode='lines',
                                line=dict(color='deepskyblue', dash='dash'),
                                hovertemplate=f'<b>Target Output ({target_output_perc:.0f}%)</b>: %{{y:,.0f}}<extra></extra>'
                            ))
                            
                        fig_ts.add_trace(go.Scatter(
                            x=chart_df['X-Axis'], y=chart_df['Optimal Output (parts)'],
                            name='Optimal Output (100%)', mode='lines',
                            line=dict(color='darkblue', dash='dot'),
                            hovertemplate='Optimal Output (100%): %{y:,.0f}<extra></extra>'
                        ))

                        fig_ts.update_layout(
                            title=chart_title, xaxis_title=xaxis_title, yaxis_title='Parts (Output & Loss)',
                            legend_title='Metric', hovermode="x unified"
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)

                        # --- Full Data Table ---
                        display_df_totals = display_df
                        
                        st.header(f"Production Totals Report ({data_frequency})")
                        if data_frequency == 'by Run':
                            report_table_1_df = display_df_totals.reset_index().rename(columns={'run_id': 'Run ID'})
                            # --- v7.51: Add 1 to run_id for display ---
                            report_table_1_df['Run ID'] = report_table_1_df['Run ID'] + 1
                            report_table_1_df['Total Downtime (sec)'] = report_table_1_df['Filtered Run Time (sec)'] - report_table_1_df['Actual Cycle Time Total (sec)']
                            report_table_1_df['Total Downtime (d/h/m)'] = report_table_1_df['Total Downtime (sec)'].apply(format_seconds_to_dhm)
                            report_table_1 = pd.DataFrame(index=report_table_1_df.index)
                            report_table_1['Run ID'] = report_table_1_df['Run ID']
                            report_table_1['Start Time'] = report_table_1_df['Start Time_str']
                            report_table_1['Overall Run Time'] = report_table_1_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']}", axis=1)
                            report_table_1['Actual Production Time'] = report_table_1_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']}", axis=1)
                            report_table_1['Total Downtime'] = report_table_1_df.apply(lambda r: f"{r['Total Downtime (d/h/m)']}", axis=1)
                            report_table_1['Total Shots'] = report_table_1_df['Total Shots (all)'].map('{:,.0f}'.format)
                            report_table_1['Production Shots'] = report_table_1_df['Production Shots'].map('{:,.0f}'.format)
                            report_table_1['Downtime Shots'] = report_table_1_df['Downtime Shots'].map('{:,.0f}'.format)
                            report_table_1['Mode CT'] = report_table_1_df['Mode CT'].map('{:.2f}s'.format)
                        else: # Daily, Weekly, Monthly
                            report_table_1 = pd.DataFrame(index=display_df_totals.index)
                            report_table_1_df = display_df_totals
                            report_table_1['Total Shots (all)'] = report_table_1_df['Total Shots (all)'].map('{:,.0f}'.format)
                            report_table_1['Production Shots'] = report_table_1_df.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                            report_table_1['Downtime Shots'] = report_table_1_df['Downtime Shots'].map('{:,.0f}'.format)
                            report_table_1[run_time_label] = report_table_1_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                            report_table_1['Actual Production Time'] = report_table_1_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                        st.dataframe(report_table_1, use_container_width=True)

                        # --- Conditional Tables ---
                        st.header(f"Capacity Loss & Gain Report (vs Optimal) ({data_frequency})")
                        display_df_optimal = display_df

                        if data_frequency == 'by Run':
                            report_table_optimal_df = display_df_optimal.reset_index().rename(columns={'run_id': 'Run ID'})
                            # --- v7.51: Add 1 to run_id for display ---
                            report_table_optimal_df['Run ID'] = report_table_optimal_df['Run ID'] + 1
                            report_table_optimal = pd.DataFrame(index=report_table_optimal_df.index)
                            report_table_optimal['Run ID'] = report_table_optimal_df['Run ID']
                        else:
                            report_table_optimal = pd.DataFrame(index=display_df_optimal.index)
                            report_table_optimal_df = display_df_optimal

                        report_table_optimal['Optimal Output (parts)'] = report_table_optimal_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                        report_table_optimal['Actual Output (parts)'] = report_table_optimal_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                        report_table_optimal['Loss (RR Downtime)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                        report_table_optimal['Loss (Slow Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                        report_table_optimal['Gain (Fast Cycles)'] = report_table_optimal_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                        # --- v7.47: FIX IndentationError ---
                        report_table_optimal['Total Net Loss'] = report_table_optimal_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                        
                        def style_loss_gain_table(col):
                            col_name = col.name
                            if col_name == 'Loss (RR Downtime)': return ['color: red'] * len(col)
                            if col_name == 'Loss (Slow Cycles)': return ['color: red'] * len(col)
                            if col_name == 'Gain (Fast Cycles)': return ['color: green'] * len(col)
                            if col_name == 'Total Net Loss':
                                # Style based on the raw numeric value from the underlying dataframe
                                return ['color: green' if v < 0 else 'color: red' for v in display_df_optimal['Total Capacity Loss (parts)']]
                            return [''] * len(col)

                        st.dataframe(
                            report_table_optimal.style.apply(style_loss_gain_table, axis=0),
                            use_container_width=True
                        )
                        
                        if benchmark_view == "Target Output": 
                            st.header(f"Target Report (90%) ({data_frequency})")
                            st.info("This table allocates your Capacity Loss (vs Target) based on the proportional impact of all your true losses and gains (Downtime, Slow Cycles, and Fast Cycles).")
                            
                            display_df_target = display_df
                            
                            if data_frequency == 'by Run':
                                report_table_target_df = display_df_target.reset_index().rename(columns={'run_id': 'Run ID'})
                                # --- v7.51: Add 1 to run_id for display ---
                                report_table_target_df['Run ID'] = report_table_target_df['Run ID'] + 1
                                report_table_target = pd.DataFrame(index=report_table_target_df.index)
                                report_table_target['Run ID'] = report_table_target_df['Run ID']
                            else:
                                report_table_target = pd.DataFrame(index=display_df_target.index)
                                report_table_target_df = display_df_target

                            report_table_target['Target Output (parts)'] = report_table_target_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f}", axis=1)
                            report_table_target['Actual Output (parts)'] = report_table_target_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                            report_table_target['Actual % (vs Target)'] = report_table_target_df.apply(lambda r: r['Actual Output (parts)'] / r['Target Output (parts)'] if r['Target Output (parts)'] > 0 else 0, axis=1).apply(lambda x: "{:.1%}".format(x) if pd.notna(x) else "N/A")
                            report_table_target['Net Gap to Target (parts)'] = report_table_target_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                            report_table_target['Capacity Loss (vs Target)'] = report_table_target_df['Capacity Loss (vs Target) (parts)'].apply(lambda x: "{:,.2f}".format(x) if pd.notna(x) else "N/A")
                            
                            report_table_target['Allocated Loss (RR Downtime)'] = report_table_target_df.apply(
                                lambda r: f"{r['Allocated Loss (RR Downtime)']:,.2f} ({r['loss_downtime_ratio']:.1%})", 
                                axis=1
                            )
                            report_table_target['Allocated Loss (Slow Cycles)'] = report_table_target_df.apply(
                                lambda r: f"{r['Allocated Loss (Slow Cycles)']:,.2f} ({r['loss_slow_ratio']:.1%})", 
                                axis=1
                            )
                            report_table_target['Allocated Gain (Fast Cycles)'] = report_table_target_df.apply(
                                lambda r: f"{r['Allocated Gain (Fast Cycles)']:,.2f} ({r['gain_fast_ratio']:.1%})", 
                                axis=1
                            )

                            def style_target_report_table(col):
                                col_name = col.name
                                if col_name == 'Net Gap to Target (parts)':
                                    return ['color: green' if v > 0 else 'color: red' for v in display_df_target['Gap to Target (parts)']]
                                if col_name == 'Actual % (vs Target)':
                                    return ['color: green' if v > 1 else 'color: red' for v in (display_df_target['Actual Output (parts)'] / display_df_target['Target Output (parts)']).fillna(0)]
                                if col_name == 'Capacity Loss (vs Target)': return ['color: red'] * len(col)
                                if col_name == 'Allocated Loss (RR Downtime)': return ['color: red'] * len(col)
                                if col_name == 'Allocated Loss (Slow Cycles)': return ['color: red'] * len(col)
                                if col_name == 'Allocated Gain (Fast Cycles)': return ['color: green'] * len(col)
                                return [''] * len(col)
                            
                            st.dataframe(
                                report_table_target.style.apply(style_target_report_table, axis=0),
                                use_container_width=True
                            )

                    # --- 4. SHOT-BY-SHOT ANALYSIS ---
                    st.divider()
                    st.header("Shot-by-Shot Analysis (All Shots)")
                    st.info(f"This chart shows all shots. 'Production' shots are color-coded based on the **Optimal Output (Approved CT)** benchmark. 'RR Downtime (Stop)' shots are grey.")

                    if all_shots_df.empty:
                        st.warning("No shots were found in the file to analyze.")
                    else:
                        available_dates_list = sorted(all_shots_df['date'].unique(), reverse=True)
                        available_dates = ["All Dates"] + available_dates_list
                        
                        if not available_dates_list:
                            st.warning("No valid dates found in shot data.")
                        else:
                            selected_date = st.selectbox(
                                "Select a Date to Analyze",
                                options=available_dates,
                                format_func=lambda d: "All Dates" if isinstance(d, str) else d.strftime('%Y-%m-%d')
                            )
                            
                            if selected_date == "All Dates":
                                df_day_shots = all_shots_df.copy()
                                chart_title = "All Shots for Full Period"
                            else:
                                df_day_shots = all_shots_df[all_shots_df['date'] == selected_date]
                                chart_title = f"All Shots for {selected_date}"
                            
                            st.subheader("Chart Controls")
                            non_break_df = df_day_shots[df_day_shots['Shot Type'] != 'Run Break (Excluded)']
                            max_ct_for_day = 100
                            if not non_break_df.empty:
                                max_ct_for_day = non_break_df['Actual CT'].max()

                            slider_max = int(np.ceil(max_ct_for_day / 10.0)) * 10
                            slider_max = max(slider_max, 50)
                            slider_max = min(slider_max, 1000)

                            y_axis_max = st.slider(
                                "Zoom Y-Axis (sec)",
                                min_value=10, max_value=1000,
                                value=min(slider_max, 200), step=10,
                                help="Adjust the max Y-axis to zoom in on the cluster. (Set to 1000 to see all outliers)."
                            )

                            required_shot_cols = ['reference_ct', 'Mode CT Lower', 'Mode CT Upper', 'run_id', 'mode_ct', 'rr_time_diff', 'adj_ct_sec']
                            missing_shot_cols = [col for col in required_shot_cols if col not in df_day_shots.columns]
                            
                            if missing_shot_cols:
                                st.error(f"Error: Missing required columns. {', '.join(missing_shot_cols)}")
                            elif df_day_shots.empty:
                                st.warning(f"No shots found for {selected_date}.")
                            else:
                                reference_ct_for_day = df_day_shots['reference_ct'].iloc[0] 
                                reference_ct_label = "Approved CT"
                                
                                fig_ct = go.Figure()
                                color_map = {
                                    'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#3498DB',
                                    'RR Downtime (Stop)': '#808080', 'Run Break (Excluded)': '#d3d3d3'
                                }

                                for shot_type, color in color_map.items():
                                    df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                                    if not df_subset.empty:
                                        fig_ct.add_bar(
                                            x=df_subset['SHOT TIME'], y=df_subset['Actual CT'],
                                            name=shot_type, marker_color=color,
                                            # --- v7.51: Add 1 to run_id for display ---
                                            customdata=(df_subset['run_id'] + 1),
                                            hovertemplate='<b>%{x|%H:%M:%S}</b><br>Run ID: %{customdata}<br>Shot Type: %{fullData.name}<br>Actual CT: %{y:.2f}s<extra></extra>'
                                        )
                                
                                for run_id, df_run in df_day_shots.groupby('run_id'):
                                    if not df_run.empty:
                                        mode_ct_lower_for_run = df_run['Mode CT Lower'].iloc[0]
                                        mode_ct_upper_for_run = df_run['Mode CT Upper'].iloc[0]
                                        run_start_time = df_run['SHOT TIME'].min()
                                        run_end_time = df_run['SHOT TIME'].max()
                                        
                                        fig_ct.add_hrect(
                                            x0=run_start_time, x1=run_end_time,
                                            y0=mode_ct_lower_for_run, y1=mode_ct_upper_for_run,
                                            fillcolor="grey", opacity=0.20,
                                            line_width=0,
                                            # --- v7.51: Add 1 to run_id for display ---
                                            name=f"Run {run_id + 1} Mode Band" if len(df_day_shots['run_id'].unique()) > 1 else "Mode CT Band"
                                        )
                                
                                legend_names_seen = set()
                                for trace in fig_ct.data:
                                    if "Mode Band" in trace.name:
                                        if trace.name in legend_names_seen:
                                            trace.showlegend = False
                                        else:
                                            legend_names_seen.add(trace.name)
                                
                                fig_ct.add_shape(
                                    type='line',
                                    x0=df_day_shots['SHOT TIME'].min(), x1=df_day_shots['SHOT TIME'].max(),
                                    y0=reference_ct_for_day, y1=reference_ct_for_day,
                                    line=dict(color='green', dash='dash'), name=f'{reference_ct_label} ({reference_ct_for_day:.2f}s)'
                                )
                                fig_ct.add_annotation(
                                    x=df_day_shots['SHOT TIME'].max(), y=reference_ct_for_day,
                                    text=f"{reference_ct_label}: {reference_ct_for_day:.2f}s", showarrow=True, arrowhead=1
                                )
                                
                                if 'run_id' in df_day_shots.columns:
                                    run_starts = df_day_shots.groupby('run_id')['SHOT TIME'].min().sort_values()
                                    for start_time in run_starts.iloc[1:]:
                                        run_id_val = df_day_shots[df_day_shots['SHOT TIME'] == start_time]['run_id'].iloc[0]
                                        fig_ct.add_vline(
                                            x=start_time, line_width=2, 
                                            line_dash="dash", line_color="purple"
                                        )
                                        fig_ct.add_annotation(
                                            x=start_time, y=y_axis_max * 0.95,
                                            # --- v7.50: FIX for graph label ---
                                            # --- v7.51: Re-apply fix: run_id is 0-based, so add 1
                                            text=f"Run {run_id_val + 1} Start",
                                            showarrow=False, yshift=10, textangle=-90
                                        )

                                fig_ct.update_layout(
                                    title=chart_title, xaxis_title='Time of Day',
                                    yaxis_title='Actual Cycle Time (sec)',
                                    hovermode="closest", yaxis_range=[0, y_axis_max],
                                )
                                st.plotly_chart(fig_ct, use_container_width=True)

                                selected_date_str = "All Dates" if isinstance(selected_date, str) else selected_date.strftime('%Y-%m-%d')
                                st.subheader(f"Data for all {len(df_day_shots)} shots ({selected_date_str})")
                                
                                if len(df_day_shots) > 10000:
                                    st.info(f"Displaying first 10,000 shots of {len(df_day_shots)} total.")
                                    df_to_display = df_day_shots.head(10000)
                                else:
                                    df_to_display = df_day_shots
                                    
                                st.dataframe(
                                    df_to_display[[
                                        'SHOT TIME', 'Actual CT', 'Approved CT',
                                        'Working Cavities', 'run_id', 'mode_ct', 
                                        'Shot Type', 'stop_flag',
                                        'rr_time_diff', 'adj_ct_sec',
                                        'reference_ct', 'Mode CT Lower', 'Mode CT Upper'
                                    ]].style.format({
                                        'Actual CT': '{:.2f}',
                                        'Approved CT': '{:.1f}',
                                        'reference_ct': '{:.2f}', 
                                        'Mode CT Lower': '{:.2f}',
                                        'Mode CT Upper': '{:.2f}',
                                        'mode_ct': '{:.2f}',
                                        'rr_time_diff': '{:.1f}s',
                                        'adj_ct_sec': '{:.1f}s',
                                        'SHOT TIME': lambda t: t.strftime('%Y-%m-%d %H:%M:%S') if selected_date == "All Dates" else t.strftime('%H:%M:%S')
                                    }),
                                    use_container_width=True
                                )

                # --- v7.52: Commented out tabs 2 & 3 ---
                # with tab2:
                #     # --- v7.45: Call new Tab 2 render function ---
                #     render_automated_risk_tab(run_summary_df_for_total, all_time_totals)

                # with tab3:
                #     # --- v7.45: Call new Tab 3 render function ---
                #     render_demand_planning_tab(daily_summary_df, all_shots_df, run_summary_df_for_total)


else:
    st.info("👈 Please upload a data file to begin.")