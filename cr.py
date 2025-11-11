import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
__version__ = "6.54 (Restored Target CT logic, fixed core bug)"
# ==================================================================

# ==================================================================
#                         HELPER FUNCTIONS
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

# --- v6.5: Removed get_progress_bar_html ---

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
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc, mode_ct_tolerance, approved_ct_tolerance, rr_downtime_gap, run_interval_hours):
    """
    Core function to process the raw DataFrame and calculate all Capacity Risk fields
    using the new hybrid RR (downtime) + CR (inefficiency) logic.
    
    All losses/gains are calculated relative to the benchmark set by 'target_output_perc'.
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

    # --- 5. Group by Day ---
    df_production_only['date'] = df_production_only['SHOT TIME'].dt.date

    daily_results_list = []
    all_shots_list = []

    # --- Define all columns that will be created for the daily results ---
    # This prevents KeyErrors if all days are skipped
    all_result_columns = [
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

    # Iterate over each day's data
    for date, daily_df in df_production_only.groupby('date'):

        results = {col: 0 for col in all_result_columns} # Pre-fill all with 0
        results['Date'] = date

        if daily_df.empty or len(daily_df) < 2:
            results['Total Shots (all)'] = len(daily_df)
            daily_results_list.append(results) # Append the zero-filled dict
            continue
            
        # --- 6. Get Wall Clock Time (Basis for Segment 4) ---
        first_shot_time = daily_df['SHOT TIME'].min()
        last_shot_time = daily_df['SHOT TIME'].max()
        last_shot_ct_series = daily_df.loc[daily_df['SHOT TIME'] == last_shot_time, 'Actual CT']
        last_shot_ct = last_shot_ct_series.iloc[0] if not last_shot_ct_series.empty else 0
        time_span_sec = (last_shot_time - first_shot_time).total_seconds()
        # This is the 'wall clock' time. It will be adjusted for run breaks.
        base_run_time_sec = time_span_sec + last_shot_ct

        # --- 7. EMBED RR LOGIC (Pass 1: Find Downtime) ---
        df_rr = daily_df.copy().sort_values("SHOT TIME").reset_index(drop=True)

        df_rr["time_diff_sec"] = df_rr["SHOT TIME"].diff().dt.total_seconds()
        df_rr.loc[0, "time_diff_sec"] = df_rr.loc[0, "Actual CT"] 

        # --- v6.27: Identify Run Breaks ---
        run_break_threshold_sec = run_interval_hours * 3600
        is_run_break = df_rr["time_diff_sec"] > run_break_threshold_sec
        run_break_time_sec = df_rr.loc[is_run_break, 'time_diff_sec'].sum()

        # --- MODIFY Filtered Run Time (sec) ---
        results['Filtered Run Time (sec)'] = base_run_time_sec - run_break_time_sec


        # --- 7a. "DUAL TOLERANCE" / "SAFE HARBOR" LOGIC ---
        
        # Get Approved CT for the day
        if not daily_df['Approved CT'].mode().empty:
            APPROVED_CT_day = daily_df['Approved CT'].mode().iloc[0]
        else:
            APPROVED_CT_day = 0
            
        # Get Actual Mode CT for the day
        df_for_mode = df_rr[df_rr["Actual CT"] < 999.9]
        if not df_for_mode.empty and not df_for_mode['Actual CT'].mode().empty:
            mode_ct = df_for_mode['Actual CT'].mode().iloc[0]
        else:
            mode_ct = 0
            
        # --- v6.54: DEFINE REFERENCE CT ---
        target_perc_ratio = target_output_perc / 100.0
        if target_perc_ratio == 0:
            REFERENCE_CT_day = np.inf 
        else:
            REFERENCE_CT_day = APPROVED_CT_day / target_perc_ratio

        # Create the two "Safe" bands
        mode_lower_limit = mode_ct * (1 - mode_ct_tolerance)
        mode_upper_limit = mode_ct * (1 + mode_ct_tolerance)
        
        # --- v6.54: "Approved" band is now the REFERENCE_CT ---
        reference_lower_limit = REFERENCE_CT_day * (1 - approved_ct_tolerance)
        reference_upper_limit = REFERENCE_CT_day * (1 + approved_ct_tolerance)

        # --- 7b. RR Stop Detection ---
        is_hard_stop_code = df_rr["Actual CT"] >= 999.9
        prev_actual_ct = df_rr["Actual CT"].shift(1).fillna(0)
        
        # --- v6.27: Modify is_time_gap to exclude run breaks ---
        # --- v6.54: A time gap is now relative to REFERENCE_CT, not prev_actual_ct ---
        is_time_gap = (df_rr["time_diff_sec"] > (REFERENCE_CT_day + rr_downtime_gap)) & ~is_run_break
        
        # Check if shot is inside EITHER band
        in_mode_band = (df_rr["Actual CT"] >= mode_lower_limit) & (df_rr["Actual CT"] <= mode_upper_limit)
        
        # --- v6.54: Check against REFERENCE band ---
        in_reference_band = (df_rr["Actual CT"] >= reference_lower_limit) & (df_rr["Actual CT"] <= reference_upper_limit)
        
        # Abnormal cycle = NOT in mode band AND NOT in reference band
        is_abnormal_cycle = ~(in_mode_band | in_reference_band) & ~is_hard_stop_code

        # --- v6.27: Modify stop_flag to include run breaks ---
        df_rr["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code | is_run_break, 1, 0)
        if not df_rr.empty:
            df_rr.loc[0, "stop_flag"] = 0 

        df_rr['adj_ct_sec'] = df_rr['Actual CT']
        df_rr.loc[is_time_gap, 'adj_ct_sec'] = df_rr['time_diff_sec']
        df_rr.loc[is_hard_stop_code, 'adj_ct_sec'] = 0 
        # --- v6.27: Add exclusion for run breaks (so they aren't counted as downtime) ---
        df_rr.loc[is_run_break, 'adj_ct_sec'] = 0 

        # --- 8. "THE GREAT SEPARATION" ---
        df_production = df_rr[df_rr['stop_flag'] == 0].copy() # For Segments 1 & 2
        df_downtime   = df_rr[df_rr['stop_flag'] == 1].copy() # For Segment 3
        
        # --- 9. Get Config (Approved CT & Max Cavities) ---
        if APPROVED_CT_day == 0 or pd.isna(APPROVED_CT_day): 
            APPROVED_CT_day = 1 # Avoid divide-by-zero

        max_cavities = daily_df['Working Cavities'].max()
        if max_cavities == 0 or pd.isna(max_cavities): max_cavities = 1
        
        
        # --- 10. Calculate The 4 Segments (in Parts) ---

        # SEGMENT 4: Optimal Production (Theoretical Max)
        # --- This MUST always use APPROVED_CT_day ---
        results['Optimal Output (parts)'] = (results['Filtered Run Time (sec)'] / APPROVED_CT_day) * max_cavities

        # SEGMENT 3: RR Downtime Loss
        results['Capacity Loss (downtime) (sec)'] = df_downtime['adj_ct_sec'].sum()
        # --- v6.54: Use REFERENCE_CT_day ---
        results['Capacity Loss (downtime) (parts)'] = (results['Capacity Loss (downtime) (sec)'] / REFERENCE_CT_day) * max_cavities

        # SEGMENT 1: Actual Production
        results['Actual Output (parts)'] = df_production['Working Cavities'].sum()
        results['Actual Cycle Time Total (sec)'] = df_production['Actual CT'].sum() # True production time

        # SEGMENT 2: Inefficiency (CT Slow/Fast) Loss
        
        # Calculate TIME Loss/Gain
        # --- v6.54: Use REFERENCE_CT_day ---
        df_production['time_gain_sec'] = np.where(
            df_production['Actual CT'] < REFERENCE_CT_day,
            (REFERENCE_CT_day - df_production['Actual CT']), # Time gained per shot
            0
        )
        df_production['time_loss_sec'] = np.where(
            df_production['Actual CT'] > REFERENCE_CT_day,
            (df_production['Actual CT'] - REFERENCE_CT_day), # Time lost per shot
            0
        )
        results['Capacity Gain (fast cycle time) (sec)'] = df_production['time_gain_sec'].sum()
        results['Capacity Loss (slow cycle time) (sec)'] = df_production['time_loss_sec'].sum()

        # Calculate PARTS Loss/Gain
        # --- v6.54: Use REFERENCE_CT_day ---
        df_production['parts_gain'] = np.where(
            df_production['Actual CT'] < REFERENCE_CT_day,
            ((REFERENCE_CT_day - df_production['Actual CT']) / REFERENCE_CT_day) * df_production['Working Cavities'],
            0
        )
        df_production['parts_loss'] = np.where(
            df_production['Actual CT'] > REFERENCE_CT_day,
            ((df_production['Actual CT'] - REFERENCE_CT_day) / REFERENCE_CT_day) * df_production['Working Cavities'],
            0
        )
        
        results['Capacity Loss (slow cycle time) (parts)'] = df_production['parts_loss'].sum()
        results['Capacity Gain (fast cycle time) (parts)'] = df_production['parts_gain'].sum()

        # --- Create a unified 'Shot Type' column for all shots ---
        # --- v6.54: Use REFERENCE_CT_day ---
        conditions = [
            (df_production['Actual CT'] > REFERENCE_CT_day),
            (df_production['Actual CT'] < REFERENCE_CT_day),
            (df_production['Actual CT'] == REFERENCE_CT_day)
        ]
        choices = ['Slow', 'Fast', 'On Target']
        df_production['Shot Type'] = np.select(conditions, choices, default='N/A')
        
        df_rr['Shot Type'] = df_production['Shot Type'] 
        # --- v6.27: Add new Shot Type for run breaks ---
        df_rr.loc[is_run_break, 'Shot Type'] = 'Run Break (Excluded)'
        df_rr['Shot Type'].fillna('RR Downtime (Stop)', inplace=True)
        
        df_rr['Approved CT'] = APPROVED_CT_day
        # --- v6.54: Add REFERENCE_CT_day to the shot df for analysis ---
        df_rr['Reference CT'] = REFERENCE_CT_day


        if not df_rr.empty:
            all_shots_list.append(df_rr)

        # --- 11. Final Aggregations ---
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']

        net_cycle_loss_sec = results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # Target Calculations
        # --- Target Output is based on Optimal (APPROVED_CT) ---
        results['Target Output (parts)'] = results['Optimal Output (parts)'] * (target_output_perc / 100.0)
        results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Target Output (parts)'] # Can be positive
        results['Capacity Loss (vs Target) (parts)'] = np.maximum(0, results['Target Output (parts)'] - results['Actual Output (parts)']) # Only the negative gap
        
        # --- v6.54: This column is not really used, set to 0 ---
        results['Capacity Loss (vs Target) (sec)'] = 0


        # New Shot Counts
        results['Total Shots (all)'] = len(daily_df)
        results['Production Shots'] = len(df_production)
        results['Downtime Shots'] = len(df_downtime)

        daily_results_list.append(results)

    # --- 12. Format and Return Final DataFrame ---
    if not daily_results_list:
        st.warning("No data found to process.")
        return None, None

    # --- v6.1.1 FIX: Replace inf with NaN before fillna ---
    final_df = pd.DataFrame(daily_results_list).replace([np.inf, -np.inf], np.nan).fillna(0) # Fill NaNs with 0
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.set_index('Date')

    if not all_shots_list:
        return final_df, pd.DataFrame()

    all_shots_df = pd.concat(all_shots_list, ignore_index=True)
    all_shots_df['date'] = all_shots_df['SHOT TIME'].dt.date
    
    return final_df, all_shots_df

# ==================================================================
#                        CACHING WRAPPER
# ==================================================================

@st.cache_data
def run_capacity_calculation(raw_data_df, toggle, cavities, target_perc, mode_tol, approved_tol, rr_gap, run_interval, _cache_version=None):
    """Cached wrapper for the main calculation function."""
    return calculate_capacity_risk(
        raw_data_df,
        toggle,
        cavities,
        target_perc,
        mode_tol,      # Pass new arg
        approved_tol,  # Pass new arg
        rr_gap,        # Pass new arg
        run_interval   # Pass new arg
    )

# ==================================================================
#                        STREAMLIT APP LAYOUT
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

mode_ct_tolerance = st.sidebar.slider(
    "Mode CT Tolerance (%)", 0.01, 0.50, 0.25, 0.01, 
    help="Tolerance band (±) around the **Actual Mode CT**."
)
approved_ct_tolerance = st.sidebar.slider(
    "Approved CT Tolerance (%)", 0.01, 0.50, 0.25, 0.01, 
    help="Tolerance band (±) around the **Approved CT**. This creates a 'Safe Harbor' for good shots."
)
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
    ['Daily', 'Weekly', 'Monthly'],
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
    # --- v6.54: Set to 100.0 so REFERENCE_CT = APPROVED_CT ---
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
            
            # --- v6.54: Dual Calculation Logic ---
            
            # 1. Always calculate vs. Optimal (100%)
            # --- v6.50: Add cache buster to fix KeyError ---
            cache_key_optimal = f"{__version__}_{uploaded_file.name}_100.0"
            results_df_optimal, all_shots_df_optimal = run_capacity_calculation(
                df_raw,
                toggle_filter,
                default_cavities,
                100.0, # Force 100% for this run
                mode_ct_tolerance,      
                approved_ct_tolerance,   
                rr_downtime_gap,         
                run_interval_hours,      
                _cache_version=cache_key_optimal
            )

            if results_df_optimal is None or results_df_optimal.empty:
                st.error("No valid data found in file. Cannot proceed.")
            else:
                # 2. If in Target View, calculate vs. Target as well
                if benchmark_view == "Target Output":
                    # --- v6.50: Add cache buster to fix KeyError ---
                    cache_key_target = f"{__version__}_{uploaded_file.name}_{target_output_perc}"
                    results_df_target, all_shots_df_target = run_capacity_calculation(
                        df_raw,
                        toggle_filter,
                        default_cavities,
                        target_output_perc, # Use user-selected target
                        mode_ct_tolerance,      
                        approved_ct_tolerance,   
                        rr_downtime_gap,         
                        run_interval_hours,      
                        _cache_version=cache_key_target
                    )
                    
                    if results_df_target is None or results_df_target.empty:
                        st.warning("Could not calculate Target view (no data). Defaulting to Optimal view.")
                        results_df = results_df_optimal
                        all_shots_df = all_shots_df_optimal
                        benchmark_view = "Optimal Output" # Force view back
                    else:
                        # Set the *main* dfs to the target results
                        results_df = results_df_target
                        all_shots_df = all_shots_df_target
                else:
                    # Otherwise, the *main* dfs are the optimal results
                    results_df = results_df_optimal
                    all_shots_df = all_shots_df_optimal
                # --- End v6.54 Dual Calculation ---

                # --- 1. All-Time Summary Dashboard Calculations ---
                st.header("All-Time Summary")

                # 1. Calculate totals (based on primary calculation, which is now benchmark-aware)
                total_produced = results_df['Actual Output (parts)'].sum()
                total_downtime_loss_parts = results_df['Capacity Loss (downtime) (parts)'].sum()
                total_slow_loss_parts = results_df['Capacity Loss (slow cycle time) (parts)'].sum()
                total_fast_gain_parts = results_df['Capacity Gain (fast cycle time) (parts)'].sum()
                total_net_cycle_loss_parts = total_slow_loss_parts - total_fast_gain_parts
                
                # These are always based on the 100% (Optimal) run
                total_optimal = results_df_optimal['Optimal Output (parts)'].sum() 
                # --- v6.54: Get target from the *correct* dataframe ---
                total_target = results_df['Target Output (parts)'].sum()
                
                # Calculate corresponding time values
                total_downtime_loss_sec = results_df['Capacity Loss (downtime) (sec)'].sum()
                total_slow_loss_sec = results_df['Capacity Loss (slow cycle time) (sec)'].sum()
                total_fast_gain_sec = results_df['Capacity Gain (fast cycle time) (sec)'].sum()
                total_net_cycle_loss_sec = total_slow_loss_sec - total_fast_gain_sec

                total_actual_ct_sec = results_df['Actual Cycle Time Total (sec)'].sum()
                total_actual_ct_dhm = format_seconds_to_dhm(total_actual_ct_sec)

                run_time_sec_total = results_df['Filtered Run Time (sec)'].sum()
                run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)
                run_time_label = "Overall Run Time" if not toggle_filter else "Filtered Run Time"
                actual_output_perc_val = (total_produced / total_optimal) if total_optimal > 0 else 0

                # --- NEW: Calculate the final headline numbers ---
                total_calculated_net_loss_parts = total_downtime_loss_parts + total_net_cycle_loss_parts
                total_calculated_net_loss_sec = total_downtime_loss_sec + total_net_cycle_loss_sec
                
                # --- v6.5: Removed percentages for progress bars ---


                # --- NEW LAYOUT (Replaces old 4-column layout) ---
                
                # --- v6.42: Set title based on benchmark ---
                benchmark_title = "Optimal Output" if benchmark_view == "Optimal Output" else f"Target Output ({target_output_perc:.0f}%)"

                # --- Box 1: Overall Summary ---
                st.subheader(f"Overall Summary")
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.metric(run_time_label, run_time_dhm_total)
                    
                    with c2:
                        st.metric("Optimal Output (parts)", f"{total_optimal:,.0f}")
                        if benchmark_view == "Target Output":
                             st.caption(f"Target Output: {total_target:,.0f}")
                        
                    with c3:
                        # --- v6.2: Removed delta ---
                        st.metric(f"Actual Output ({actual_output_perc_val:.1%})", f"{total_produced:,.0f} parts")
                        st.caption(f"Actual Production Time: {total_actual_ct_dhm}")
                        
                    with c4:
                        # --- v6.42: Show Gap to Target if in Target view ---
                        if benchmark_view == "Target Output":
                            gap_to_target = total_produced - total_target
                            st.metric("Gap to Target", f"{gap_to_target:+,.0f} parts")
                            gap_perc = (gap_to_target / total_target) if total_target > 0 else 0
                            st.caption(f"Gap: {gap_perc:+.1%}")
                        else:
                            total_true_net_loss_parts = total_optimal - total_produced
                            # --- v6.44: Fix calculation ---
                            total_true_net_loss_sec = (total_true_net_loss_parts / total_optimal) * run_time_sec_total if total_optimal > 0 else 0
                            
                            st.metric("Total Capacity Loss (True)", f"{total_true_net_loss_parts:,.0f} parts")
                            st.caption(f"Total Time Lost: {format_seconds_to_dhm(total_true_net_loss_sec)}")

                # --- Box 2: Capacity Loss Breakdown ---
                st.subheader(f"Capacity Loss Breakdown (vs {benchmark_title})")
                st.info(f"These values are calculated based on the *time-based* logic (Downtime + Slow/Fast Cycles) using **{benchmark_title}** as the benchmark.")
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        # --- v6.2: Removed delta ---
                        st.metric("Loss (RR Downtime)", f"{total_downtime_loss_parts:,.0f} parts")
                        st.caption(f"Time Lost: {format_seconds_to_dhm(total_downtime_loss_sec)}")

                    with c2:
                        # --- v6.2: Removed delta ---
                        st.metric("Loss (Slow Cycles)", f"{total_slow_loss_parts:,.0f} parts")
                        st.caption(f"Time Lost: {format_seconds_to_dhm(total_slow_loss_sec)}")

                    with c3:
                        # --- v6.2: Removed delta ---
                        st.metric("Gain (Fast Cycles)", f"{total_fast_gain_parts:,.0f} parts")
                        st.caption(f"Time Gained: {format_seconds_to_dhm(total_fast_gain_sec)}")
                
                    with c4:
                        # --- v6.2: Removed delta ---
                        st.metric("Total Net Loss", f"{total_calculated_net_loss_parts:,.0f} parts")
                        st.caption(f"Net Time Lost: {format_seconds_to_dhm(total_net_cycle_loss_sec)}")


                # --- Collapsible Daily Summary Table ---
                with st.expander("View Daily Summary Data"):

                    # --- v6.44: Use primary results_df ---
                    daily_summary_df = results_df.copy()

                    # Calculate all % and formatted columns needed for the table
                    daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    daily_summary_df['Actual Output (parts %)'] = np.where( daily_summary_df['Optimal Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / daily_summary_df['Optimal Output (parts)'], 0 )
                    
                    # --- v6.42: Change percentage base ---
                    if benchmark_view == "Optimal Output":
                        perc_base_parts = daily_summary_df['Optimal Output (parts)']
                        perc_base_sec = daily_summary_df['Filtered Run Time (sec)']
                    else: # Target View
                        perc_base_parts = daily_summary_df['Target Output (parts)']
                        # --- v6.54: Fix base sec calculation ---
                        perc_base_sec = np.where(
                            daily_summary_df['Optimal Output (parts)'] > 0,
                            daily_summary_df['Target Output (parts)'] * (daily_summary_df['Filtered Run Time (sec)'] / daily_summary_df['Optimal Output (parts)']),
                            0
                        )

                    
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

                    # --- v6.1.1: Conditional Styling ---
                    # --- v6.3.2: Fixed IndentationError ---
                    if benchmark_view == "Optimal Output":
                        daily_kpi_table['Total Capacity Loss (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                        daily_kpi_table['Total Capacity Loss (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                        
                        st.dataframe(daily_kpi_table, width='stretch')

                    else: # Target Output
                        # --- v6.3.2: FIX for ValueError ---
                        # Force the column to numeric to handle any non-numeric values (like inf) before formatting
                        daily_summary_df['Gap to Target (parts)'] = pd.to_numeric(daily_summary_df['Gap to Target (parts)'], errors='coerce').fillna(0)
                        
                        # --- v6.22 FIX: Corrected format string (space removed) ---
                        daily_kpi_table['Gap to Target (parts)'] = daily_summary_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                        
                        # --- v6.45: Remove unused column ---
                        # daily_kpi_table['Capacity Loss (vs Target) (Time)'] = ...

                        st.dataframe(daily_kpi_table.style.applymap(
                            lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                            subset=['Gap to Target (parts)']
                        ), width='stretch')

                st.divider()

                # --- 2. WATERFALL CHART (REMOVED) ---
                # --- v6.35: Removing this section to prevent crashes. ---
                # ... (Waterfall code remains commented out) ...
                # st.divider() # <-- Also commenting out this divider

                # --- 3. AGGREGATED REPORT (Chart & Table) ---
                
                # --- v6.54: Helper function for processing dataframes ---
                def process_aggregated_dataframe(df_to_process, benchmark_mode, target_perc_val):
                    if data_frequency == 'Weekly':
                        agg_df = df_to_process.resample('W').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Weekly"
                    elif data_frequency == 'Monthly':
                        agg_df = df_to_process.resample('ME').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                        chart_title_prefix = "Monthly"
                    else: # Daily
                        agg_df = df_to_process.copy()
                        chart_title_prefix = "Daily"

                    # --- Calculate Percentage Columns AFTER aggregation ---
                    if benchmark_mode == "Optimal Output":
                        perc_base_parts = agg_df['Optimal Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Optimal)"
                    else: # Target View
                        perc_base_parts = agg_df['Target Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Target {target_perc_val:.0f}%)"

                    agg_df['Actual Output (%)'] = np.where( agg_df['Optimal Output (parts)'] > 0, agg_df['Actual Output (parts)'] / agg_df['Optimal Output (parts)'], 0)
                    agg_df['Production Shots (%)'] = np.where( agg_df['Total Shots (all)'] > 0, agg_df['Production Shots'] / agg_df['Total Shots (all)'], 0)
                    agg_df['Actual Cycle Time Total (time %)'] = np.where( agg_df['Filtered Run Time (sec)'] > 0, agg_df['Actual Cycle Time Total (sec)'] / agg_df['Filtered Run Time (sec)'], 0)
                    
                    agg_df['Capacity Loss (downtime) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (downtime) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (slow cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Gain (fast cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, agg_df['Total Capacity Loss (parts)'] / perc_base_parts, 0)

                    agg_df['Capacity Loss (vs Target) (parts %)'] = np.where( agg_df['Target Output (parts)'] > 0, agg_df['Capacity Loss (vs Target) (parts)'] / agg_df['Target Output (parts)'], 0)
                    agg_df['Total Capacity Loss (cycle time) (parts)'] = agg_df['Capacity Loss (slow cycle time) (parts)'] - agg_df['Capacity Gain (fast cycle time) (parts)']
                    
                    agg_df['Filtered Run Time (d/h/m)'] = agg_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                    agg_df['Actual Cycle Time Total (d/h/m)'] = agg_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                    
                    return agg_df, chart_title
                # --- End v6.54 Helper ---

                # --- v6.54: Process the main dataframe for the chart ---
                display_df, chart_title = process_aggregated_dataframe(results_df, benchmark_view, target_output_perc)
                
                if data_frequency == 'Weekly':
                    xaxis_title = "Week"
                elif data_frequency == 'Monthly':
                    xaxis_title = "Month"
                else: # Daily
                    xaxis_title = "Date"
                
                chart_df = display_df.reset_index()

                # --- NEW: Unified Performance Breakdown Chart (Time Series) ---
                st.header(f"{data_frequency} Performance Breakdown (vs {benchmark_title})")
                fig_ts = go.Figure()

                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Actual Output (parts)'],
                    name='Actual Output',
                    marker_color='green',
                    customdata=chart_df['Actual Output (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual Output: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                chart_df['Net Cycle Time Loss (parts)'] = chart_df['Total Capacity Loss (cycle time) (parts)']
                chart_df['Net Cycle Time Loss (positive)'] = np.maximum(0, chart_df['Net Cycle Time Loss (parts)'])

                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Net Cycle Time Loss (positive)'],
                    name='Capacity Loss (cycle time)',
                    marker_color='#ffb347', # Pastel Orange
                    customdata=np.stack((
                        chart_df['Net Cycle Time Loss (parts)'],
                        chart_df['Capacity Loss (slow cycle time) (parts)'],
                        chart_df['Capacity Gain (fast cycle time) (parts)']
                    ), axis=-1),
                    hovertemplate=
                        '<b>%{x|%Y-%m-%d}</b><br>' +
                        '<b>Net Cycle Time Loss: %{customdata[0]:,.0f}</b><br>' +
                        'Slow Cycle Loss: %{customdata[1]:,.0f}<br>' +
                        'Fast Cycle Gain: -%{customdata[2]:,.0f}<br>' + 
                        '<extra></extra>'
                ))
                
                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Capacity Loss (downtime) (parts)'],
                    name='Run Rate Downtime (Stops)',
                    marker_color='#ff6961', # Pastel Red
                    customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></all>'
                ))
                
                fig_ts.update_layout(barmode='stack')

                if benchmark_view == "Target Output":
                    fig_ts.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Target Output (parts)'],
                        name='Target Output',
                        mode='lines',
                        line=dict(color='deepskyblue', dash='dash'),
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Target: %{y:,.0f}<extra></extra>'
                    ))
                    
                fig_ts.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    # --- v6.54: Use Optimal data for this line ---
                    y=results_df_optimal.resample(data_frequency[0]).sum()['Optimal Output (parts)'],
                    name='Optimal Output',
                    mode='lines',
                    line=dict(color='darkblue', dash='dot'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Optimal: %{y:,.0f}<extra></extra>'
                ))

                fig_ts.update_layout(
                    title=chart_title,
                    xaxis_title=xaxis_title,
                    yaxis_title='Parts (Output & Loss)',
                    legend_title='Metric',
                    hovermode="x unified"
                )
                st.plotly_chart(fig_ts, width='stretch')

                # --- Full Data Table (Open by Default) ---
                
                # --- v6.54: Process main df for totals table ---
                display_df_totals, _ = process_aggregated_dataframe(results_df, benchmark_view, target_output_perc)
                
                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df_totals.index)

                report_table_1['Total Shots (all)'] = display_df_totals['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Production Shots'] = display_df_totals.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                report_table_1['Downtime Shots'] = display_df_totals['Downtime Shots'].map('{:,.0f}'.format)
                report_table_1[run_time_label] = display_df_totals.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Production Time'] = display_df_totals.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                st.dataframe(report_table_1, width='stretch')

                # --- v6.54: Conditional Tables ---
                
                # --- TABLE 1: vs Optimal ---
                # Always show this table
                st.header(f"Capacity Loss & Gain Report (vs Optimal) ({data_frequency})")
                
                # Process the optimal dataframe
                display_df_optimal, _ = process_aggregated_dataframe(results_df_optimal, "Optimal Output", 100.0)
                
                report_table_optimal = pd.DataFrame(index=display_df_optimal.index)
                report_table_optimal['Optimal Output (parts)'] = display_df_optimal['Optimal Output (parts)'].map('{:,.2f}'.format)
                report_table_optimal['Actual Output (parts)'] = display_df_optimal.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                report_table_optimal['Loss (RR Downtime)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                report_table_optimal['Loss (Slow Cycles)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                report_table_optimal['Gain (Fast Cycles)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                report_table_optimal['Total Net Loss'] = display_df_optimal.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                st.dataframe(report_table_optimal, width='stretch')
                
                
                if benchmark_view == "Target Output": 
                    # --- TABLE 2: vs Target ---
                    st.header(f"Capacity Loss & Gain Report (vs Target {target_output_perc:.0f}%) ({data_frequency})")
                    st.info(f"All Loss/Gain values in this table are calculated relative to the Target CT ({target_output_perc:.0f}% of Optimal).")
                    
                    # We already processed this, it's in 'display_df'
                    report_table_target = pd.DataFrame(index=display_df.index)
                    report_table_target['Target Output (parts)'] = display_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f}", axis=1)
                    report_table_target['Actual Output (parts)'] = display_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                    
                    report_table_target['Gap to Target (parts)'] = display_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                    report_table_target['Gap % (vs Target)'] = display_df.apply(lambda r: r['Gap to Target (parts)'] / r['Target Output (parts)'] if r['Target Output (parts)'] > 0 else 0, axis=1).apply(lambda x: "{:+.1%}".format(x) if pd.notna(x) else "N/A")
                    
                    report_table_target['Loss (RR Downtime)'] = display_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                    report_table_target['Loss (Slow Cycles)'] = display_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                    report_table_target['Gain (Fast Cycles)'] = display_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                    report_table_target['Total Net Loss'] = display_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                    
                    st.dataframe(report_table_target.style.applymap(
                        lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                        subset=['Gap to Target (parts)', 'Gap % (vs Target)']
                    ), width='stretch')
                # --- End v6.54 ---


                # --- 4. SHOT-BY-SHOT ANALYSIS ---
                st.divider()
                st.header("Shot-by-Shot Analysis (All Shots)")
                
                # --- v6.54: Benchmark title is dynamic ---
                st.info(f"This chart shows all shots. 'Production' shots are color-coded based on the **{benchmark_title}** benchmark. 'RR Downtime (Stop)' shots are grey.")

                if all_shots_df.empty:
                    st.warning("No shots were found in the file to analyze.")
                else:
                    available_dates = sorted(all_shots_df['date'].unique(), reverse=True)
                    selected_date = st.selectbox(
                        "Select a Date to Analyze",
                        options=available_dates,
                        format_func=lambda d: d.strftime('%Y-%m-%d') # Format for display
                    )

                    df_day_shots = all_shots_df[all_shots_df['date'] == selected_date]
                    
                    st.subheader("Chart Controls")
                    # --- v6.27: Filter out huge run breaks from the slider max calculation ---
                    non_break_df = df_day_shots[df_day_shots['Shot Type'] != 'Run Break (Excluded)']
                    max_ct_for_day = 100 # Default
                    if not non_break_df.empty:
                        max_ct_for_day = non_break_df['Actual CT'].max()

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

                    if df_day_shots.empty:
                        st.warning(f"No shots found for {selected_date}.")
                    else:
                        # --- v6.54: Use Reference CT for chart ---
                        reference_ct_for_day = df_day_shots['Reference CT'].iloc[0] 
                        reference_ct_label = "Approved CT" if benchmark_view == "Optimal Output" else "Target CT"

                        fig_ct = go.Figure()
                        # --- v6.27: Add new color for run breaks ---
                        color_map = {
                            'Slow': '#ff6961', 
                            'Fast': '#ffb347', 
                            'On Target': '#3498DB', 
                            'RR Downtime (Stop)': '#808080',
                            'Run Break (Excluded)': '#d3d3d3' # Light grey
                        }


                        for shot_type, color in color_map.items():
                            df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                            if not df_subset.empty:
                                fig_ct.add_bar(
                                    x=df_subset['SHOT TIME'], y=df_subset['Actual CT'],
                                    name=shot_type, marker_color=color,
                                    # --- v6.24: Add Shot Type to hover text ---
                                    hovertemplate='<b>%{x|%H:%M:%S}</b><br>Shot Type: %{fullData.name}<br>Actual CT: %{y:.2f}s<extra></extra>'
                                )
                        
                        # --- v6.54: Use Reference CT for line ---
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
                        # --- v6.54 End ---

                        fig_ct.update_layout(
                            title=f"All Shots for {selected_date}",
                            xaxis_title='Time of Day',
                            yaxis_title='Actual Cycle Time (sec)',
                            hovermode="closest",
                            # --- v6.31: Fix typo y_aws_max -> y_axis_max ---
                            yaxis_range=[0, y_axis_max], # Apply the zoom
                            # --- v6.25: REMOVED barmode='overlay' ---
                        )
                        st.plotly_chart(fig_ct, width='stretch')

                        st.subheader(f"Data for all {len(df_day_shots)} shots on {selected_date}")
                        st.dataframe(
                            df_day_shots[[
                                'SHOT TIME', 'Actual CT', 'Approved CT',
                                'Working Cavities', 'Shot Type', 'stop_flag', 'Reference CT'
                            ]].style.format({
                                'Actual CT': '{:.2f}',
                                'Approved CT': '{:.1f}',
                                'Reference CT': '{:.2f}',
                                'SHOT TIME': lambda t: t.strftime('%H:%M:%S')
                            }),
                            width='stretch'
                        )

else:
    st.info("👈 Please upload a data file to begin.")