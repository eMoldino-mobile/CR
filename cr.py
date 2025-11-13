import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
__version__ = "6.86 (Cleaned up chart tooltips)"
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
# --- v6.83: This function now respects target_output_perc ---
def calculate_capacity_risk(_df_raw, toggle_filter, default_cavities, target_output_perc, mode_ct_tolerance, rr_downtime_gap, run_interval_hours):
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
        'Date', 'Filtered Run Time (sec)', 
        'Benchmark Output (parts)', # Renamed from Optimal Output
        'Capacity Loss (downtime) (sec)',
        'Capacity Loss (downtime) (parts)',
        'Actual Output (parts)', 'Actual Cycle Time Total (sec)',
        'Capacity Gain (fast cycle time) (sec)', 'Capacity Loss (slow cycle time) (sec)',
        'Capacity Loss (slow cycle time) (parts)', 'Capacity Gain (fast cycle time) (parts)',
        'Total Capacity Loss (parts)', 'Total Capacity Loss (sec)',
        'Optimal Output (100%) (parts)', # New column for reference
        'Target Output (parts)', 'Gap to Target (parts)',
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
        # --- v6.63: FIX BUG --- Set first shot's diff to 0, not its own CT
        df_rr.loc[0, "time_diff_sec"] = 0.0

        # --- v6.27: Identify Run Breaks ---
        run_break_threshold_sec = run_interval_hours * 3600
        is_run_break = df_rr["time_diff_sec"] > run_break_threshold_sec
        run_break_time_sec = df_rr.loc[is_run_break, 'time_diff_sec'].sum()

        # --- MODIFY Filtered Run Time (sec) ---
        results['Filtered Run Time (sec)'] = base_run_time_sec - run_break_time_sec


        # --- 7a. "SAFE HARBOR" LOGIC ---
        
        # Get Approved CT for the day
        if not daily_df['Approved CT'].mode().empty:
            APPROVED_CT_day = daily_df['Approved CT'].mode().iloc[0]
        else:
            APPROVED_CT_day = 0
            
        # --- v6.57: Restore Mode CT calculation ---
        df_for_mode = df_rr[df_rr["Actual CT"] < 999.9]
        if not df_for_mode.empty and not df_for_mode['Actual CT'].mode().empty:
            mode_ct = df_for_mode['Actual CT'].mode().iloc[0]
        else:
            mode_ct = 0
            
        # --- v6.83: REFERENCE_CT is *dynamic* based on target_output_perc ---
        target_perc_ratio = target_output_perc / 100.0
        if target_perc_ratio == 0:
            REFERENCE_CT_day = np.inf
        else:
            REFERENCE_CT_day = APPROVED_CT_day / target_perc_ratio

        # --- v6.61: Create the *only* "Safe" band (Mode CT) ---
        mode_lower_limit = mode_ct * (1 - mode_ct_tolerance)
        mode_upper_limit = mode_ct * (1 + mode_ct_tolerance)

        # --- 7b. RR Stop Detection ---
        is_hard_stop_code = df_rr["Actual CT"] >= 999.9
        
        # --- v6.58: Restore correct time gap logic ---
        prev_actual_ct = df_rr["Actual CT"].shift(1).fillna(0)
        is_time_gap = (df_rr["time_diff_sec"] > (prev_actual_ct + rr_downtime_gap)) & ~is_run_break
        
        # --- v6.61: Check against MODE band ONLY ---
        in_mode_band = (df_rr["Actual CT"] >= mode_lower_limit) & (df_rr["Actual CT"] <= mode_upper_limit)
        
        # Abnormal cycle = NOT in mode band
        is_abnormal_cycle = ~in_mode_band & ~is_hard_stop_code

        # --- v6.27: Modify stop_flag to include run breaks ---
        df_rr["stop_flag"] = np.where(is_abnormal_cycle | is_time_gap | is_hard_stop_code | is_run_break, 1, 0)
        
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

        # SEGMENT 4: Benchmark Production
        # --- v6.83: Use REFERENCE_CT_day ---
        results['Benchmark Output (parts)'] = (results['Filtered Run Time (sec)'] / REFERENCE_CT_day) * max_cavities

        # SEGMENT 3: RR Downtime Loss
        results['Capacity Loss (downtime) (sec)'] = df_downtime['adj_ct_sec'].sum()

        # SEGMENT 1: Actual Production
        results['Actual Output (parts)'] = df_production['Working Cavities'].sum()
        results['Actual Cycle Time Total (sec)'] = df_production['Actual CT'].sum() # True production time

        # SEGMENT 2: Inefficiency (CT Slow/Fast) Loss
        
        # Calculate TIME Loss/Gain
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

        # Calculate PARTS Loss/Gain (Accurate, per-shot)
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
        
        # --- v6.56: RECONCILIATION LOGIC ---
        true_capacity_loss_parts = results['Benchmark Output (parts)'] - results['Actual Output (parts)']
        net_cycle_loss_parts = results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        results['Capacity Loss (downtime) (parts)'] = true_capacity_loss_parts - net_cycle_loss_parts
        # --- END v.6.56 RECONCILIATION ---


        # --- Create a unified 'Shot Type' column for all shots ---
        conditions = [
            (df_production['Actual CT'] > REFERENCE_CT_day),
            (df_production['Actual CT'] < REFERENCE_CT_day),
            (df_production['Actual CT'] == REFERENCE_CT_day)
        ]
        choices = ['Slow', 'Fast', 'On Target']
        df_production['Shot Type'] = np.select(conditions, choices, default='N/A')
        
        df_rr['Shot Type'] = df_production['Shot Type'] 
        df_rr.loc[is_run_break, 'Shot Type'] = 'Run Break (Excluded)'
        df_rr['Shot Type'].fillna('RR Downtime (Stop)', inplace=True)
        
        df_rr['Approved CT'] = APPROVED_CT_day
        df_rr['Reference CT'] = REFERENCE_CT_day
        df_rr['Mode CT Lower'] = mode_lower_limit
        df_rr['Mode CT Upper'] = mode_upper_limit


        if not df_rr.empty:
            all_shots_list.append(df_rr)

        # --- 11. Final Aggregations ---
        results['Total Capacity Loss (parts)'] = results['Capacity Loss (downtime) (parts)'] + results['Capacity Loss (slow cycle time) (parts)'] - results['Capacity Gain (fast cycle time) (parts)']
        net_cycle_loss_sec = results['Capacity Loss (slow cycle time) (sec)'] - results['Capacity Gain (fast cycle time) (sec)']
        results['Total Capacity Loss (sec)'] = results['Capacity Loss (downtime) (sec)'] + net_cycle_loss_sec

        # --- v6.83: Add Optimal (100%) and Target (X%) ---
        results['Optimal Output (100%) (parts)'] = (results['Filtered Run Time (sec)'] / APPROVED_CT_day) * max_cavities
        
        # Target Output is the Benchmark in this run
        results['Target Output (parts)'] = results['Benchmark Output (parts)']
        if target_output_perc == 100.0:
            # We are in Optimal run, so Gap is vs Optimal
            results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Benchmark Output (parts)']
        else:
            # We are in Target run, so Gap is vs Target
            results['Gap to Target (parts)'] = results['Actual Output (parts)'] - results['Benchmark Output (parts)']


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
# --- v6.83: Renamed target_output_perc ---
def run_capacity_calculation(raw_data_df, toggle, cavities, target_output_perc, mode_tol, rr_gap, run_interval, _cache_version=None):
    """Cached wrapper for the main calculation function."""
    return calculate_capacity_risk(
        raw_data_df,
        toggle,
        cavities,
        target_output_perc, # This is now the dynamic benchmark
        mode_tol,      
        rr_gap,        
        run_interval   
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
            
            # --- v6.83: Dual Calculation Logic ---
            
            # 1. Always calculate vs. Optimal (100%)
            cache_key_optimal = f"{__version__}_{uploaded_file.name}_100.0_{mode_ct_tolerance}_{rr_downtime_gap}"
            results_df_optimal, all_shots_df_optimal = run_capacity_calculation(
                df_raw,
                toggle_filter,
                default_cavities,
                100.0, # Force 100% for this run
                mode_ct_tolerance,  
                rr_downtime_gap,         
                run_interval_hours,      
                _cache_version=cache_key_optimal
            )

            if results_df_optimal is None or results_df_optimal.empty:
                st.error("No valid data found in file. Cannot proceed.")
            else:
                # 2. If in Target View, calculate vs. Target as well
                if benchmark_view == "Target Output":
                    cache_key_target = f"{__version__}_{uploaded_file.name}_{target_output_perc}_{mode_ct_tolerance}_{rr_downtime_gap}"
                    results_df_target, all_shots_df_target = run_capacity_calculation(
                        df_raw,
                        toggle_filter,
                        default_cavities,
                        target_output_perc, # Use user-selected target
                        mode_ct_tolerance,  
                        rr_downtime_gap,         
                        run_interval_hours,      
                        _cache_version=cache_key_target
                    )
                    
                    if results_df_target is None or results_df_target.empty:
                        st.warning("Could not calculate Target view (no data). Defaulting to Optimal view.")
                        display_df = results_df_optimal
                        display_shots_df = all_shots_df_optimal
                        benchmark_view = "Optimal Output" # Force view back
                        target_output_perc = 100.0 # Reset perc
                    else:
                        # Set the *main* dfs to the target results
                        display_df = results_df_target
                        display_shots_df = all_shots_df_target
                else:
                    # Otherwise, the *main* dfs are the optimal results
                    display_df = results_df_optimal
                    display_shots_df = all_shots_df_optimal
                # --- End v6.83 Dual Calculation ---

                # --- 1. All-Time Summary Dashboard Calculations ---
                st.header("All-Time Summary")

                # 1. Calculate totals (based on the *active* display_df)
                total_produced = display_df['Actual Output (parts)'].sum()
                total_downtime_loss_parts = display_df['Capacity Loss (downtime) (parts)'].sum()
                total_slow_loss_parts = display_df['Capacity Loss (slow cycle time) (parts)'].sum()
                total_fast_gain_parts = display_df['Capacity Gain (fast cycle time) (parts)'].sum()
                total_net_cycle_loss_parts = total_slow_loss_parts - total_fast_gain_parts
                
                # --- v6.83: Get 100% Optimal from the optimal-specific run ---
                total_optimal_100 = results_df_optimal['Benchmark Output (parts)'].sum()
                
                # --- v6.83: Get Target from the *active* display_df ---
                total_benchmark_output = display_df['Benchmark Output (parts)'].sum()
                total_gap_to_benchmark = display_df['Gap to Target (parts)'].sum()

                
                # Calculate corresponding time values
                total_downtime_loss_sec = display_df['Capacity Loss (downtime) (sec)'].sum()
                total_slow_loss_sec = display_df['Capacity Loss (slow cycle time) (sec)'].sum()
                total_fast_gain_sec = display_df['Capacity Gain (fast cycle time) (sec)'].sum()
                total_net_cycle_loss_sec = total_slow_loss_sec - total_fast_gain_sec

                total_actual_ct_sec = display_df['Actual Cycle Time Total (sec)'].sum()
                total_actual_ct_dhm = format_seconds_to_dhm(total_actual_ct_sec)

                run_time_sec_total = display_df['Filtered Run Time (sec)'].sum()
                run_time_dhm_total = format_seconds_to_dhm(run_time_sec_total)
                run_time_label = "Overall Run Time" if not toggle_filter else "Filtered Run Time"
                actual_output_perc_val = (total_produced / total_optimal_100) if total_optimal_100 > 0 else 0

                total_calculated_net_loss_parts = display_df['Total Capacity Loss (parts)'].sum()
                total_calculated_net_loss_sec = display_df['Total Capacity Loss (sec)'].sum()
                
                # --- v6.56: Calculate True Loss (based on current benchmark) ---
                total_true_net_loss_parts = display_df['Total Capacity Loss (parts)'].sum()

                
                # --- v6.5: Removed percentages for progress bars ---


                # --- NEW LAYOUT (Replaces old 4-column layout) ---
                
                # --- v6.83: Title is now dynamic ---
                if benchmark_view == "Optimal Output":
                    benchmark_title = "Optimal Output"
                    benchmark_label = "Optimal Output (100%)"
                    benchmark_perc_label = ""
                else:
                    benchmark_title = f"Target Output ({target_output_perc:.0f}%)"
                    benchmark_label = f"Target Output ({target_output_perc:.0f}%)"
                    benchmark_perc_label = f"vs Target {target_output_perc:.0f}%"


                # --- Box 1: Overall Summary ---
                st.subheader(f"Overall Summary")
                with st.container(border=True):
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.metric(run_time_label, run_time_dhm_total)
                    
                    with c2:
                        # --- v6.83: Dynamic Benchmark Column ---
                        st.metric(benchmark_label, f"{total_benchmark_output:,.0f}")
                        if benchmark_view == "Target Output":
                             st.caption(f"Optimal (100%): {total_optimal_100:,.0f}")
                        
                    with c3:
                        # --- v6.2: Removed delta ---
                        st.metric(f"Actual Output ({actual_output_perc_val:.1%})", f"{total_produced:,.0f} parts")
                        st.caption(f"Actual Production Time: {total_actual_ct_dhm}")
                        
                    with c4:
                        # --- v6.83: Dynamic 4th Column ---
                        # This is now the "Total Net Impact" (reconciled loss)
                        st.markdown(f"**Total Net Impact ({benchmark_perc_label})**")
                        
                        net_impact_val = total_true_net_loss_parts
                        if net_impact_val > 0: # It's a loss
                            impact_color = "red"
                        else: # It's a gain
                            impact_color = "green"
                            
                        st.markdown(f"<h3><span style='color:{impact_color};'>{net_impact_val:,.0f} parts</span></h3>", unsafe_allow_html=True) 
                        st.caption(f"Net Time Lost: {format_seconds_to_dhm(total_calculated_net_loss_sec)}")

                        # --- v6.83: Add Gap to Target as a caption ---
                        if benchmark_view == "Target Output":
                            gap_perc = (total_gap_to_benchmark / total_benchmark_output) if total_benchmark_output > 0 else 0
                            st.caption(f"Gap to Target: {total_gap_to_benchmark:+,.0f} parts ({gap_perc:+.1%})")


                # --- v6.84: Replaced Stacked Bar with Waterfall ---
                st.subheader(f"Capacity Loss Breakdown (vs {benchmark_title})")
                st.info(f"These values are calculated based on the *time-based* logic (Downtime + Slow/Fast Cycles) using **{benchmark_title}** as the benchmark.")
                
                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown("<h6 style='text-align: center;'>Overall Performance Breakdown</h6>", unsafe_allow_html=True)
                    
                    # --- Waterfall Chart ---
                    # Because of our reconciliation, we have perfect numbers.
                    # Benchmark = Actual + RR Loss + Net Cycle Loss
                    
                    # --- v6.85: Dynamic Benchmark Label ---
                    waterfall_x = [f"<b>{benchmark_label}</b>", "Loss (RR Downtime)"]
                    waterfall_y = [total_benchmark_output, -total_downtime_loss_parts]
                    waterfall_measure = ["absolute", "relative"]
                    waterfall_text = [f"{total_benchmark_output:,.0f}", f"{-total_downtime_loss_parts:,.0f}"]

                    if total_net_cycle_loss_parts >= 0:
                        # It's a net loss
                        waterfall_x.append("Net Loss (Cycle Time)")
                        waterfall_y.append(-total_net_cycle_loss_parts)
                        waterfall_measure.append("relative")
                        waterfall_text.append(f"{-total_net_cycle_loss_parts:,.0f}")
                    else:
                        # It's a net gain
                        waterfall_x.append("Net Gain (Cycle Time)")
                        waterfall_y.append(abs(total_net_cycle_loss_parts)) # Add it back
                        waterfall_measure.append("relative")
                        waterfall_text.append(f"{abs(total_net_cycle_loss_parts):+,.0f}")
                    
                    # Add the final total
                    waterfall_x.append("<b>Actual Output</b>") # --- v6.85: Bold label ---
                    waterfall_y.append(total_produced)
                    waterfall_measure.append("total")
                    waterfall_text.append(f"{total_produced:,.0f}")
                    

                    fig_waterfall = go.Figure(go.Waterfall(
                        name = "Breakdown",
                        orientation = "v",
                        measure = waterfall_measure,
                        x = waterfall_x,
                        y = waterfall_y,
                        text = waterfall_text,
                        textposition = "outside",
                        connector = {"line":{"color":"rgb(63, 63, 63)"}},
                        increasing = {"marker":{"color":"#2ca02c"}}, # Green for gains
                        decreasing = {"marker":{"color":"#ff6961"}},  # Red for losses
                        totals = {"marker":{"color":"#1f77b4"}} # Blue for totals (Benchmark & Actual)
                    ))
                    
                    fig_waterfall.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0), # --- v6.84: Removed title, set margin-top to 0 ---
                        height=400,
                        yaxis_title='Parts'
                    )
                    
                    # Add Optimal (100%) line if we are in Target view
                    if benchmark_view == "Target Output":
                        # --- v6.85: Add Target Line ---
                        fig_waterfall.add_shape(
                            type='line',
                            x0=-0.5, x1=len(waterfall_x)-0.5, # Span all columns
                            y0=total_benchmark_output, y1=total_benchmark_output,
                            line=dict(color='deepskyblue', dash='dash', width=2)
                        )
                        fig_waterfall.add_annotation(
                            x=0, y=total_benchmark_output,
                            text=f"Target: {total_benchmark_output:,.0f}",
                            showarrow=True, arrowhead=1, ax=-40, ay=-20
                        )
                        
                        # --- v6.85: Add Optimal Line ---
                        fig_waterfall.add_shape(
                            type='line',
                            x0=-0.5, x1=len(waterfall_x)-0.5, # Span all columns
                            y0=total_optimal_100, y1=total_optimal_100,
                            line=dict(color='darkblue', dash='dot', width=3)
                        )
                        fig_waterfall.add_annotation(
                            x=len(waterfall_x)-0.5, y=total_optimal_100,
                            text=f"Optimal (100%): {total_optimal_100:,.0f}",
                            showarrow=True, arrowhead=1, ax=40, ay=-20
                        )
                    
                    st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
                    

                with c2:
                    # --- v6.79: New compact table layout with color ---
                    
                    # --- Helper function for color ---
                    def get_color_css(val):
                        if val > 0: return "color: red;"
                        if val < 0: return "color: green;"
                        return "color: black;"

                    # --- Color-code Total Net Loss ---
                    net_loss_val = total_calculated_net_loss_parts
                    net_loss_color = get_color_css(net_loss_val)
                    with st.container(border=True):
                        # --- v6.80: Rename to "Total Net Impact" ---
                        st.markdown(f"**Total Net Impact**")
                        st.markdown(f"<h3><span style='{net_loss_color}'>{net_loss_val:,.0f} parts</span></h3>", unsafe_allow_html=True)
                        st.caption(f"Net Time Lost: {format_seconds_to_dhm(total_calculated_net_loss_sec)}")
                    
                    # --- Create Data for the table ---
                    table_data = {
                        "Metric": [
                            "Loss (RR Downtime)", 
                            "Net Loss (Cycle Time)", 
                            # --- v6.80: Fix &nbsp; formatting ---
                            "\u00A0\u00A0\u00A0 └ Loss (Slow Cycles)", 
                            "\u00A0\u00A0\u00A0 └ Gain (Fast Cycles)"
                        ],
                        "Parts": [
                            total_downtime_loss_parts,
                            total_net_cycle_loss_parts,
                            total_slow_loss_parts,
                            total_fast_gain_parts
                        ],
                        "Time": [
                            format_seconds_to_dhm(total_downtime_loss_sec),
                            format_seconds_to_dhm(total_net_cycle_loss_sec),
                            format_seconds_to_dhm(total_slow_loss_sec),
                            format_seconds_to_dhm(total_fast_gain_sec)
                        ]
                    }
                    df_table = pd.DataFrame(table_data)

                    # --- Function to apply color styling to the "Parts" column ---
                    def style_parts_col(val, row_index):
                        # Get the correct color based on the metric
                        if row_index == 0: # Loss (RR Downtime)
                            color_style = get_color_css(val)
                        elif row_index == 1: # Net Loss (Cycle Time)
                            color_style = get_color_css(val)
                        elif row_index == 2: # Loss (Slow Cycles)
                            color_style = get_color_css(val)
                        elif row_index == 3: # Gain (Fast Cycles)
                            color_style = get_color_css(val * -1) # Invert gain for color
                        else:
                            color_style = "color: black;"
                        
                        return color_style

                    # --- Apply styling to the DataFrame ---
                    styled_df = df_table.style.apply(
                        lambda row: [style_parts_col(row['Parts'], row.name) if col == 'Parts' else '' for col in row.index],
                        axis=1
                    ).format(
                        {"Parts": "{:,.0f}"} # Apply comma formatting
                    ).set_properties(
                        **{'text-align': 'left'}, subset=['Metric', 'Time']
                    ).set_properties(
                        **{'text-align': 'right'}, subset=['Parts']
                    ).hide(axis='index') # Hide the 0,1,2,3 index
                    
                    # --- Display the styled table ---
                    st.dataframe(
                        styled_df,
                        use_container_width=True
                    )

                # --- End v6.79 Layout ---


                # --- Collapsible Daily Summary Table ---
                with st.expander("View Daily Summary Data"):

                    # --- v6.83: Use display_df ---
                    daily_summary_df = display_df.copy()

                    # Calculate all % and formatted columns needed for the table
                    daily_summary_df['Actual Cycle Time Total (time %)'] = np.where( daily_summary_df['Filtered Run Time (sec)'] > 0, daily_summary_df['Actual Cycle Time Total (sec)'] / daily_summary_df['Filtered Run Time (sec)'], 0 )
                    # --- v6.83: Actual Output % is always vs 100% Optimal ---
                    daily_summary_df['Actual Output (parts %)'] = np.where( results_df_optimal['Benchmark Output (parts)'] > 0, daily_summary_df['Actual Output (parts)'] / results_df_optimal['Benchmark Output (parts)'], 0 )
                    
                    # --- v6.83: Perc base is dynamic ---
                    perc_base_parts = daily_summary_df['Benchmark Output (parts)']
                    perc_base_sec = daily_summary_df['Filtered Run Time (sec)'] # <-- This could be improved, but let's see
                    
                    daily_summary_df['Total Capacity Loss (time %)'] = np.where( perc_base_sec > 0, daily_summary_df['Total Capacity Loss (sec)'] / perc_base_sec, 0 )
                    daily_summary_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, daily_summary_df['Total Capacity Loss (parts)'] / perc_base_parts, 0 )
                    
                    daily_summary_df['Total Capacity Loss (d/h/m)'] = daily_summary_df['Total Capacity Loss (sec)'].apply(format_seconds_to_dhm)

                    # --- v6.83: These columns are not used by Gap Allocation, but we'll calc them anyway ---
                    daily_summary_df['Capacity Loss (vs Target) (parts %)'] = np.where( daily_summary_df['Target Output (parts)'] > 0, (daily_summary_df['Target Output (parts)'] - daily_summary_df['Actual Output (parts)']) / daily_summary_df['Target Output (parts)'], 0 )
                    daily_summary_df['Capacity Loss (vs Target) (time %)'] = 0 # Not used
                    daily_summary_df['Capacity Loss (vs Target) (d/h/m)'] = "N/A" # Not used

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
                        
                        st.dataframe(daily_kpi_table, use_container_width=True)

                    else: # Target Output
                        # --- v6.3.2: FIX for ValueError ---
                        # Force the column to numeric to handle any non-numeric values (like inf) before formatting
                        daily_summary_df['Gap to Target (parts)'] = pd.to_numeric(daily_summary_df['Gap to Target (parts)'], errors='coerce').fillna(0)
                        
                        # --- v6.22 FIX: Corrected format string (space removed) ---
                        daily_kpi_table['Gap to Target (parts)'] = daily_summary_df['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                        
                        daily_kpi_table['Total Net Impact (Time)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (d/h/m)']} ({r['Total Capacity Loss (time %)']:.1%})", axis=1)
                        daily_kpi_table['Total Net Impact (parts)'] = daily_summary_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)


                        st.dataframe(daily_kpi_table.style.applymap(
                            lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                            subset=['Gap to Target (parts)']
                        ), use_container_width=True)

                st.divider()

                # --- 2. WATERFALL CHART (REMOVED) ---
                # ... (Waterfall code remains commented out) ...
                # st.divider() # <-- Also commenting out this divider

                # --- 3. AGGREGATED REPORT (Chart & Table) ---
                
                # --- v6.83: Helper function for processing dataframes ---
                def process_aggregated_dataframe(df_to_process, bm_view, target_perc, optimal_df_agg):
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
                    if bm_view == "Optimal Output":
                        perc_base_parts = agg_df['Benchmark Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Optimal)"
                    else: # Target View
                        perc_base_parts = agg_df['Benchmark Output (parts)']
                        chart_title = f"{chart_title_prefix} Capacity Report (vs Target {target_perc:.0f}%)"
                    
                    # Optimal (100%) is always from the optimal_df_agg
                    optimal_100_base = optimal_df_agg['Benchmark Output (parts)']


                    agg_df['Actual Output (%)'] = np.where( optimal_100_base > 0, agg_df['Actual Output (parts)'] / optimal_100_base, 0)
                    agg_df['Production Shots (%)'] = np.where( agg_df['Total Shots (all)'] > 0, agg_df['Production Shots'] / agg_df['Total Shots (all)'], 0)
                    agg_df['Actual Cycle Time Total (time %)'] = np.where( agg_df['Filtered Run Time (sec)'] > 0, agg_df['Actual Cycle Time Total (sec)'] / agg_df['Filtered Run Time (sec)'], 0)
                    
                    agg_df['Capacity Loss (downtime) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (downtime) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Loss (slow cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( perc_base_parts > 0, agg_df['Capacity Gain (fast cycle time) (parts)'] / perc_base_parts, 0)
                    agg_df['Total Capacity Loss (parts %)'] = np.where( perc_base_parts > 0, agg_df['Total Capacity Loss (parts)'] / perc_base_parts, 0)

                    agg_df['Total Capacity Loss (cycle time) (parts)'] = agg_df['Capacity Loss (slow cycle time) (parts)'] - agg_df['Capacity Gain (fast cycle time) (parts)']
                                            
                    
                    agg_df['Filtered Run Time (d/h/m)'] = agg_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                    agg_df['Actual Cycle Time Total (d/h/m)'] = agg_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)
                    
                    return agg_df, chart_title
                # --- End v6.83 Helper ---
                
                # --- v6.83: Pre-aggregate optimal_df for helper function ---
                if data_frequency == 'Weekly':
                    optimal_df_agg = results_df_optimal.resample('W').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                elif data_frequency == 'Monthly':
                    optimal_df_agg = results_df_optimal.resample('ME').sum().replace([np.inf, -np.inf], np.nan).fillna(0)
                else: # Daily
                    optimal_df_agg = results_df_optimal.copy()

                # --- v6.83: Process the main dataframe for the chart ---
                display_df_agg, chart_title = process_aggregated_dataframe(display_df, benchmark_view, target_output_perc, optimal_df_agg)
                
                if data_frequency == 'Weekly':
                    xaxis_title = "Week"
                elif data_frequency == 'Monthly':
                    xaxis_title = "Month"
                else: # Daily
                    xaxis_title = "Date"
                
                chart_df = display_df_agg.reset_index()

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
                    # --- v6.65: Fix SyntaxError ---
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
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                fig_ts.update_layout(barmode='stack')
                
                # --- v6.83: Add Target/Benchmark line ---
                fig_ts.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Benchmark Output (parts)'],
                    name=benchmark_label,
                    mode='lines',
                    line=dict(color='deepskyblue', dash='dash'),
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Benchmark: %{y:,.0f}<extra></extra>'
                ))
                    
                fig_ts.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=optimal_df_agg['Benchmark Output (parts)'], # Use 100% optimal
                    name='Optimal Output (100%)',
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
                st.plotly_chart(fig_ts, use_container_width=True)

                # --- Full Data Table (Open by Default) ---
                
                # --- v6.83: Use display_df_agg ---
                display_df_totals, _ = process_aggregated_dataframe(display_df, benchmark_view, target_output_perc, optimal_df_agg)
                
                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df_totals.index)

                report_table_1['Total Shots (all)'] = display_df_totals['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Production Shots'] = display_df_totals.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                report_table_1['Downtime Shots'] = display_df_totals['Downtime Shots'].map('{:,.0f}'.format)
                report_table_1[run_time_label] = display_df_totals.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Production Time'] = display_df_totals.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                st.dataframe(report_table_1, use_container_width=True)

                # --- v6.83: Conditional Tables ---
                
                # --- TABLE 1: vs Optimal ---
                st.header(f"Capacity Loss & Gain Report (vs Optimal) ({data_frequency})")
                
                display_df_optimal, _ = process_aggregated_dataframe(results_df_optimal, "Optimal Output", 100.0, optimal_df_agg)
                
                report_table_optimal = pd.DataFrame(index=display_df_optimal.index)
                report_table_optimal['Optimal Output (parts)'] = display_df_optimal['Benchmark Output (parts)'].map('{:,.2f}'.format)
                report_table_optimal['Actual Output (parts)'] = display_df_optimal.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                report_table_optimal['Loss (RR Downtime)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                report_table_optimal['Loss (Slow Cycles)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                report_table_optimal['Gain (Fast Cycles)'] = display_df_optimal.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                report_table_optimal['Total Net Loss'] = display_df_optimal.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                st.dataframe(report_table_optimal, use_container_width=True)
                
                
                if benchmark_view == "Target Output": 
                    # --- TABLE 2: vs Target ---
                    st.header(f"Capacity Loss & Gain Report (vs Target {target_output_perc:.0f}%) ({data_frequency})")
                    st.info(f"All Loss/Gain values in this table are calculated relative to the Target CT ({target_output_perc:.0f}% of Optimal).")
                    
                    display_df_target = display_df_agg
                    
                    report_table_target = pd.DataFrame(index=display_df_target.index)
                    report_table_target['Target Output (parts)'] = display_df_target.apply(lambda r: f"{r['Benchmark Output (parts)']:,.2f}", axis=1)
                    report_table_target['Actual Output (parts)'] = display_df_target.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)
                    
                    report_table_target['Gap to Target (parts)'] = display_df_target['Gap to Target (parts)'].apply(lambda x: "{:+,.2f}".format(x) if pd.notna(x) else "N/A")
                    report_table_target['Gap % (vs Target)'] = display_df_target.apply(lambda r: r['Gap to Target (parts)'] / r['Benchmark Output (parts)'] if r['Benchmark Output (parts)'] > 0 else 0, axis=1).apply(lambda x: "{:+.1%}".format(x) if pd.notna(x) else "N/A")
                    
                    report_table_target['Loss (RR Downtime)'] = display_df_target.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                    report_table_target['Loss (Slow Cycles)'] = display_df_target.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                    report_table_target['Gain (Fast Cycles)'] = display_df_target.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                    report_table_target['Total Net Loss'] = display_df_target.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                    
                    st.dataframe(report_table_target.style.applymap(
                        lambda x: 'color: green' if str(x).startswith('+') else 'color: red' if str(x).startswith('-') else None,
                        subset=['Gap to Target (parts)', 'Gap % (vs Target)']
                    ), use_container_width=True)
                # --- End v6.83 ---


                # --- 4. SHOT-BY-SHOT ANALYSIS ---
                st.divider()
                st.header("Shot-by-Shot Analysis (All Shots)")
                
                # --- v6.83: Benchmark title is dynamic ---
                st.info(f"This chart shows all shots. 'Production' shots are color-coded based on the **{benchmark_title}** benchmark. 'RR Downtime (Stop)' shots are grey.")

                if display_shots_df.empty:
                    st.warning("No shots were found in the file to analyze.")
                else:
                    available_dates = sorted(display_shots_df['date'].unique(), reverse=True)
                    selected_date = st.selectbox(
                        "Select a Date to Analyze",
                        options=available_dates,
                        format_func=lambda d: d.strftime('%Y-%m-%d') # Format for display
                    )

                    df_day_shots = display_shots_df[display_shots_df['date'] == selected_date]
                    
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
                        # --- v6.83: Use Reference CT for chart ---
                        reference_ct_for_day = df_day_shots['Reference CT'].iloc[0] 
                        reference_ct_label = "Approved CT" if benchmark_view == "Optimal Output" else f"Target CT ({target_output_perc:.0f}%)"
                        # --- v6.62: Get Mode CT limits ---
                        mode_ct_lower_for_day = df_day_shots['Mode CT Lower'].iloc[0]
                        mode_ct_upper_for_day = df_day_shots['Mode CT Upper'].iloc[0]

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
                        
                        # --- v6.62: Add shaded Mode CT band ---
                        fig_ct.add_hrect(
                            y0=mode_ct_lower_for_day, y1=mode_ct_upper_for_day,
                            fillcolor="grey", opacity=0.20,
                            line_width=0,
                            name="Mode CT Band"
                        )
                        
                        # --- v6.54: Use Reference CT for line ---
                        # --- v6.59: Fix NameError ---
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
                        # --- vs6.54 End ---

                        fig_ct.update_layout(
                            title=f"All Shots for {selected_date}",
                            xaxis_title='Time of Day',
                            yaxis_title='Actual Cycle Time (sec)',
                            hovermode="closest",
                            # --- v6.31: Fix typo y_aws_max -> y_axis_max ---
                            yaxis_range=[0, y_axis_max], # Apply the zoom
                            # --- v6.25: REMOVED barmode='overlay' ---
                        )
                        st.plotly_chart(fig_ct, use_container_width=True)

                        st.subheader(f"Data for all {len(df_day_shots)} shots on {selected_date}")
                        st.dataframe(
                            df_day_shots[[
                                'SHOT TIME', 'Actual CT', 'Approved CT',
                                'Working Cavities', 'Shot Type', 'stop_flag', 
                                'Reference CT', 'Mode CT Lower', 'Mode CT Upper' # --- v6.62: Added columns ---
                            ]].style.format({
                                'Actual CT': '{:.2f}',
                                'Approved CT': '{:.1f}',
                                'Reference CT': '{:.2f}',
                                'Mode CT Lower': '{:.2f}', # --- v6.62: Added format ---
                                'Mode CT Upper': '{:.2f}', # --- v6.62: Added format ---
                                'SHOT TIME': lambda t: t.strftime('%H:%M:%S')
                            }),
                            use_container_width=True
                        )

else:
    st.info("👈 Please upload a data file to begin.")