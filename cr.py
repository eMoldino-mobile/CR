import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import cr_rebuild_utils as cr_utils
import importlib
importlib.reload(cr_utils)

# --- Setup ---
st.set_page_config(layout="wide", page_title="Capacity Risk (Rebuild)")
st.title("Capacity Risk Analysis (Base)")
st.info("Rebuilt on Unified 'Run-Based' Logic")

# --- Sidebar ---
st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

st.sidebar.header("Logic Parameters")
tolerance = st.sidebar.slider("Tolerance Band", 0.01, 0.5, 0.05)
gap = st.sidebar.slider("Downtime Gap (s)", 0.0, 10.0, 2.0)
interval = st.sidebar.slider("Run Interval (h)", 1, 24, 8, help="Gaps > X hours are excluded from Time calculations.")
default_cavs = st.sidebar.number_input("Default Cavities", 1, 100, 4)

if uploaded_file:
    # 1. Load
    df_all = cr_utils.load_data(uploaded_file)
    
    if df_all.empty:
        st.stop()
        
    # 2. Select Tool
    if "TOOL_ID" in df_all.columns:
        tools = df_all["TOOL_ID"].unique()
        selected_tool = st.sidebar.selectbox("Select Tool", tools)
        df_tool = df_all[df_all["TOOL_ID"] == selected_tool].copy()
    else:
        st.warning("No TOOL_ID column found. Analyzing entire file.")
        df_tool = df_all.copy()
        
    if "APPROVED CT" not in df_tool.columns:
        st.error("Missing 'APPROVED CT' column. Essential for Capacity Analysis.")
        st.stop()

    # 3. Calculate
    metrics = cr_utils.calculate_capacity_metrics(df_tool, tolerance, gap, interval, default_cavs)
    
    if not metrics:
        st.warning("No valid production runs found.")
        st.stop()
        
    # --- DASHBOARD ---
    
    # Row 1: The Basics (Time & Parts)
    st.subheader("1. The Efficiency Bridge")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Total Run Duration", cr_utils.format_duration(metrics["Total Run Duration (sec)"]), help="Sum of all runs. Excludes weekends/nights.")
    c2.metric("Optimal Output", f"{metrics['Optimal Output (parts)']:,.0f}", help="(Total Run Duration / Approved CT) * Cavities")
    c3.metric("Actual Output", f"{metrics['Actual Output (parts)']:,.0f}")
    
    gap = metrics['Optimal Output (parts)'] - metrics['Actual Output (parts)']
    c4.metric("Total Capacity Loss", f"{gap:,.0f}", delta="Lost Parts", delta_color="inverse")
    
    # Row 2: Waterfall Chart (The "Why")
    st.markdown("---")
    st.subheader("2. Where did the capacity go?")
    
    fig = go.Figure(go.Waterfall(
        name = "20", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["Optimal Capacity", "Loss: Downtime", "Loss: Speed (Slow Cycles)", "Gain: Speed (Fast Cycles)", "Actual Output"],
        textposition = "outside",
        text = [
            f"{metrics['Optimal Output (parts)']:,.0f}", 
            f"-{metrics['Loss - Downtime (parts)']:,.0f}", 
            f"-{metrics['Loss - Speed (parts)']:,.0f}", 
            f"+{metrics['Gain - Speed (parts)']:,.0f}", 
            f"{metrics['Actual Output (parts)']:,.0f}"
        ],
        y = [
            metrics['Optimal Output (parts)'], 
            -metrics['Loss - Downtime (parts)'], 
            -metrics['Loss - Speed (parts)'], 
            metrics['Gain - Speed (parts)'], 
            metrics['Actual Output (parts)']
        ],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ff6961"}}, # Red
        increasing = {"marker":{"color":"#77dd77"}}, # Green
        totals = {"marker":{"color":"#3498DB"}} # Blue
    ))
    
    fig.update_layout(title = "Capacity Loss Bridge", watermark=dict(text='Run-Based Logic', opacity=0.1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Detailed Metrics
    st.markdown("---")
    with st.expander("View Detailed Calculations"):
        st.json(metrics)

else:
    st.info("Upload a file to begin.")