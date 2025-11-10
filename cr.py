import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==================================================================
# 🚨 DEPLOYMENT CONTROL: INCREMENT THIS VALUE ON EVERY NEW DEPLOYMENT
# ==================================================================
__version__ = "5.6 (Chart Colors & Fixes)"
# ==================================================================

# ==================================================================
#        ... existing code ...
                st.header("Production Output Overview (The 4 Segments)")

                fig_waterfall = go.Figure(go.Waterfall(
                    name = "Segments",
                    orientation = "v",
                    measure = ["absolute", "relative", "relative", "total"],
                    x = [
                        "<b>Segment 4: Optimal</b>", 
                        "<b>Segment 3: Run Rate Downtime (Stops)</b>", 
                        "<b>Segment 2: Capacity Loss (cycle time)</b>",  # v5.5 - Terminology
                        "<b>Segment 1: Actual Output</b>"
                    ],
                    text = [
                        f"{total_optimal:,.0f}",
                        f"{-total_downtime_loss_parts:,.0f}",
                        f"{-total_net_cycle_loss_parts:,.0f}",
                        f"{total_produced:,.0f}"
                    ],
                    y = [
                        total_optimal, 
                        -total_downtime_loss_parts, 
                        -total_net_cycle_loss_parts, 
                        total_produced
                    ],
                    textposition = "outside",
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                    
                    # --- v5.6 - Manual Color Coding (Replaces decreasing, increasing, totals) ---
                    marker = dict(
                        color=[
                            "darkblue", # Segment 4: Optimal
                            "#ff6961",  # Segment 3: RR Downtime (Pastel Red)
                            "#ffb347",  # Segment 2: Cycle Time Loss (Pastel Orange)
                            "green"   # Segment 1: Actual Output (Green)
                        ],
                        line=dict(
                            color="rgb(63, 63, 63)", # Optional: add a border
                            width=1
                        )
                    )
                    # --- End v5.6 Change ---
                ))

                fig_waterfall.update_layout(
                    title="Capacity Breakdown (All Time)",
                    yaxis_title="Parts",
                    showlegend=False
                )
                
                fig_waterfall.add_shape(
                    type='line', x0=-0.5, x1=3.5,
                    y0=total_target, y1=total_target,
                    line=dict(color='deepskyblue', dash='dash') # v5.5 - Color change
                )
                fig_waterfall.add_annotation(
                    x=3.5, y=total_target, text="Target Output", 
                    showarrow=True, arrowhead=1, ax=20, ay=-20
                )
                
                fig_waterfall.add_shape(
                    type='line', x0=-0.5, x1=3.5,
                    y0=total_produced, y1=total_produced,
                    line=dict(color='green', dash='dot')
                )

                st.plotly_chart(fig_waterfall, use_container_width=True)


                st.divider()

                # --- 3. AGGREGATED REPORT (Chart & Table) ---

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
                    display_df = results_df.copy()
                    chart_title = "Daily Capacity Report"
                    xaxis_title = "Date"

                # --- Calculate Percentage Columns AFTER aggregation ---
                display_df['Actual Output (%)'] = np.where( display_df['Optimal Output (parts)'] > 0, display_df['Actual Output (parts)'] / display_df['Optimal Output (parts)'], 0)
                display_df['Production Shots (%)'] = np.where( display_df['Total Shots (all)'] > 0, display_df['Production Shots'] / display_df['Total Shots (all)'], 0)
                display_df['Actual Cycle Time Total (time %)'] = np.where( display_df['Filtered Run Time (sec)'] > 0, display_df['Actual Cycle Time Total (sec)'] / display_df['Filtered Run Time (sec)'], 0)
                display_df['Capacity Loss (downtime) (parts %)'] = np.where( display_df['Optimal Output (parts)'] > 0, display_df['Capacity Loss (downtime) (parts)'] / display_df['Optimal Output (parts)'], 0)
                display_df['Capacity Loss (slow cycle time) (parts %)'] = np.where( display_df['Optimal Output (parts)'] > 0, display_df['Capacity Loss (slow cycle time) (parts)'] / display_df['Optimal Output (parts)'], 0)
                display_df['Capacity Gain (fast cycle time) (parts %)'] = np.where( display_df['Optimal Output (parts)'] > 0, display_df['Capacity Gain (fast cycle time) (parts)'] / display_df['Optimal Output (parts)'], 0)
                display_df['Total Capacity Loss (parts %)'] = np.where( display_df['Optimal Output (parts)'] > 0, display_df['Total Capacity Loss (parts)'] / display_df['Optimal Output (parts)'], 0)
                display_df['Capacity Loss (vs Target) (parts %)'] = np.where( display_df['Target Output (parts)'] > 0, display_df['Capacity Loss (vs Target) (parts)'] / display_df['Target Output (parts)'], 0)

                _target_output_perc_array = np.full(len(display_df), target_output_perc / 100.0)

                display_df['Filtered Run Time (d/h/m)'] = display_df['Filtered Run Time (sec)'].apply(format_seconds_to_dhm)
                display_df['Actual Cycle Time Total (d/h/m)'] = display_df['Actual Cycle Time Total (sec)'].apply(format_seconds_to_dhm)

                chart_df = display_df.reset_index()

                # --- NEW: Unified Performance Breakdown Chart (Time Series) ---
                st.header(f"{data_frequency} Performance Breakdown")
                fig_ts = go.Figure()

                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Actual Output (parts)'],
                    name='Actual Output (Segment 1)',
                    marker_color='#77dd77', # Pastel Green
                    customdata=chart_df['Actual Output (%)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Actual Output: %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                chart_df['Net Cycle Time Loss (parts)'] = chart_df['Capacity Loss (slow cycle time) (parts)'] - chart_df['Capacity Gain (fast cycle time) (parts)'] # v5.5 - Terminology
                chart_df['Net Cycle Time Loss (positive)'] = np.maximum(0, chart_df['Net Cycle Time Loss (parts)']) # v5.5 - Terminology

                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Net Cycle Time Loss (positive)'], # v5.5 - Terminology
                    name='Capacity Loss (cycle time) (Segment 2)', # v5.5 - Terminology
                    marker_color='#ffb347', # Pastel Orange
                    customdata=np.stack((
                        chart_df['Net Cycle Time Loss (parts)'], # v5.5 - Terminology
                        chart_df['Capacity Loss (slow cycle time) (parts)'],
                        chart_df['Capacity Gain (fast cycle time) (parts)']
                    ), axis=-1),
                    hovertemplate=
                        '<b>%{x|%Y-%m-%d}</b><br>' +
                        '<b>Net Cycle Time Loss: %{customdata[0]:,.0f}</b><br>' + # v5.5 - Terminology
                        'Slow Cycle Loss: %{customdata[1]:,.0f}<br>' +
                        'Fast Cycle Gain: -%{customdata[2]:,.0f}<br>' +
                        '<extra></extra>'
                ))
                
                fig_ts.add_trace(go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Capacity Loss (downtime) (parts)'],
                    name='Run Rate Downtime (Stops) (Segment 3)',
                    marker_color='#ff6961', # Pastel Red
                    customdata=chart_df['Capacity Loss (downtime) (parts %)'],
                    hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Run Rate Downtime (Stops): %{y:,.0f} (%{customdata:.1%})<extra></extra>'
                ))
                
                fig_ts.update_layout(barmode='stack')

                if benchmark_view == "Target Output":
                    fig_ts.add_trace(go.Scatter(
                        x=chart_df['Date'],
                        y=chart_df['Target Output (parts)'],
                        name='Target Output',
                        mode='lines',
                        line=dict(color='deepskyblue', dash='dash'), # v5.5 - Color change
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Target: %{y:,.0f}<extra></extra>'
                    ))
                    
                fig_ts.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['Optimal Output (parts)'],
                    name='Optimal Output (Segment 4)',
                    mode='lines',
                    line=dict(color='darkblue', dash='dot'), # v5.5 - Color change
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

                st.header(f"Production Totals Report ({data_frequency})")
                report_table_1 = pd.DataFrame(index=display_df.index)

                report_table_1['Total Shots (all)'] = display_df['Total Shots (all)'].map('{:,.0f}'.format)
                report_table_1['Production Shots'] = display_df.apply(lambda r: f"{r['Production Shots']:,.0f} ({r['Production Shots (%)']:.1%})", axis=1)
                report_table_1['Downtime Shots'] = display_df['Downtime Shots'].map('{:,.0f}'.format)
                report_table_1[run_time_label] = display_df.apply(lambda r: f"{r['Filtered Run Time (d/h/m)']} ({r['Filtered Run Time (sec)']:,.0f}s)", axis=1)
                report_table_1['Actual Production Time'] = display_df.apply(lambda r: f"{r['Actual Cycle Time Total (d/h/m)']} ({r['Actual Cycle Time Total (time %)']:.1%})", axis=1)

                st.dataframe(report_table_1, use_container_width=True)

                table_2_title = "Capacity Loss & Gain Report"
                st.header(f"{table_2_title} ({data_frequency})")

                report_table_2 = pd.DataFrame(index=display_df.index)

                report_table_2['Optimal Output (parts)'] = display_df['Optimal Output (parts)'].map('{:,.2f}'.format)
                
                if benchmark_view == "Target Output":
                    report_table_2['Target Output (parts)'] = display_df.apply(lambda r: f"{r['Target Output (parts)']:,.2f} ({target_output_perc / 100.0:.0%})", axis=1)
                    
                # --- v5.5 - NAMEERROR FIX: display_table -> display_df ---
                report_table_2['Actual Output (parts)'] = display_df.apply(lambda r: f"{r['Actual Output (parts)']:,.2f} ({r['Actual Output (%)']:.1%})", axis=1)

                report_table_2['Loss (RR Downtime)'] = display_df.apply(lambda r: f"{r['Capacity Loss (downtime) (parts)']:,.2f} ({r['Capacity Loss (downtime) (parts %)']:.1%})", axis=1)
                report_table_2['Loss (Slow Cycles)'] = display_df.apply(lambda r: f"{r['Capacity Loss (slow cycle time) (parts)']:,.2f} ({r['Capacity Loss (slow cycle time) (parts %)']:.1%})", axis=1)
                report_table_2['Gain (Fast Cycles)'] = display_df.apply(lambda r: f"{r['Capacity Gain (fast cycle time) (parts)']:,.2f} ({r['Capacity Gain (fast cycle time) (parts %)']:.1%})", axis=1)
                report_table_2['Total Net Loss'] = display_df.apply(lambda r: f"{r['Total Capacity Loss (parts)']:,.2f} ({r['Total Capacity Loss (parts %)']:.1%})", axis=1)
                
                if benchmark_view == "Target Output":
                    report_table_2['Loss (vs Target)'] = display_df.apply(lambda r: f"{r['Capacity Loss (vs Target) (parts)']:,.2f} ({r['Capacity Loss (vs Target) (parts %)']:.1%})", axis=1)

                st.dataframe(report_table_2, use_container_width=True)

                # --- 4. SHOT-BY-SHOT ANALYSIS ---
                st.divider()
                st.header("Shot-by-Shot Analysis (All Shots)")
                st.info("This chart shows all shots. 'Production' shots are color-coded (Slow/Fast/On Target), and 'RR Downtime (Stop)' shots are grey.")

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
                    max_ct_for_day = df_day_shots['Actual CT'].max()
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
                        approved_ct_for_day = df_day_shots['Approved CT'].iloc[0]

                        fig_ct = go.Figure()
                        color_map = {'Slow': '#ff6961', 'Fast': '#ffb347', 'On Target': '#3498DB', 'RR Downtime (Stop)': '#808080'}

                        for shot_type, color in color_map.items():
                            df_subset = df_day_shots[df_day_shots['Shot Type'] == shot_type]
                            if not df_subset.empty:
                                fig_ct.add_bar(
                                    x=df_subset['SHOT TIME'], y=df_subset['Actual CT'],
                                    name=shot_type, marker_color=color,
                                    hovertemplate='<b>%{x|%H:%M:%S}</b><br>Actual CT: %{y:.2f}s<extra></extra>'
                                )

                        # v5.2 - Fix typo `approved_T_day`
                        fig_ct.add_shape(
                            type='line',
                            x0=df_day_shots['SHOT TIME'].min(), x1=df_day_shots['SHOT TIME'].max(),
                            y0=approved_ct_for_day, y1=approved_ct_for_day,
                            line=dict(color='green', dash='dash'), name=f'Approved CT ({approved_ct_for_day}s)'
                        )
                        # --- v5.5 - NAMEERROR FIX: approved_sct_for_day -> approved_ct_for_day ---
                        fig_ct.add_annotation(
                            x=df_day_shots['SHOT TIME'].max(), y=approved_ct_for_day,
                            text=f"Approved CT: {approved_ct_for_day}s", showarrow=True, arrowhead=1
                        )

                        fig_ct.update_layout(
                            title=f"All Shots for {selected_date}",
                            xaxis_title='Time of Day',
                            yaxis_title='Actual Cycle Time (sec)',
                            hovermode="closest",
                            yaxis_range=[0, y_axis_max], # Apply the zoom
                            barmode='overlay' 
                        )
                        st.plotly_chart(fig_ct, use_container_width=True)

                        st.subheader(f"Data for all {len(df_day_shots)} shots on {selected_date}")
                        st.dataframe(
                            df_day_shots[[
                                'SHOT TIME', 'Actual CT', 'Approved CT',
                                'Working Cavities', 'Shot Type', 'stop_flag'
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
    st.info("👈 Please upload a data file to begin.")