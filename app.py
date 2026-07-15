import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import io

# Page configuration
st.set_page_config(
    page_title="Cordillera Negra Dam Restoration Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HEADER SECTION (RESEARCH FOCUS) ---
st.title("🏛️ INVESTIGATING EXTREME WEATHER AND POSSIBLE IMPACTS ON PRE-HISPANIC DAM RESTORATION")
st.markdown("## *IN CORDILLERA NEGRA, PERU*")

st.info(
    "**Research Framework:** Analyzing localized hydro-climatic extreme indices across a clean "
    "30-year operational climate baseline (1996–2025). These metrics provide empirical boundary "
    "conditions to guide structural stability, spillway calculations, and storage reliability "
    "for ancient water infrastructure restoration."
)
st.markdown("---")

# --- DATA LOADING HELPERS ---
@st.cache_data
def load_processed_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def load_raw_daily_data(file_path):
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Baseline File Mappings
site_paths = {
    "Shukkloc (Proposed System)": "data/shukkloc_proposed_master_metrics_1996_2025.csv",
    "Ricococha / Weetacocha Grid Matrix": "data/ricococha_weetacocha_grid_master_metrics_1996_2025.csv"
}

raw_paths = {
    "Shukkloc (Proposed System)": "raw_daily/shukkloc_raw_daily_1996_2025.csv",
    "Ricococha / Weetacocha Grid Matrix": "raw_daily/ricococha_raw_daily_1996_2025.csv"
}

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🗺️ Dashboard Configuration Panel")

# 1. User Data Source Selection (Baseline vs Custom Upload)
data_source = st.sidebar.selectbox(
    "Select Data Universe:",
    ["Use Baseline Research Sites", "📤 Upload Custom Annual Metrics File"]
)

# Render upload interface if selected
uploaded_file = None
if data_source == "📤 Upload Custom Annual Metrics File":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Custom Time Series")
    uploaded_file = st.sidebar.file_uploader(
        "Upload processed annual CSV file:", 
        type=["csv"],
        help="Upload a custom time-series matching the baseline shape format."
    )
    
    # Generate Template Data Stream for Sandbox Testing
    template_df = pd.DataFrame({
        "Year": [2020, 2021, 2022, 2023, 2024, 2025],
        "Rx1day": [25.4, 18.2, 33.1, 22.0, 15.6, 28.9],
        "Rx5day": [45.2, 38.0, 62.4, 41.1, 29.8, 51.2],
        "R5mm": [42, 38, 55, 40, 31, 47],
        "R10mm": [12, 9, 21, 14, 6, 18],
        "SDII": [6.2, 5.8, 7.1, 6.0, 5.4, 6.5],
        "CDD": [120, 145, 98, 130, 160, 115],
        "CWD": [5, 4, 8, 5, 3, 6],
        "PRCPTOT": [450.2, 390.5, 610.4, 480.0, 320.1, 540.6],
        "R95pTOT": [85.4, 42.1, 140.6, 92.0, 25.4, 112.1],
        "N95_Frequency": [2, 1, 4, 2, 0, 3]
    })
    
    template_csv = template_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Format Template CSV",
        data=template_csv,
        file_name="custom_climate_metrics_template.csv",
        mime="text/csv",
        help="Download an empty/sample structured template to arrange your data variables correctly."
    )

# 2. Setup Traditional View Modes if using baseline data
view_mode = "Single Catchment Focus"
if data_source == "Use Baseline Research Sites":
    view_mode = st.sidebar.radio("Analysis View Mode:", ["Single Catchment Focus", "📊 Compare Both Catchments Side-by-Side"])

# Metric Selection Setup
metric_categories = {
    "⚠️ Core Rainfall Extremes (Rx1day, Rx5day)": ["Rx1day", "Rx5day"],
    "⏳ Sedimentation & Structural Stress (R5mm, R10mm, SDII)": ["R5mm", "R10mm", "SDII"],
    "💧 Storage Reliability & Water Security (CDD, CWD, PRCPTOT)": ["CDD", "CWD", "PRCPTOT"],
    "📈 Percentile Threshold Volume (R95pTOT, N95)": ["R95pTOT", "N95_Frequency"]
}
selected_category = st.sidebar.radio("Analysis Metric Category:", list(metric_categories.keys()))
metrics_to_plot = metric_categories[selected_category]

y_units = {
    "Rx1day": "Precipitation (mm/day)", "Rx5day": "Precipitation (mm/5-days)",
    "R5mm": "Days/Year", "R10mm": "Days/Year", "SDII": "Intensity (mm/wet day)",
    "CDD": "Consecutive Days", "CWD": "Consecutive Days", "PRCPTOT": "Total Rainfall (mm/year)",
    "R95pTOT": "Total Rainfall (mm/year)", "N95_Frequency": "Days/Year"
}

# --- MAIN ANALYSIS ROUTING ENGINE ---
try:
    if data_source == "📤 Upload Custom Annual Metrics File":
        if uploaded_file is not None:
            # Parse user provided runtime file object
            df = pd.read_csv(uploaded_file)
            
            # Basic validation block checking for minimum structural index criteria
            if "Year" not in df.columns or not any(m in df.columns for m in metrics_to_plot):
                st.error("❌ Format Mismatch! Please verify your uploaded spreadsheet contains a 'Year' column and correct core metric index names.")
            else:
                st.success("✅ Custom time series read successfully! Executing trend calculations.")
                
                st.subheader("📊 Uploaded System: Category Summary Indicators")
                cols = st.columns(len(metrics_to_plot))
                for i, metric in enumerate(metrics_to_plot):
                    with cols[i]:
                        if metric in df.columns:
                            st.metric(label=f"Uploaded Mean ({metric})", value=f"{df[metric].mean():.2f}", delta=f"Max: {df[metric].max():.1f}")
                        else:
                            st.warning(f"Metric '{metric}' omitted in uploaded dataset.")
                            
                st.markdown("---")
                st.subheader("📈 Localized Trend Profile: Custom Data Target")
                
                for metric in metrics_to_plot:
                    if metric in df.columns:
                        x, y = df["Year"].values, df[metric].values
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        fig = go.Figure()
                        if "Frequency" in metric or "mm" in metric and "SDII" not in metric:
                            fig.add_trace(go.Bar(x=x, y=y, name="Annual Metric Value", opacity=0.7, marker_color="rgb(44, 160, 44)"))
                        else:
                            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Annual Metric Value", line=dict(color="#2ca02c")))
                            
                        fig.add_trace(go.Scatter(x=x, y=p(x), mode="lines", name="Linear Trend Profile", line=dict(color="gray", dash="dash")))
                        
                        fig.update_layout(
                            title=f"Temporal Dynamics for Custom Array: {metric}",
                            xaxis_title="Year",
                            yaxis_title=y_units.get(metric, "Scale Value"),
                            height=380,
                            margin=dict(l=20, r=20, t=50, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("📥 Awaiting file upload. Download the format template template from the left panel to test with custom parameters.")
            
    else:
        # Standard execution block logic for primary research site baselines
        if "Compare" in view_mode:
            st.subheader("📊 Cross-Catchment Comparative Baseline Analytics")
            df_shuk = load_processed_data(site_paths["Shukkloc (Proposed System)"])
            df_rico = load_processed_data(site_paths["Ricococha / Weetacocha Grid Matrix"])
            
            for metric in metrics_to_plot:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_shuk["Year"], y=df_shuk[metric], mode="lines+markers", name="Shukkloc System", line=dict(color="#1f77b4", width=2.5)))
                fig.add_trace(go.Scatter(x=df_rico["Year"], y=df_rico[metric], mode="lines+markers", name="Ricococha Grid Matrix", line=dict(color="#ff7f0e", width=2.5)))
                fig.update_layout(
                    title=f"Comparative Dynamics for {metric} (1996-2025)",
                    xaxis_title="Year",
                    yaxis_title=y_units.get(metric, "Scale Value"),
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            selected_site = st.sidebar.selectbox("Select Target Location:", list(site_paths.keys()))
            df = load_processed_data(site_paths[selected_site])
            
            st.subheader(f"📊 Summary Indicators: {selected_site}")
            cols = st.columns(len(metrics_to_plot))
            for i, metric in enumerate(metrics_to_plot):
                with cols[i]:
                    st.metric(label=f"Historical Mean ({metric})", value=f"{df[metric].mean():.2f}", delta=f"Max: {df[metric].max():.1f}")
                    
            st.markdown("---")
            st.subheader(f"📈 Localized Trend Profile: {selected_site}")
            
            for metric in metrics_to_plot:
                x, y = df["Year"].values, df[metric].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                fig = go.Figure()
                if "Frequency" in metric or "mm" in metric and "SDII" not in metric:
                    fig.add_trace(go.Bar(x=x, y=y, name="Annual Metric Value", opacity=0.7, marker_color="rgb(56, 112, 134)"))
                else:
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Annual Metric Value"))
                    
                fig.add_trace(go.Scatter(x=x, y=p(x), mode="lines", name="Linear Trend Profile", line=dict(color="gray", dash="dash")))
                
                fig.update_layout(
                    title=f"Temporal Dynamics for {metric}",
                    xaxis_title="Year",
                    yaxis_title=y_units.get(metric, "Scale Value"),
                    height=380,
                    margin=dict(l=20, r=20, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- RENDER CONTINUOUS RAW RAIN DATA MATRIX IF SITES SELECTED ---
    if data_source == "Use Baseline Research Sites":
        st.markdown("---")
        st.subheader("🌧️ Raw Daily Precipitation Temporal Explorer")
        st.markdown("Dynamically alter the frequency window to inspect chronological daily rainfall trends or upscale them to macro-scale totals.")
        
        selected_raw_site = st.selectbox("Select Catchment to Inspect Raw Data:", list(raw_paths.keys()), key="raw_site_choice")
        time_frequency = st.segmented_control("Select Visualization Time Step Resolution:", ["Daily", "Monthly Total", "Annual Total"], default="Daily")
        
        raw_df = load_raw_daily_data(raw_paths[selected_raw_site])
        raw_df.set_index("Date", inplace=True)
        
        if time_frequency == "Daily":
            plot_df = raw_df.copy()
            chart_title = "Historical Continuous Daily Precipitation Time Series"
            y_label = "Precipitation (mm/day)"
        elif time_frequency == "Monthly Total":
            plot_df = raw_df.resample("ME").sum()
            chart_title = "Aggregated Cumulative Monthly Rainfall Blocks"
            y_label = "Precipitation (mm/month)"
        else:
            plot_df = raw_df.resample("YE").sum()
            chart_title = "Aggregated Cumulative Historical Annual Totals"
            y_label = "Precipitation (mm/year)"
            
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Precipitation_mm"], mode="lines", name="Rainfall Level", line=dict(color="#2ca02c")))
        fig_raw.update_layout(
            title=f"{chart_title} — {selected_raw_site}",
            xaxis_title="Timeline Horizon",
            yaxis_title=y_label,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # --- ARCHIVAL REPLICATION DECK & OPEN SCIENCE EXPORTS ---
    st.markdown("---")
    st.subheader("🔓 Open Science & Replication Data Deck")
    
    tab1, tab2 = st.tabs(["📊 Processed Annual Metrics Data", "🌧️ Raw Base Rainfall Series"])
    
    with tab1:
        if data_source == "Use Baseline Research Sites":
            site_for_csv = selected_raw_site if "Compare" in view_mode else selected_site
            proc_df = load_processed_data(site_paths[site_for_csv])
        else:
            proc_df = df if uploaded_file is not None else template_df
            site_for_csv = "Custom_Upload"
            
        st.dataframe(proc_df.set_index("Year") if "Year" in proc_df.columns else proc_df, use_container_width=True)
        st.download_button(
            label="📥 Download Active Processed Annual Data Matrix (CSV)",
            data=proc_df.to_csv(index=False).encode('utf-8'),
            file_name=f"{site_for_csv.lower().replace(' ', '_')}_annual_metrics.csv",
            mime="text/csv"
        )
        
    with tab2:
        if data_source == "Use Baseline Research Sites":
            st.markdown("**Preview of raw chronological baseline data (First 500 lines):**")
            st.dataframe(raw_df.head(500), use_container_width=True)
            st.download_button(
                label="📥 Download Unaggregated Raw Daily Rainfall Dataset (CSV)",
                data=raw_df.to_csv().encode('utf-8'),
                file_name=f"{selected_raw_site.lower().replace(' ', '_')}_raw_daily_precipitation.csv",
                mime="text/csv"
            )
        else:
            st.info("ℹ️ Raw base time-series uploads are restricted to structural index matrices. Daily upscaling calculations are maintained via the local NetCDF processing environment.")

except FileNotFoundError:
    st.error("❌ Component files missing. Verify your CSV assets exist inside the /data and /raw_daily workspace subdirectories.")