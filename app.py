import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

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

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🗺️ Catchment & Index Control")

site_options = {
    "Shukkloc (Proposed System)": "data/shukkloc_proposed_master_metrics_1996_2025.csv",
    "Ricococha / Weetacocha Grid Matrix": "data/ricococha_weetacocha_grid_master_metrics_1996_2025.csv"
}
selected_site = st.sidebar.selectbox("Select Target Location:", list(site_options.keys()))

metric_categories = {
    "⚠️ Core Rainfall Extremes (Rx1day, Rx5day)": ["Rx1day", "Rx5day"],
    "⏳ Sedimentation & Structural Stress (R5mm, R10mm, SDII)": ["R5mm", "R10mm", "SDII"],
    "💧 Storage Reliability & Water Security (CDD, CWD, PRCPTOT)": ["CDD", "CWD", "PRCPTOT"],
    "📈 Percentile Threshold Volume (R95pTOT, N95)": ["R95pTOT", "N95_Frequency"]
}
selected_category = st.sidebar.radio("Analysis Category:", list(metric_categories.keys()))
metrics_to_plot = metric_categories[selected_category]

# Load dataset helper
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

try:
    df = load_data(site_options[selected_site])
    
    # Y-Label Definitions for proper scientific display
    y_units = {
        "Rx1day": "Precipitation (mm/day)", "Rx5day": "Precipitation (mm/5-days)",
        "R5mm": "Days/Year", "R10mm": "Days/Year", "SDII": "Intensity (mm/wet day)",
        "CDD": "Consecutive Days", "CWD": "Consecutive Days", "PRCPTOT": "Total Rainfall (mm/year)",
        "R95pTOT": "Total Rainfall (mm/year)", "N95_Frequency": "Days/Year"
    }

    # --- DISPLAY METRIC CARD MATRIX ---
    st.subheader("📊 Category Summary Indicators")
    cols = st.columns(len(metrics_to_plot))
    for i, metric in enumerate(metrics_to_plot):
        with cols[i]:
            mean_val = df[metric].mean()
            max_val = df[metric].max()
            st.metric(
                label=f"Historical Mean ({metric})",
                value=f"{mean_val:.2f}",
                delta=f"Max Record: {max_val:.1f}"
            )
            
    st.markdown("---")
    
    # --- RENDER CHARTS LOOP ---
    st.subheader(f"📈 Localized Trend Profile: {selected_site}")
    
    for metric in metrics_to_plot:
        x = df["Year"].values
        y = df[metric].values
        
        # Calculate standard visual trend overlay
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        fig = go.Figure()
        
        # Chart style switcher based on the index type
        if "Frequency" in metric or "mm" in metric and "SDII" not in metric:
            fig.add_trace(go.Bar(x=x, y=y, name="Annual Metric Value", opacity=0.7, marker_color='rgb(56, 112, 134)'))
        else:
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Annual Metric Value", line=dict(width=2)))
            
        # Draw Trend line
        fig.add_trace(go.Scatter(
            x=x, y=p(x), mode="lines", name="Linear Trend Profile",
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Cleaned and theme-adaptive layout configuration
        fig.update_layout(
            title=f"Temporal Dynamics for {metric}",
            xaxis_title="Year",
            yaxis_title=y_units.get(metric, "Scale Value"),
            template="none", 
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="auto"),
            xaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)", zeroline=False),
            yaxis=dict(gridcolor="rgba(128, 128, 128, 0.2)", zeroline=False),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # --- ARCHIVAL TABULAR EXPLORER & REPLICATION DECK ---
    st.markdown("---")
    st.subheader("🔓 Open Science & Replication Data Deck")
    st.markdown(
        "To support independent validation and reproduction of this hydro-climatic analysis, "
        "you can explore and download both the computed annual climate metrics and the underlying raw daily time series below."
    )

    tab1, tab2 = st.tabs(["📊 Processed Annual Metrics (10 Indices)", "🌧️ Raw Daily Time Series Data"])

    with tab1:
        display_df = df[["Year"] + metrics_to_plot].set_index("Year")
        st.dataframe(display_df, use_container_width=True)
        
        metrics_csv = df.to_csv().encode('utf-8')
        st.download_button(
            label=f"📥 Download Processed Annual Indices ({selected_site})",
            data=metrics_csv,
            file_name=f"{selected_site.lower().replace(' ', '_')}_annual_metrics.csv",
            mime="text/csv",
            key="download_metrics"
        )

    with tab2:
        # Construct the matching raw daily file path based on selection
        raw_file_mapping = {
            "Shukkloc (Proposed System)": "raw_daily/shukkloc_raw_daily_1996_2025.csv",
            "Ricococha / Weetacocha Grid Matrix": "raw_daily/ricococha_raw_daily_1996_2025.csv"
        }
        
        try:
            raw_daily_df = pd.read_csv(raw_file_mapping[selected_site]).set_index("Date")
            
            # Show a slice of the daily dataset (first 500 rows for smooth interface rendering)
            st.markdown("**Preview of raw daily grid points (First 500 records):**")
            st.dataframe(raw_daily_df.head(500), use_container_width=True)
            
            # Convert full daily dataset to bytes for user download
            raw_csv_bytes = raw_daily_df.to_csv().encode('utf-8')
            st.download_button(
                label=f"📥 Download Raw Daily Precipitation Time Series ({selected_site})",
                data=raw_csv_bytes,
                file_name=f"{selected_site.lower().replace(' ', '_')}_raw_daily_rainfall.csv",
                mime="text/csv",
                key="download_raw_daily",
                help="Download the complete, unaggregated daily rainfall time series (1996–2025) for independent modeling."
            )
        except FileNotFoundError:
            st.warning("⚠️ Raw daily data file not found in the /raw_daily folder directory.")

except FileNotFoundError:
    st.error("❌ Data layer paths missing. Ensure your generated CSV files exist inside the correct directories.")