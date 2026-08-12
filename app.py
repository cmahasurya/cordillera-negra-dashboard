import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pymannkendall as mk
from streamlit_folium import st_folium
import folium

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
    "**Research Framework:** This project is an ongoing research framework continuing from an MSc dissertation completed at the "
    "**Department of Meteorology, University of Reading** (*Graduated with Distinction; Awarded Best Dissertation*). "
    "The framework investigates localized hydro-climatic extreme indices across clean, homogeneous baseline timelines "
    "in the Cordillera Negra range of Peru. These empirical metrics establish crucial boundary conditions to guide spillway design, "
    "structural stability, and long-term water storage reliability for pre-Hispanic dam restoration and heritage water engineering."
)

# --- COLLAPSIBLE SATELLITE MAP SECTION ---
with st.expander("🌍 View Study Area Satellite Map (Cordillera Negra Catchments)", expanded=False):
    st.markdown("Inspect the regional terrain and physical catchment locations for the three pre-Hispanic dam study sites:")
    
    # Map center coordinates around Cordillera Negra
    map_center = [-9.35, -77.80]
    
    # Create Folium Map with Satellite Imagery (Esri World Imagery) and zoomed-out level (zoom_start=8)
    m = folium.Map(
        location=map_center,
        zoom_start=8,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )
    
    # Target sites coordinates
    sites_coords = [
        {"name": "Shukkloc (Proposed)", "lat": -9.68118056, "lon": -77.70907778, "color": "red"},
        {"name": "Ricococha (Reconstructed)", "lat": -9.06465000, "lon": -77.91718889, "color": "blue"},
        {"name": "Weetacocha (Reconstructed)", "lat": -9.03331944, "lon": -77.92741389, "color": "cyan"}
    ]
    
    # Add Markers with Labels
    for site in sites_coords:
        folium.Marker(
            location=[site["lat"], site["lon"]],
            popup=f"<b>{site['name']}</b><br>Lat: {site['lat']:.5f}<br>Lon: {site['lon']:.5f}",
            tooltip=site["name"],
            icon=folium.Icon(color=site["color"], icon="info-sign")
        ).add_to(m)
        
    st_folium(m, width=1200, height=450)

st.markdown("---")

# --- DATA LOADING HELPERS ---
@st.cache_data
def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    col_map = {
        "year": "Year",
        "N95_days": "N95_Frequency",
        "RX1day_mm": "Rx1day",
        "RX5day_mm": "Rx5day",
        "R5mm_days": "R5mm",
        "R10mm_days": "R10mm",
        "CDD_days": "CDD",
        "CWD_days": "CWD",
        "PRCPTOT_mm": "PRCPTOT",
        "R95pTOT_mm": "R95pTOT",
        "SDII_mm_per_wetday": "SDII"
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    if "site" in df.columns:
        df["site"] = df["site"].astype("category")
    return df

@st.cache_data
def load_raw_daily_data(file_path):
    df = pd.read_csv(file_path)
    if "time" in df.columns:
        df.rename(columns={"time": "Date"}, inplace=True)
    if "precip" in df.columns:
        df.rename(columns={"precip": "Precipitation_mm"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    if "site" in df.columns:
        df["site"] = df["site"].astype("category")
    if "version" in df.columns:
        df["version"] = df["version"].astype("category")
    return df

def calculate_trend_summary(y_values):
    """Computes Mann-Kendall trend and Theil-Sen slope, returning formatted text and stats."""
    try:
        y_clean = y_values[~np.isnan(y_values)]
        if len(y_clean) < 3:
            return False, 1.0, "Insufficient Data", 0.0, "⚠️ Insufficient data points for trend testing"
            
        mk_res = mk.original_test(y_clean)
        trend_status = mk_res.trend.title()
        p_val = mk_res.p
        
        sen_res = mk.sens_slope(y_clean)
        slope_val = sen_res.slope
        
        is_significant = p_val < 0.05
        
        if is_significant:
            summary = f"✨ **Statistically Significant {trend_status} Trend** (p = {p_val:.4f}) | **Theil-Sen Slope:** {slope_val:+.4f} units/year"
        else:
            summary = f"⚪ **No Statistically Significant Trend** (p = {p_val:.4f}) | **Theil-Sen Slope:** {slope_val:+.4f} units/year"
            
        return is_significant, p_val, trend_status, slope_val, summary
    except Exception:
        return False, 1.0, "Error", 0.0, "⚠️ Trend test calculation error"

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("🗺️ Dashboard Configuration Panel")

# 1. Version Selector
version_choice = st.sidebar.radio(
    "Select PISCO Dataset Version:",
    ["PISCO v3 (1981–2025)", "PISCO v2 (1981–2019)", "⚡ Overlay PISCO v2 vs PISCO v3"]
)

# 2. Time Horizon Selector
time_horizon = st.sidebar.radio(
    "Select Temporal Analysis Window:",
    ["Post-1996 Homogeneous Reference Period (1996–End) ⭐", "Full Historical Record (1981–End)"],
    index=0,
    help="Pre-1996 data in gridded products often exhibits inhomogeneity due to sparse gauge density and satellite transitions."
)

start_year = 1996 if "1996" in time_horizon else 1981

if start_year == 1996:
    st.warning(
        "🔬 **Methodological Inhomogeneity Notice (Pre-1996 Breakpoint):** "
        "The analysis is restricted to **1996–present**. Pre-1996 gridded precipitation datasets across the high Andes "
        "exhibit documented **non-climatic artifacts and artificial variance shifts** (inhomogeneity) caused by the transition "
        "in satellite sensor inputs (e.g., introduction of CHIRP/TRMM products) and sparse surface rain gauge density prior to 1996. "
        "Filtering to 1996–present ensures a **homogeneous reference baseline** for infrastructure risk assessment."
    )

# File Paths
v2_data_path = "data/PISCO_v2_annual_metrics.csv"
v3_data_path = "data/PISCO_v3_annual_metrics.csv"
raw_file_path = "raw_daily/raw_daily_precipitation_all_sites.csv"

data_source = st.sidebar.selectbox(
    "Select Data Universe:",
    ["Use Baseline Research Sites", "📤 Upload Custom Annual Metrics File"]
)

uploaded_file = None
if data_source == "📤 Upload Custom Annual Metrics File":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload Custom Time Series")
    uploaded_file = st.sidebar.file_uploader(
        "Upload processed annual CSV file:", 
        type=["csv"],
        help="Upload a custom time-series matching the baseline shape format."
    )

view_mode = "Single Catchment Focus"
if data_source == "Use Baseline Research Sites":
    view_mode = st.sidebar.radio("Analysis View Mode:", ["Single Catchment Focus", "📊 Compare Both Catchments Side-by-Side"])

# --- DISSERTATION CATEGORY GROUPING ---
metric_categories = {
    "(a) Flood and Overtopping Risk": ["Rx1day", "Rx5day", "R95pTOT", "N95_Frequency"],
    "(b) Sedimentation and Structural Stress": ["R5mm", "R10mm", "SDII"],
    "(c) Storage Reliability and Water Security": ["CDD", "CWD", "PRCPTOT"]
}

selected_category = st.sidebar.radio("Analysis Metric Category (MSc Dissertation Framework):", list(metric_categories.keys()))
metrics_to_plot = metric_categories[selected_category]

y_units = {
    "Rx1day": "Precipitation (mm/day)", 
    "Rx5day": "Precipitation (mm/5-days)",
    "R95pTOT": "Total Rainfall (mm/year)", 
    "N95_Frequency": "Days/Year",
    "R5mm": "Days/Year", 
    "R10mm": "Days/Year", 
    "SDII": "Intensity (mm/wet day)",
    "CDD": "Consecutive Days", 
    "CWD": "Consecutive Days", 
    "PRCPTOT": "Total Rainfall (mm/year)"
}

metric_help_text = {
    "Rx1day": "Maximum 1-day precipitation: max{RR_t}",
    "Rx5day": "Maximum consecutive 5-day precipitation sum",
    "R95pTOT": "Annual rainfall total from days exceeding the monthly 95th percentile baseline",
    "N95_Frequency": "Annual count of days exceeding the monthly 95th percentile baseline",
    "R5mm": "Frequency of days with daily rainfall ≥ 5 mm",
    "R10mm": "Frequency of heavy rainfall days with daily rainfall ≥ 10 mm",
    "SDII": "Simple Daily Intensity Index: Mean rainfall per wet day (≥ 1 mm)",
    "CDD": "Maximum consecutive dry days (< 1 mm)",
    "CWD": "Maximum consecutive wet days (≥ 1 mm)",
    "PRCPTOT": "Total annual precipitation accumulated on wet days (≥ 1 mm)"
}

site_names = {
    "Shukkloc (Proposed System)": "Shukkloc",
    "Ricococha / Weetacocha Grid Matrix": "Ricococha_Weetacocha"
}

# --- DISSERTATION DEFINITIONS EXPANDER ---
with st.expander("📖 View MSc Dissertation Metric Equations & Technical Definitions"):
    st.markdown(r"""
    * **(a) Flood and Overtopping Risk:** Evaluates peak volume and high-percentile daily rain events to estimate spillway capacity limits.
      * $Rx1day = \max(RR_t)$, $Rx5day = \max\left(\sum_{j=t}^{t+4} RR_j\right)$
      * $R95pTOT = \sum RR_t \quad \text{for } RR_t > RR_{95, m}$, $\quad N95 = \sum I(RR_t > RR_{95, m})$
    * **(b) Sedimentation and Structural Stress:** Measures runoff initiation thresholds and storm intensity driving catchment soil erosion.
      * $R5mm = \sum I(RR_t \ge 5)$, $\quad R10mm = \sum I(RR_t \ge 10)$
      * $SDII = \frac{\sum RR_t}{N_{\text{wet}}} \quad \text{for } RR_t \ge 1 \text{ mm/day}$
    * **(c) Storage Reliability and Water Security:** Assesses dry spell durations and total effective water yield available for storage.
      * $CDD = \max(\text{consecutive days } RR_t < 1)$, $\quad CWD = \max(\text{consecutive days } RR_t \ge 1)$
      * $PRCPTOT = \sum RR_t \quad \text{for } RR_t \ge 1 \text{ mm/day}$
    """)

# --- MAIN ANALYSIS ROUTING ENGINE ---
try:
    if data_source == "📤 Upload Custom Annual Metrics File":
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if "year" in df.columns:
                df.rename(columns={"year": "Year"}, inplace=True)
            df = df[df["Year"] >= start_year]
                
            if "Year" not in df.columns or not any(m in df.columns for m in metrics_to_plot):
                st.error("❌ Format Mismatch! Please verify your uploaded spreadsheet contains a 'Year' column.")
            else:
                st.success("✅ Custom time series read successfully!")
                cols = st.columns(len(metrics_to_plot))
                for i, metric in enumerate(metrics_to_plot):
                    with cols[i]:
                        if metric in df.columns:
                            st.metric(label=f"Mean ({metric})", value=f"{df[metric].mean():.2f}", delta=f"Max: {df[metric].max():.1f}", help=metric_help_text.get(metric, ""))
                            
                st.markdown("---")
                for metric in metrics_to_plot:
                    if metric in df.columns and df[metric].notnull().sum() > 2:
                        x, y = df["Year"].values, df[metric].values
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        is_significant, _, _, slope_val, mk_summary = calculate_trend_summary(y)
                        line_color = "red" if is_significant else "white"
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Annual Metric Value", line=dict(color="#2ca02c")))
                        fig.add_trace(go.Scatter(
                            x=x, y=p(x), mode="lines", 
                            name=f"Trend Line (Slope: {slope_val:+.2f})", 
                            line=dict(color=line_color, width=2, dash="dash")
                        ))
                        
                        fig.update_layout(
                            title=f"Temporal Dynamics for Custom Array: {metric} ({start_year}–End)",
                            xaxis_title="Year",
                            yaxis_title=y_units.get(metric, "Scale Value"),
                            yaxis=dict(rangemode="tozero"),
                            height=380,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**Trend Context (Mann-Kendall):** {mk_summary}")
        else:
            st.warning("📥 Awaiting file upload.")
            
    else:
        df_v2 = load_processed_data(v2_data_path)
        df_v2 = df_v2[df_v2["Year"] >= start_year]
        
        df_v3 = load_processed_data(v3_data_path)
        df_v3 = df_v3[df_v3["Year"] >= start_year]
        
        if "Compare" in view_mode:
            st.subheader(f"📊 Cross-Catchment Comparative Analytics: {selected_category} ({start_year}–End)")
            
            for metric in metrics_to_plot:
                fig = go.Figure()
                
                if "Overlay" in version_choice:
                    for s_label, s_key in site_names.items():
                        d_v2_s = df_v2[df_v2["site"] == s_key]
                        d_v3_s = df_v3[df_v3["site"] == s_key]
                        
                        if metric in d_v2_s.columns and len(d_v2_s) > 0:
                            fig.add_trace(go.Scatter(x=d_v2_s["Year"], y=d_v2_s[metric], mode="lines+markers", name=f"{s_key} (v2)", line=dict(width=1.5, dash="dot")))
                        if metric in d_v3_s.columns and len(d_v3_s) > 0:
                            fig.add_trace(go.Scatter(x=d_v3_s["Year"], y=d_v3_s[metric], mode="lines+markers", name=f"{s_key} (v3)", line=dict(width=2)))
                else:
                    active_df = df_v3 if "v3" in version_choice else df_v2
                    df_shuk = active_df[active_df["site"] == "Shukkloc"]
                    df_rico = active_df[active_df["site"] == "Ricococha_Weetacocha"]
                    
                    if metric in df_shuk.columns and metric in df_rico.columns:
                        fig.add_trace(go.Scatter(x=df_shuk["Year"], y=df_shuk[metric], mode="lines+markers", name="Shukkloc System", line=dict(color="#1f77b4", width=2.5)))
                        fig.add_trace(go.Scatter(x=df_rico["Year"], y=df_rico[metric], mode="lines+markers", name="Ricococha Grid Matrix", line=dict(color="#ff7f0e", width=2.5)))
                        
                        if len(df_shuk) > 2 and len(df_rico) > 2:
                            x_shuk, y_shuk = df_shuk["Year"].values, df_shuk[metric].values
                            _, _, _, slope_shuk, mk_shuk = calculate_trend_summary(y_shuk)
                            p_shuk = np.poly1d(np.polyfit(x_shuk, y_shuk, 1))
                            fig.add_trace(go.Scatter(x=x_shuk, y=p_shuk(x_shuk), mode="lines", name="Shukkloc Trend", line=dict(color="#1f77b4", width=1.5, dash="dash")))
                            
                            x_rico, y_rico = df_rico["Year"].values, df_rico[metric].values
                            _, _, _, slope_rico, mk_rico = calculate_trend_summary(y_rico)
                            p_rico = np.poly1d(np.polyfit(x_rico, y_rico, 1))
                            fig.add_trace(go.Scatter(x=x_rico, y=p_rico(x_rico), mode="lines", name="Ricococha Trend", line=dict(color="#ff7f0e", width=1.5, dash="dash")))

                fig.update_layout(
                    title=f"Comparative Dynamics for {metric} ({start_year}–End Window)",
                    xaxis_title="Year",
                    yaxis_title=y_units.get(metric, "Scale Value"),
                    yaxis=dict(rangemode="tozero"),
                    height=420,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        else:
            selected_site_label = st.sidebar.selectbox("Select Target Location:", list(site_names.keys()))
            site_key = site_names[selected_site_label]
            
            st.subheader(f"📊 Summary Indicators: {selected_site_label} — {selected_category}")
            
            if "Overlay" in version_choice:
                d_v2 = df_v2[df_v2["site"] == site_key]
                d_v3 = df_v3[df_v3["site"] == site_key]
                
                cols = st.columns(len(metrics_to_plot))
                for i, metric in enumerate(metrics_to_plot):
                    with cols[i]:
                        val_v3 = f"{d_v3[metric].mean():.1f}" if metric in d_v3.columns and len(d_v3) > 0 else "N/A"
                        val_v2 = f"{d_v2[metric].mean():.1f}" if metric in d_v2.columns and len(d_v2) > 0 else "N/A"
                        st.metric(label=f"Mean ({metric})", value=f"v3: {val_v3}", delta=f"v2: {val_v2}", help=metric_help_text.get(metric, ""))
                st.markdown("---")
                
                for metric in metrics_to_plot:
                    fig = go.Figure()
                    
                    if metric in d_v2.columns and metric in d_v3.columns and len(d_v2) > 2 and len(d_v3) > 2:
                        fig.add_trace(go.Scatter(x=d_v2["Year"], y=d_v2[metric], mode="lines+markers", name="PISCO v2", line=dict(color="#1f77b4", width=2)))
                        fig.add_trace(go.Scatter(x=d_v3["Year"], y=d_v3[metric], mode="lines+markers", name="PISCO v3", line=dict(color="#ff7f0e", width=2)))
                        
                        x_v2, y_v2 = d_v2["Year"].values, d_v2[metric].values
                        _, _, _, slope_v2, mk_v2 = calculate_trend_summary(y_v2)
                        p_v2 = np.poly1d(np.polyfit(x_v2, y_v2, 1))
                        fig.add_trace(go.Scatter(x=x_v2, y=p_v2(x_v2), mode="lines", name=f"v2 Trend ({slope_v2:+.2f})", line=dict(color="#1f77b4", width=1.5, dash="dash")))
                        
                        x_v3, y_v3 = d_v3["Year"].values, d_v3[metric].values
                        _, _, _, slope_v3, mk_v3 = calculate_trend_summary(y_v3)
                        p_v3 = np.poly1d(np.polyfit(x_v3, y_v3, 1))
                        fig.add_trace(go.Scatter(x=x_v3, y=p_v3(x_v3), mode="lines", name=f"v3 Trend ({slope_v3:+.2f})", line=dict(color="#ff7f0e", width=1.5, dash="dash")))
                        
                        fig.update_layout(
                            title=f"{selected_site_label} — {metric} (PISCO v2 vs v3 Overlay | {start_year}–End)",
                            xaxis_title="Year",
                            yaxis_title=y_units.get(metric, "Scale Value"),
                            yaxis=dict(rangemode="tozero"),
                            height=400,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**PISCO v2 Mann-Kendall ({start_year}–End):** {mk_v2}")
                        st.markdown(f"**PISCO v3 Mann-Kendall ({start_year}–End):** {mk_v3}")
                        st.markdown("---")
            else:
                active_df = df_v3 if "v3" in version_choice else df_v2
                df = active_df[active_df["site"] == site_key]
                
                cols = st.columns(len(metrics_to_plot))
                for i, metric in enumerate(metrics_to_plot):
                    with cols[i]:
                        if metric in df.columns and len(df) > 0:
                            st.metric(label=f"Mean ({metric})", value=f"{df[metric].mean():.2f}", delta=f"Max: {df[metric].max():.1f}", help=metric_help_text.get(metric, ""))
                        else:
                            st.metric(label=f"Mean ({metric})", value="N/A")
                        
                st.markdown("---")
                for metric in metrics_to_plot:
                    if metric in df.columns and len(df) > 2:
                        x, y = df["Year"].values, df[metric].values
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        
                        is_significant, _, _, slope_val, mk_summary = calculate_trend_summary(y)
                        line_color = "red" if is_significant else "white"
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Annual Metric Value", line=dict(color="#2ca02c")))
                        fig.add_trace(go.Scatter(
                            x=x, y=p(x), mode="lines", 
                            name=f"Trend Line (Slope: {slope_val:+.2f})", 
                            line=dict(color=line_color, width=2, dash="dash")
                        ))
                        
                        fig.update_layout(
                            title=f"Temporal Dynamics for {metric} ({version_choice} | {start_year}–End)",
                            xaxis_title="Year",
                            yaxis_title=y_units.get(metric, "Scale Value"),
                            yaxis=dict(rangemode="tozero"),
                            height=380,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"**Statistical Significance Summary (Mann-Kendall):** {mk_summary}")
                        st.markdown("---")

    if data_source == "Use Baseline Research Sites":
        st.markdown("---")
        st.subheader("🌧️ Raw Daily Precipitation Temporal Explorer")
        
        selected_raw_label = st.selectbox("Select Catchment to Inspect Raw Data:", list(site_names.keys()))
        raw_site_key = site_names[selected_raw_label]
        
        time_frequency = st.segmented_control("Select Visualization Time Step Resolution:", ["Daily", "Monthly Total", "Annual Total"], default="Daily")
        
        raw_df_all = load_raw_daily_data(raw_file_path)
        raw_df_all = raw_df_all[raw_df_all["Date"].dt.year >= start_year]
        
        fig_raw = go.Figure()
        
        if "Overlay" in version_choice:
            v_keys = ["v2", "v3"]
        else:
            v_keys = ["v3"] if "v3" in version_choice else ["v2"]
            
        for vk in v_keys:
            r_df = raw_df_all[(raw_df_all["site"] == raw_site_key) & (raw_df_all["version"] == f"PISCO_{vk}")].copy()
            r_df.set_index("Date", inplace=True)
            
            if len(r_df) > 0:
                if time_frequency == "Daily":
                    plot_df = r_df.copy()
                    chart_title = "Continuous Daily Precipitation Time Series"
                    y_label = "Precipitation (mm/day)"
                elif time_frequency == "Monthly Total":
                    plot_df = r_df.resample("ME")["Precipitation_mm"].sum().to_frame()
                    chart_title = "Aggregated Cumulative Monthly Rainfall Blocks"
                    y_label = "Precipitation (mm/month)"
                else:
                    plot_df = r_df.resample("YE")["Precipitation_mm"].sum().to_frame()
                    chart_title = "Aggregated Cumulative Historical Annual Totals"
                    y_label = "Precipitation (mm/year)"
                    
                line_c = "#1f77b4" if vk == "v2" else "#ff7f0e"
                fig_raw.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Precipitation_mm"], mode="lines", name=f"PISCO {vk}", line=dict(color=line_c, width=1.5)))
            
        fig_raw.update_layout(
            title=f"{chart_title} — {selected_raw_label} ({start_year}–End Window)",
            xaxis_title="Timeline Horizon",
            yaxis_title=y_label,
            yaxis=dict(rangemode="tozero"),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_raw, use_container_width=True)

    # --- ARCHIVAL REPLICATION DECK & OPEN SCIENCE EXPORTS ---
    st.markdown("---")
    st.subheader("🔓 Open Science & Replication Data Deck")
    
    tab1, tab2 = st.tabs(["📊 Processed Annual Metrics Data", "🌧️ Raw Base Rainfall Series"])
    
    with tab1:
        if "v3" in version_choice:
            out_df = df_v3
        elif "v2" in version_choice:
            out_df = df_v2
        else:
            out_df = pd.concat([df_v2, df_v3], ignore_index=True)
            
        st.dataframe(out_df.set_index("Year") if "Year" in out_df.columns else out_df, use_container_width=True)
        st.download_button(
            label="📥 Download Active Processed Annual Data Matrix (CSV)",
            data=out_df.to_csv(index=False).encode('utf-8'),
            file_name=f"pisco_annual_metrics_{start_year}_end.csv",
            mime="text/csv"
        )
        
    with tab2:
        st.dataframe(raw_df_all.head(500), use_container_width=True)
        st.download_button(
            label="📥 Download Unaggregated Raw Daily Rainfall Dataset (CSV)",
            data=raw_df_all.to_csv(index=False).encode('utf-8'),
            file_name=f"raw_daily_precipitation_{start_year}_end.csv",
            mime="text/csv"
        )

except FileNotFoundError:
    st.error("❌ Component files missing. Please ensure PISCO_v2_annual_metrics.csv, PISCO_v3_annual_metrics.csv, and raw_daily_precipitation_all_sites.csv exist in /data and /raw_daily directories.")

# --- RESEARCHER PROFILE & DATA ATTRIBUTION FOOTER ---
st.markdown("---")
col_profile, col_attribution = st.columns([1, 1])

with col_profile:
    st.markdown("### 👤 About the Researcher & Supervision")
    st.markdown(
        "**Cakra Mahasurya Atmojo Pamungkas**\n\n"
        "This interactive dashboard is an ongoing research framework continuing from a Master of Science "
        "dissertation project submitted to the **Department of Meteorology, University of Reading**.\n\n"
        "🎖️ *Graduated with Distinction; Awarded Best Dissertation*\n\n"
        "🎓 **Academic Supervision:**\n"
        "* **Supervisor:** Prof. Joy Singarayer – Department of Meteorology, University of Reading\n\n"
        "The project is developed in alignment with modern hydro-climatological monitoring initiatives to assist heritage engineering "
        "and climate adaptation efforts in high-altitude mountain environments."
    )
    st.markdown("✉️ **Direct Contact:** [cakra.pamungkas@bmkg.go.id](mailto:cakra.pamungkas@bmkg.go.id)")
    st.markdown("🔗 **Professional Profile:** [LinkedIn Portfolio](https://www.linkedin.com/in/cakra-mahasurya-atmojo-pamungkas)")

with col_attribution:
    st.markdown("### 🗄️ Primary Dataset Attribution")
    st.markdown(
        "The empirical baseline forcing values and daily time-series matrices driving this application are derived from:\n\n"
        "> **Dataset 1 (PISCOp v3.0):** High-resolution grids of rainfall for Peru - PISCOp v3.0 dataset\n"
        "> 🔗 [https://doi.org/10.6084/m9.figshare.32411886](https://doi.org/10.6084/m9.figshare.32411886)\n\n"
        "> **Dataset 2 (PISCOp v2.1):** High-resolution gridded rainfall dataset for Peru - PISCOp v2.1 update\n"
        "> 🔗 [https://doi.org/10.6084/m9.figshare.21127423](https://doi.org/10.6084/m9.figshare.21127423)\n\n"
        "> **Authors:** Leonardo Gutierrez & Waldo Lavado-Casimiro\n\n"
        "Special acknowledgment is extended to the authors and the *Servicio Nacional de Meteorología e Hidrología del Perú (SENAMHI)* for "
        "maintaining transparent, open-science replication pipelines via the Figshare data ecosystem."
    )