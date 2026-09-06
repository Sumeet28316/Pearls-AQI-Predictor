"""
Streamlit Dashboard for Karachi AQI Predictor
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from src.database import MongoDBHandler
from src.model_registry import ModelRegistry
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — "Monitoring Station" theme: dark instrument panel,
# flat readout colors, condensed display type + monospace data type.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-base: #11161a;
        --bg-panel: #171f25;
        --bg-panel-alt: #1c252c;
        --line: rgba(236,231,216,0.10);
        --ink: #ece7d8;
        --ink-dim: #8b968f;
        --amber: #e3b148;
        --teal: #4f9d8a;
        --good: #4f9d8a;
        --moderate: #d9a83f;
        --sensitive: #d98a3f;
        --unhealthy: #c15a3a;
        --very-unhealthy: #8b5a99;
        --hazardous: #7a2f3a;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--ink) !important;
    }

    h1, h2, h3, h4, h5, h6,
    p, span, li, label, div,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stText"],
    [data-testid="stCaptionContainer"],
    .stMarkdown, .stCaption, .stRadio label, .stRadio span {
        color: var(--ink) !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Barlow Condensed', sans-serif;
    }

    .sub-header, .pred-label, .pred-meta, .station-label,
    [data-testid="stCaptionContainer"], .stCaption {
        color: var(--ink-dim) !important;
    }

    /* App shell — faint instrument-panel grid texture */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-base);
        background-image:
            linear-gradient(var(--line) 1px, transparent 1px),
            linear-gradient(90deg, var(--line) 1px, transparent 1px);
        background-size: 42px 42px;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stAppViewBlockContainer"] {
        padding-top: 1.5rem;
    }

    /* Station title */
    .station-strap {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.2rem;
        padding-left: 2rem;
    }

    .live-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: var(--amber);
        box-shadow: 0 0 0 0 rgba(227,177,72,0.6);
        animation: pulse 2.4s infinite;
    }

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(227,177,72,0.55); }
        70%  { box-shadow: 0 0 0 9px rgba(227,177,72,0); }
        100% { box-shadow: 0 0 0 0 rgba(227,177,72,0); }
    }

    .station-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--amber);
        letter-spacing: 1px;
    }

    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: var(--ink);
        text-align: left;
        margin-bottom: 0.2rem;
        padding-left: 2rem;
        line-height: 1.05;
    }

    .sub-header {
        text-align: left;
        color: var(--ink-dim);
        font-size: 0.92rem;
        margin-bottom: 1.4rem;
        padding-left: 2rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Prediction readout panels */
    .prediction-card {
        background: var(--bg-panel);
        padding: 1.4rem 1.5rem 1.2rem 1.5rem;
        border-radius: 4px;
        border: 1px solid var(--line);
        border-left: 3px solid var(--ink-dim);
        color: var(--ink);
        text-align: left;
    }

    .pred-label {
        font-size: 0.78rem;
        color: var(--ink-dim);
        margin-bottom: 0.6rem;
        letter-spacing: 0.5px;
        font-family: 'IBM Plex Mono', monospace;
    }

    .pred-value {
        font-size: 2.6rem;
        font-weight: 600;
        margin: 0.1rem 0 0.6rem 0;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: -1px;
    }

    .pred-category {
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.25rem 0.7rem;
        border-radius: 3px;
        display: inline-block;
        margin-top: 0.2rem;
        font-family: 'IBM Plex Mono', monospace;
        border: 1px solid currentColor;
    }

    .tick-scale {
        width: 100%;
        height: 5px;
        background: var(--bg-panel-alt);
        border-radius: 3px;
        margin-top: 1rem;
        overflow: hidden;
    }

    .tick-fill {
        height: 100%;
        border-radius: 3px;
    }

    /* Flat category colors — left border, digit color, tag color all match */
    .aqi-good           { border-left-color: var(--good); }
    .aqi-good .pred-value       { color: var(--good); }
    .aqi-good .pred-category    { color: var(--good); }
    .aqi-good .tick-fill        { background: var(--good); }

    .aqi-satisfactory           { border-left-color: var(--moderate); }
    .aqi-satisfactory .pred-value    { color: var(--moderate); }
    .aqi-satisfactory .pred-category { color: var(--moderate); }
    .aqi-satisfactory .tick-fill     { background: var(--moderate); }

    .aqi-moderate           { border-left-color: var(--sensitive); }
    .aqi-moderate .pred-value    { color: var(--sensitive); }
    .aqi-moderate .pred-category { color: var(--sensitive); }
    .aqi-moderate .tick-fill     { background: var(--sensitive); }

    .aqi-poor           { border-left-color: var(--unhealthy); }
    .aqi-poor .pred-value    { color: var(--unhealthy); }
    .aqi-poor .pred-category { color: var(--unhealthy); }
    .aqi-poor .tick-fill     { background: var(--unhealthy); }

    .aqi-verypoor           { border-left-color: var(--very-unhealthy); }
    .aqi-verypoor .pred-value    { color: var(--very-unhealthy); }
    .aqi-verypoor .pred-category { color: var(--very-unhealthy); }
    .aqi-verypoor .tick-fill     { background: var(--very-unhealthy); }

    .aqi-severe           { border-left-color: var(--hazardous); }
    .aqi-severe .pred-value    { color: var(--hazardous); }
    .aqi-severe .pred-category { color: var(--hazardous); }
    .aqi-severe .tick-fill     { background: var(--hazardous); }

    .pred-meta {
        margin-top: 1rem;
        font-size: 0.78rem;
        color: var(--ink-dim);
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.5;
    }

    /* Section headers — panel divider, not a card */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--ink);
        margin: 2.2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--line);
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-panel);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace;
    }

    [data-testid="stSidebar"] h3 {
        font-family: 'Barlow Condensed', sans-serif !important;
        font-size: 1.1rem;
        letter-spacing: 0.3px;
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.4rem;
    }

    /* Buttons */
    .stButton>button {
        background: var(--bg-panel-alt);
        color: var(--amber);
        border: 1px solid var(--amber);
        border-radius: 4px;
        padding: 0.5rem 1.6rem;
        font-weight: 500;
        font-family: 'IBM Plex Mono', monospace;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        background: var(--amber);
        color: var(--bg-base);
    }

    /* Alerts / info-warning-error boxes */
    [data-testid="stNotification"], .stAlert {
        background: var(--bg-panel) !important;
        border: 1px solid var(--line) !important;
        border-left: 3px solid var(--amber) !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace;
        border-radius: 4px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 0.9rem 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--ink-dim) !important;
        font-family: 'IBM Plex Mono', monospace;
    }

    [data-testid="stMetricValue"] {
        color: var(--amber) !important;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 4px;
    }

    /* Divider */
    hr {
        border-color: var(--line) !important;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom Footer */
    .custom-footer {
        text-align: left;
        padding: 1.6rem 0.2rem;
        color: var(--ink-dim);
        font-size: 0.78rem;
        margin-top: 2.5rem;
        border-top: 1px solid var(--line);
        font-family: 'IBM Plex Mono', monospace;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize
@st.cache_resource(ttl=3600)  # Cache for 1 hour then reload
def init_registry():
    """Initialize model registry"""
    registry = ModelRegistry()
    registry.load_all_models()
    return registry

@st.cache_resource
def init_database():
    """Initialize database connection"""
    return MongoDBHandler()

def get_epa_aqi_and_style(pm25):
    """Calculates US EPA AQI from raw PM2.5 and maps to your original CSS classes"""
    c = float(np.floor(pm25 * 10) / 10)
    
    if c <= 12.0:
        aqi = ((50 - 0) / (12.0 - 0.0)) * (c - 0.0) + 0
        return int(round(aqi)), "Good", "aqi-good", "😊"
    elif c <= 35.4:
        aqi = ((100 - 51) / (35.4 - 12.1)) * (c - 12.1) + 51
        return int(round(aqi)), "Moderate", "aqi-satisfactory", "🙂"
    elif c <= 55.4:
        aqi = ((150 - 101) / (55.4 - 35.5)) * (c - 35.5) + 101
        return int(round(aqi)), "Unhealthy for Sensitive Groups", "aqi-moderate", "😐"
    elif c <= 150.4:
        aqi = ((200 - 151) / (150.4 - 55.5)) * (c - 55.5) + 151
        return int(round(aqi)), "Unhealthy", "aqi-poor", "😷"
    elif c <= 250.4:
        aqi = ((300 - 201) / (250.4 - 150.5)) * (c - 150.5) + 201
        return int(round(aqi)), "Very Unhealthy", "aqi-verypoor", "😨"
    else:
        c = min(c, 500.4)
        aqi = ((500 - 301) / (500.4 - 250.5)) * (c - 250.5) + 301
        return int(round(aqi)), "Hazardous", "aqi-severe", "☠️"

def main():
    # Station header with live indicator + AQI scale legend
    header_col, scale_col = st.columns([3, 1])
    
    with header_col:
        st.markdown("""
        <div class="station-strap">
            <div class="live-dot"></div>
            <span class="station-label">LIVE READOUT · UPDATED HOURLY</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">Karachi Air Quality Station</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Forecasting PM2.5 up to 72 hours ahead from live atmospheric data.</p>', unsafe_allow_html=True)
    
    with scale_col:
        # Kept your exact HTML layout, just updated the scale numbers to reflect EPA 0-500
        st.markdown("""
        <div style='padding: 0.9rem 1rem; margin-top: 0.4rem; background: #171f25; border: 1px solid rgba(236,231,216,0.10); border-radius: 4px;'>
            <h4 style='font-size: 0.85rem; margin-bottom: 0.6rem; color: #ece7d8; font-family: "Barlow Condensed", sans-serif; letter-spacing: 0.3px;'>AQI Scale</h4>
            <div style='font-size: 0.72rem; font-family: "IBM Plex Mono", monospace; color: #ece7d8;'>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#4f9d8a;display:inline-block;border-radius:2px;"></span> Good (0–50)</div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#d9a83f;display:inline-block;border-radius:2px;"></span> Moderate (51–100)</div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#d98a3f;display:inline-block;border-radius:2px;"></span> Sensitive (101–150)</div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#c15a3a;display:inline-block;border-radius:2px;"></span> Unhealthy (151–200)</div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#8b5a99;display:inline-block;border-radius:2px;"></span> Very Unhealthy (201–300)</div>
            <div style='display:flex; align-items:center; gap:0.5rem; margin: 0.3rem 0;'><span style="width:9px;height:9px;background:#7a2f3a;display:inline-block;border-radius:2px;"></span> Hazardous (301+)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Controls")
        
        # Model selection
        st.markdown("**Model Selection**")
        model_option = st.radio(
            "Choose prediction model:",
            ["Best Model (Auto)", "RandomForest", "XGBoost", "LightGBM"],
            label_visibility="collapsed"
        )
        
        selected_model = None if model_option == "Best Model (Auto)" else model_option
        
        st.divider()
        
        # About section
        st.markdown("### About This App")
        st.markdown("""
        This dashboard predicts **Air Quality Index (AQI)** for Karachi using:
        
        - Machine learning models
        - 78+ days of historical data
        - Hourly updates via GitHub Actions
        - Live APIs (Open-Meteo & OpenWeather)
        
        **Prediction Horizons:**
        - 24h: Tomorrow's AQI
        - 48h: Day after tomorrow
        - 72h: 3 days ahead
        """)
        
        st.divider()
        
        # Tech Stack
        st.markdown("### Tech Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **ML**: Scikit-learn, XGBoost, LightGBM
        - **Database**: MongoDB Atlas
        - **Automation**: GitHub Actions
        - **APIs**: Open-Meteo, OpenWeather
        """)
        
        # Refresh button
        st.markdown("")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content
    try:
        # Load registry and database
        registry = init_registry()
        db = init_database()
        
        # Get predictions
        with st.spinner("Generating predictions..."):
            predictions = registry.predict_multi_horizon(model_name=selected_model)
        
        if not predictions:
            st.error("Unable to generate predictions. Please check the model and data.")
            return
        
        # Display model info - Updated to use Regression metrics
        metrics = predictions.get('model_metrics', {})
        st.info(f"Model: {predictions['model_used']}  ·  RMSE: {metrics.get('test_rmse', 0):.2f}  ·  MAE: {metrics.get('test_mae', 0):.2f}  ·  R²: {metrics.get('test_r2', 0):.2f}")
        
        # Predictions Section
        st.markdown('<h2 class="section-header">Future AQI Predictions</h2>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        # 24h prediction
        if '24h_ahead' in predictions:
            pred_24h = predictions['24h_ahead']
            aqi_24h, category_24h, class_24h, emoji_24h = get_epa_aqi_and_style(pred_24h['prediction'])
            fill_24h = min(aqi_24h / 500 * 100, 100)
            
            with cols[0]:
                st.markdown(f"""
                <div class="prediction-card {class_24h}">
                    <div class="pred-label">Tomorrow</div>
                    <div class="pred-value">{aqi_24h}</div>
                    <div class="pred-category">{category_24h}</div>
                    <div class="tick-scale"><div class="tick-fill" style="width:{fill_24h:.0f}%;"></div></div>
                    <p class="pred-meta">{pred_24h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_24h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 48h prediction
        if '48h_ahead' in predictions:
            pred_48h = predictions['48h_ahead']
            aqi_48h, category_48h, class_48h, emoji_48h = get_epa_aqi_and_style(pred_48h['prediction'])
            fill_48h = min(aqi_48h / 500 * 100, 100)
            
            with cols[1]:
                st.markdown(f"""
                <div class="prediction-card {class_48h}">
                    <div class="pred-label">Day After Tomorrow</div>
                    <div class="pred-value">{aqi_48h}</div>
                    <div class="pred-category">{category_48h}</div>
                    <div class="tick-scale"><div class="tick-fill" style="width:{fill_48h:.0f}%;"></div></div>
                    <p class="pred-meta">{pred_48h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_48h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 72h prediction
        if '72h_ahead' in predictions:
            pred_72h = predictions['72h_ahead']
            aqi_72h, category_72h, class_72h, emoji_72h = get_epa_aqi_and_style(pred_72h['prediction'])
            fill_72h = min(aqi_72h / 500 * 100, 100)
            
            with cols[2]:
                st.markdown(f"""
                <div class="prediction-card {class_72h}">
                    <div class="pred-label">3 Days Ahead</div>
                    <div class="pred-value">{aqi_72h}</div>
                    <div class="pred-category">{category_72h}</div>
                    <div class="tick-scale"><div class="tick-fill" style="width:{fill_72h:.0f}%;"></div></div>
                    <p class="pred-meta">{pred_72h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_72h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("")
        st.markdown("")
        
        # Historical Data Section
        st.markdown('<h2 class="section-header">Historical AQI Trend</h2>', unsafe_allow_html=True)
        
        # Get historical data
        df_history = db.get_latest_features(n_hours=168)  # Last 7 days
        
        if not df_history.empty and 'pm2_5' in df_history.columns:
            df_history = df_history.sort_values('datetime')
            
            # Convert raw PM2.5 to AQI for the chart
            df_history['calculated_aqi'] = df_history['pm2_5'].apply(lambda x: get_epa_aqi_and_style(x)[0])
            
            # Plot historical AQI
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df_history['datetime'],
                y=df_history['calculated_aqi'],
                mode='lines+markers',
                name='Historical AQI',
                line=dict(color='#4f9d8a', width=2),
                marker=dict(size=4, color='#4f9d8a')
            ))
            
            # Add prediction points
            pred_times = []
            pred_values = []
            
            if '24h_ahead' in predictions:
                pred_times.append(predictions['24h_ahead']['prediction_time'])
                pred_values.append(get_epa_aqi_and_style(predictions['24h_ahead']['prediction'])[0])
            if '48h_ahead' in predictions:
                pred_times.append(predictions['48h_ahead']['prediction_time'])
                pred_values.append(get_epa_aqi_and_style(predictions['48h_ahead']['prediction'])[0])
            if '72h_ahead' in predictions:
                pred_times.append(predictions['72h_ahead']['prediction_time'])
                pred_values.append(get_epa_aqi_and_style(predictions['72h_ahead']['prediction'])[0])
            
            if pred_times:
                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_values,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=11, color='#e3b148', symbol='diamond', line=dict(width=1, color='#11161a'))
                ))
            
            fig.update_layout(
                title={
                    'text': "AQI — Last 7 Days + Forecast",
                    'font': {'size': 18, 'color': '#ece7d8', 'family': 'Barlow Condensed'}
                },
                xaxis_title="Date & Time",
                yaxis_title="AQI Level (0-500 Scale)",
                hovermode='x unified',
                height=440,
                plot_bgcolor='#171f25',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'IBM Plex Mono', 'color': '#ece7d8'},
                xaxis=dict(gridcolor='rgba(236,231,216,0.08)', zerolinecolor='rgba(236,231,216,0.08)'),
                yaxis=dict(gridcolor='rgba(236,231,216,0.08)', zerolinecolor='rgba(236,231,216,0.08)'),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font={'color': '#ece7d8'}
                )
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No historical PM2.5 data available")
        
        st.divider()
        
        # Model Comparison Section
        st.markdown('<h2 class="section-header">Model Performance Comparison</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Get all models from registry - Updated to RMSE
            if registry.model_metadata:
                model_data = []
                for model_name, metadata in registry.model_metadata.items():
                    model_data.append({
                        'Model': model_name,
                        'RMSE': f"{metadata['metrics'].get('test_rmse', 0):.2f}",
                        'MAE': f"{metadata['metrics'].get('test_mae', 0):.2f}",
                        'R² Score': f"{metadata['metrics'].get('test_r2', 0):.2f}",
                        'Best': '★' if metadata.get('is_best', False) else ''
                    })
                
                df_models = pd.DataFrame(model_data)
                
                # Sort logically by lowest error
                df_models['RMSE_val'] = df_models['RMSE'].astype(float)
                df_models = df_models.sort_values('RMSE_val', ascending=True).drop('RMSE_val', axis=1)
                
                st.dataframe(
                    df_models,
                    hide_index=True,
                    width='stretch'
                )
        
        with col2:
            # Error comparison chart
            if registry.model_metadata:
                err_data = []
                for model_name, metadata in registry.model_metadata.items():
                    err_data.append({
                        'Model': model_name,
                        'RMSE': metadata['metrics'].get('test_rmse', 0),
                        'Best': metadata.get('is_best', False)
                    })
                
                fig_acc = go.Figure(data=[
                    go.Bar(
                        x=[m['Model'] for m in err_data],
                        y=[m['RMSE'] for m in err_data],
                        marker_color=['#e3b148' if m.get('Best') else '#4f9d8a' for m in err_data]
                    )
                ])
                
                fig_acc.update_layout(
                    title={
                        'text': "Model Error (Lower is Better)",
                        'font': {'size': 15, 'color': '#ece7d8', 'family': 'Barlow Condensed'}
                    },
                    xaxis_title="Model",
                    yaxis_title="RMSE (PM2.5)",
                    height=300,
                    plot_bgcolor='#171f25',
                    paper_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    font={'family': 'IBM Plex Mono', 'color': '#ece7d8'},
                    xaxis=dict(gridcolor='rgba(236,231,216,0.08)'),
                    yaxis=dict(gridcolor='rgba(236,231,216,0.08)')
                )
                
                st.plotly_chart(fig_acc, use_container_width=True)
        
        st.divider()
        
        # SHAP Analysis Section
        st.markdown('<h2 class="section-header">Model Interpretability (SHAP Analysis)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: #171f25; border: 1px solid rgba(236,231,216,0.10); border-left: 3px solid #e3b148; padding: 0.9rem 1.1rem; border-radius: 4px; margin-bottom: 1rem;'>
            <p style='color: #ece7d8; margin: 0; font-family: "IBM Plex Mono", monospace; font-size: 0.85rem;'>SHAP (SHapley Additive exPlanations) shows which features are most important for predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Use the actual model that was used for predictions
            actual_model_name = predictions.get('model_used', 'LightGBM')
            model_for_shap = registry.get_model(actual_model_name)
            df_latest = db.get_latest_features(n_hours=100)
            
            if model_for_shap and not df_latest.empty:
                # Prepare features
                df_latest_sorted = df_latest.sort_values('datetime').reset_index(drop=True)
                X_recent = df_latest_sorted[registry.feature_columns].iloc[-50:]  # Last 50 hours
                X_scaled = registry.scaler.transform(X_recent)
                
                # Convert back to DataFrame with feature names for SHAP
                X_scaled_df = pd.DataFrame(X_scaled, columns=registry.feature_columns)
                
                shap_col1, shap_col2 = st.columns(2)
                
                with shap_col1:
                    st.markdown("#### Global Feature Importance")
                    st.caption("Which features matter most overall for predictions?")
                    
                    with st.spinner("Computing SHAP values..."):
                        # Create SHAP explainer
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_values = explainer.shap_values(X_scaled_df)
                        
                        # Handle regression output shape gracefully
                        if isinstance(shap_values, list):
                            shap_array = np.array(shap_values) 
                            shap_values_combined = np.abs(shap_array).mean(axis=(0, 1)) 
                        else:
                            shap_values_combined = np.abs(shap_values).mean(axis=0) 
                        
                        mean_shap_values = shap_values_combined.flatten()
                        
                        # Create feature importance dataframe
                        feature_importance = []
                        for i, feat in enumerate(registry.feature_columns):
                            feature_importance.append({
                                'feature': feat,
                                'importance': float(mean_shap_values[i])
                            })
                        
                        mean_shap = pd.DataFrame(feature_importance).sort_values('importance', ascending=True).tail(15)
                        
                        # Create horizontal bar chart
                        fig_shap = go.Figure(go.Bar(
                            x=mean_shap['importance'],
                            y=mean_shap['feature'],
                            orientation='h',
                            marker=dict(
                                color=mean_shap['importance'],
                                colorscale=[[0, '#1c252c'], [0.5, '#4f9d8a'], [1, '#e3b148']],
                                showscale=True,
                                colorbar=dict(title="Impact", tickfont=dict(color='#ece7d8'))
                            )
                        ))
                        
                        fig_shap.update_layout(
                            title="Top 15 Most Important Features",
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="",
                            height=500,
                            plot_bgcolor='#171f25',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'family': 'IBM Plex Mono', 'color': '#ece7d8'},
                            xaxis=dict(gridcolor='rgba(236,231,216,0.08)'),
                            yaxis=dict(gridcolor='rgba(236,231,216,0.08)')
                        )
                        
                        st.plotly_chart(fig_shap, use_container_width=True)
                
                with shap_col2:
                    st.markdown("#### Individual Prediction Explanation")
                    st.caption("Why did the model predict this specific PM2.5 value?")
                    
                    # Explain the most recent prediction (24h ahead input)
                    if len(df_latest_sorted) >= 72:
                        input_72h = df_latest_sorted[registry.feature_columns].iloc[-72:-71]
                        X_72h_scaled = registry.scaler.transform(input_72h)
                        X_72h_scaled_df = pd.DataFrame(X_72h_scaled, columns=registry.feature_columns)
                        
                        # Get SHAP values for this single prediction
                        shap_single = explainer.shap_values(X_72h_scaled_df)
                        
                        if isinstance(shap_single, list):
                            shap_single_values = shap_single[0][0]
                        else:
                            shap_single_values = shap_single[0]
                        
                        # Ensure we have a 1D array
                        if isinstance(shap_single_values, np.ndarray):
                            shap_single_values = shap_single_values.flatten()
                        
                        # Create waterfall-style explanation
                        feature_contrib_list = []
                        for i, feat in enumerate(registry.feature_columns):
                            shap_val = shap_single_values[i]
                            
                            while isinstance(shap_val, np.ndarray):
                                if shap_val.size == 1:
                                    shap_val = shap_val.item()
                                else:
                                    shap_val = shap_val.mean()
                            
                            feature_contrib_list.append({
                                'feature': feat,
                                'shap': float(shap_val)
                            })
                        
                        feature_contrib = pd.DataFrame(feature_contrib_list).sort_values('shap', key=abs, ascending=True).tail(10)
                        
                        shap_values_list = feature_contrib['shap'].tolist()
                        feature_names_list = feature_contrib['feature'].tolist()
                        
                        fig_waterfall = go.Figure(go.Bar(
                            x=shap_values_list,
                            y=feature_names_list,
                            orientation='h',
                            marker=dict(
                                color=shap_values_list,
                                colorscale=[[0, '#4f9d8a'], [0.5, '#8b968f'], [1, '#c15a3a']],
                                showscale=True,
                                colorbar=dict(title="Effect", tickfont=dict(color='#ece7d8'))
                            ),
                            text=[f"{v:+.3f}" for v in shap_values_list],
                            textposition='outside'
                        ))
                        
                        fig_waterfall.update_layout(
                            title="Top 10 Features Affecting Tomorrow's Prediction",
                            xaxis_title="SHAP Value (Impact on prediction)",
                            yaxis_title="",
                            height=500,
                            plot_bgcolor='#171f25',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'family': 'IBM Plex Mono', 'color': '#ece7d8'},
                            xaxis=dict(gridcolor='rgba(236,231,216,0.08)'),
                            yaxis=dict(gridcolor='rgba(236,231,216,0.08)')
                        )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                        st.info("Positive values push AQI higher · Negative values push AQI lower")
                    else:
                        st.warning("Need at least 72 hours of data for individual prediction analysis")
            
            else:
                st.warning("SHAP analysis requires model and data to be available")
        
        except Exception as e:
            st.error(f"SHAP analysis failed: {str(e)}")
            st.caption("Note: SHAP works best with tree-based models (RandomForest, XGBoost, LightGBM)")
        
        st.divider()
        
        # Statistics - Updated to use PM2.5 converted to AQI
        st.markdown('<h2 class="section-header">Dataset Statistics</h2>', unsafe_allow_html=True)
        
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.metric("Total Records", f"{len(df_history):,}")
        
        with stat_cols[1]:
            if not df_history.empty and 'calculated_aqi' in df_history.columns:
                st.metric("Avg AQI", f"{df_history['calculated_aqi'].mean():.0f}")
            else:
                st.metric("Avg AQI", "N/A")
        
        with stat_cols[2]:
            if not df_history.empty and 'calculated_aqi' in df_history.columns:
                st.metric("Max AQI", f"{df_history['calculated_aqi'].max():.0f}")
            else:
                st.metric("Max AQI", "N/A")
        
        with stat_cols[3]:
            if not df_history.empty and 'calculated_aqi' in df_history.columns:
                st.metric("Min AQI", f"{df_history['calculated_aqi'].min():.0f}")
            else:
                st.metric("Min AQI", "N/A")
        
        # Custom Footer
        st.markdown("""
        <div class="custom-footer">
            <p><strong>Karachi AQI Predictor</strong> — Powered by Machine Learning</p>
            <p>Data updated hourly via GitHub Actions. Predictions based on 78+ days of historical data.</p>
            <p>Models: RandomForest, XGBoost, LightGBM. Database: MongoDB Atlas.</p>
            <p style="margin-top: 0.8rem;">© 2026. Built with Streamlit & Python.</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
