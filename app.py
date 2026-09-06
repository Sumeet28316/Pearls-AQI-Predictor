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
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Teal & Amber Theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Header */
    .main-header {
        font-size: 3.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #0f766e 0%, #0891b2 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: left;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        text-align: left;
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }

    /* Prediction Cards */
    .prediction-card {
        background: linear-gradient(145deg, #0f766e 0%, #134e4a 100%);
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
        color: white;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .prediction-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 18px 36px rgba(15, 118, 110, 0.35);
    }

    .pred-label {
        font-size: 0.85rem;
        opacity: 0.85;
        margin-bottom: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }

    .pred-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0.4rem 0;
        font-family: 'Poppins', sans-serif;
    }

    .pred-category {
        font-size: 1rem;
        font-weight: 600;
        background: rgba(255,255,255,0.18);
        padding: 0.4rem 1.1rem;
        border-radius: 24px;
        display: inline-block;
        margin-top: 0.4rem;
        backdrop-filter: blur(4px);
    }

    /* AQI Badge Colors - warm amber/coral accent palette */
    .aqi-good { background: linear-gradient(145deg, #059669 0%, #047857 100%); }
    .aqi-satisfactory { background: linear-gradient(145deg, #ca8a04 0%, #a16207 100%); }
    .aqi-moderate { background: linear-gradient(145deg, #ea580c 0%, #c2410c 100%); }
    .aqi-poor { background: linear-gradient(145deg, #dc2626 0%, #991b1b 100%); }
    .aqi-verypoor { background: linear-gradient(145deg, #9333ea 0%, #6b21a8 100%); }
    .aqi-severe { background: linear-gradient(145deg, #881337 0%, #4c0519 100%); }

    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #134e4a;
        margin: 2.2rem 0 1.1rem 0;
        border-left: 5px solid #0d9488;
        padding-left: 1rem;
    }

    /* Stats Cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
        border-left: 4px solid #0d9488;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ecfeff 0%, #cffafe 100%);
    }

    /* Force sidebar text to be dark to contrast with the light background */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] * {
        color: #134e4a !important;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(8, 145, 178, 0.4);
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom Footer */
    .custom-footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.88rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
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
    # Modern Header with AQI Scale
    header_col, scale_col = st.columns([3, 1])
    
    with header_col:
        st.markdown('<h1 class="main-header" style="text-align: left; padding-left: 2rem;">🌍 Karachi Air Quality Predictor</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header" style="text-align: left; padding-left: 2rem;">Real-time AQI predictions powered by Machine Learning | Updated hourly with live data</p>', unsafe_allow_html=True)
    
    with scale_col:
        # Kept your exact HTML layout, just updated the scale numbers to reflect EPA 0-500
        st.markdown("""
        <div style='padding: 1rem; margin-top: 1rem; background: white; border-radius: 14px; box-shadow: 0 4px 14px rgba(15,23,42,0.06);'>
            <h4 style='font-size: 1rem; margin-bottom: 0.5rem; color: #134e4a;'>📊 AQI Scale</h4>
            <div style='font-size: 0.75rem;'>
            <div style='padding: 0.3rem 0.5rem; background: #059669; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>🟢 Good (0-50)</div>
            <div style='padding: 0.3rem 0.5rem; background: #ca8a04; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>🟡 Moderate (51-100)</div>
            <div style='padding: 0.3rem 0.5rem; background: #ea580c; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>🟠 Unhealthy/Sensitive (101-150)</div>
            <div style='padding: 0.3rem 0.5rem; background: #dc2626; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>🔴 Unhealthy (151-200)</div>
            <div style='padding: 0.3rem 0.5rem; background: #9333ea; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>🟣 Very Unhealthy (201-300)</div>
            <div style='padding: 0.3rem 0.5rem; background: #881337; color: white; border-radius: 6px; margin: 0.18rem 0; font-weight: 600;'>⚫ Hazardous (301+)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
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
        st.markdown("### 📖 About This App")
        st.markdown("""
        This dashboard predicts **Air Quality Index (AQI)** for Karachi using:
        
        - 🤖 **Machine Learning** models
        - 📊 **78+ days** of historical data
        - 🔄 **Hourly updates** via GitHub Actions
        - 🌐 **Live APIs** (Open-Meteo & OpenWeather)
        
        **Prediction Horizons:**
        - ☀️ **24h**: Tomorrow's AQI
        - 🌤️ **48h**: Day after tomorrow
        - 🌥️ **72h**: 3 days ahead
        """)
        
        st.divider()
        
        # Tech Stack
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **Frontend**: Streamlit
        - **ML**: Scikit-learn, XGBoost, LightGBM
        - **Database**: MongoDB Atlas
        - **Automation**: GitHub Actions
        - **APIs**: Open-Meteo, OpenWeather
        """)
        
        # Refresh button
        st.markdown("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
    
    # Main content
    try:
        # Load registry and database
        registry = init_registry()
        db = init_database()
        
        # Get predictions
        with st.spinner("🔮 Generating predictions..."):
            predictions = registry.predict_multi_horizon(model_name=selected_model)
        
        if not predictions:
            st.error("❌ Unable to generate predictions. Please check the model and data.")
            return
        
        # Display model info - Updated to use Regression metrics
        metrics = predictions.get('model_metrics', {})
        st.info(f"🤖 **Model:** {predictions['model_used']} | **RMSE:** {metrics.get('test_rmse', 0):.2f} | **MAE:** {metrics.get('test_mae', 0):.2f} | **R²:** {metrics.get('test_r2', 0):.2f}")
        
        # Predictions Section
        st.markdown('<h2 class="section-header">🔮 Future AQI Predictions</h2>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        # 24h prediction
        if '24h_ahead' in predictions:
            pred_24h = predictions['24h_ahead']
            aqi_24h, category_24h, class_24h, emoji_24h = get_epa_aqi_and_style(pred_24h['prediction'])
            
            with cols[0]:
                st.markdown(f"""
                <div class="prediction-card {class_24h}">
                    <div class="pred-label">Tomorrow</div>
                    <div class="pred-value">{emoji_24h} {aqi_24h}</div>
                    <div class="pred-category">{category_24h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_24h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_24h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 48h prediction
        if '48h_ahead' in predictions:
            pred_48h = predictions['48h_ahead']
            aqi_48h, category_48h, class_48h, emoji_48h = get_epa_aqi_and_style(pred_48h['prediction'])
            
            with cols[1]:
                st.markdown(f"""
                <div class="prediction-card {class_48h}">
                    <div class="pred-label">Day After Tomorrow</div>
                    <div class="pred-value">{emoji_48h} {aqi_48h}</div>
                    <div class="pred-category">{category_48h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_48h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_48h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 72h prediction
        if '72h_ahead' in predictions:
            pred_72h = predictions['72h_ahead']
            aqi_72h, category_72h, class_72h, emoji_72h = get_epa_aqi_and_style(pred_72h['prediction'])
            
            with cols[2]:
                st.markdown(f"""
                <div class="prediction-card {class_72h}">
                    <div class="pred-label">3 Days Ahead</div>
                    <div class="pred-value">{emoji_72h} {aqi_72h}</div>
                    <div class="pred-category">{category_72h}</div>
                    <p style="margin-top: 1rem; font-size: 0.85rem; opacity: 0.9;">{pred_72h['prediction_time'].strftime('%b %d, %H:%M')}<br>Raw PM2.5: {pred_72h['prediction']:.1f} µg/m³</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        st.markdown("")
        st.markdown("")
        
        # Historical Data Section
        st.markdown('<h2 class="section-header">📈 Historical AQI Trends</h2>', unsafe_allow_html=True)
        
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
                line=dict(color='#0891b2', width=2.5),
                marker=dict(size=4, color='#0f766e')
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
                    marker=dict(size=13, color='#ea580c', symbol='star', line=dict(width=1, color='#7c2d12'))
                ))
            
            fig.update_layout(
                title={
                    'text': "AQI Trend - Last 7 Days + Future Predictions",
                    'font': {'size': 20, 'color': '#134e4a', 'family': 'Poppins'}
                },
                xaxis_title="Date & Time",
                yaxis_title="AQI Level (0-500 Scale)",
                hovermode='x unified',
                height=450,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No historical PM2.5 data available")
        
        st.divider()
        
        # Model Comparison Section
        st.markdown('<h2 class="section-header">🏆 Model Performance Comparison</h2>', unsafe_allow_html=True)
        
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
                        'Best': '🥇' if metadata.get('is_best', False) else ''
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
                        marker_color=['#ea580c' if m.get('Best') else '#0891b2' for m in err_data]
                    )
                ])
                
                fig_acc.update_layout(
                    title={
                        'text': "Model Error (Lower is Better)",
                        'font': {'size': 16, 'color': '#134e4a', 'family': 'Poppins'}
                    },
                    xaxis_title="Model",
                    yaxis_title="RMSE (PM2.5)",
                    height=300,
                    template='plotly_white',
                    showlegend=False,
                    font={'family': 'Inter'}
                )
                
                st.plotly_chart(fig_acc, use_container_width=True)
        
        st.divider()
        
        # SHAP Analysis Section
        st.markdown('<h2 class="section-header">🔍 Model Interpretability (SHAP Analysis)</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background: linear-gradient(135deg, #0f766e 0%, #0891b2 100%); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;'>
            <p style='color: white; margin: 0;'><strong>🧠 Understanding Predictions:</strong> SHAP (SHapley Additive exPlanations) shows which features are most important for predictions.</p>
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
                    st.markdown("#### 🌍 Global Feature Importance")
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
                                colorscale='Teal',
                                showscale=True,
                                colorbar=dict(title="Impact")
                            )
                        ))
                        
                        fig_shap.update_layout(
                            title="Top 15 Most Important Features",
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="",
                            height=500,
                            template='plotly_white',
                            font={'family': 'Inter'}
                        )
                        
                        st.plotly_chart(fig_shap, use_container_width=True)
                
                with shap_col2:
                    st.markdown("#### 🎯 Individual Prediction Explanation")
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
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Effect")
                            ),
                            text=[f"{v:+.3f}" for v in shap_values_list],
                            textposition='outside'
                        ))
                        
                        fig_waterfall.update_layout(
                            title="Top 10 Features Affecting Tomorrow's Prediction",
                            xaxis_title="SHAP Value (Impact on prediction)",
                            yaxis_title="",
                            height=500,
                            template='plotly_white',
                            font={'family': 'Inter'}
                        )
                        
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                        
                        st.info("🔵 Positive values push AQI higher | 🟢 Negative values push AQI lower")
                    else:
                        st.warning("Need at least 72 hours of data for individual prediction analysis")
            
            else:
                st.warning("⚠️ SHAP analysis requires model and data to be available")
        
        except Exception as e:
            st.error(f"❌ SHAP analysis failed: {str(e)}")
            st.caption("Note: SHAP works best with tree-based models (RandomForest, XGBoost, LightGBM)")
        
        st.divider()
        
        # Statistics - Updated to use PM2.5 converted to AQI
        st.markdown('<h2 class="section-header">📊 Dataset Statistics</h2>', unsafe_allow_html=True)
        
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
            <p><strong>Karachi AQI Predictor</strong> | Powered by Machine Learning</p>
            <p>Data updated hourly via GitHub Actions | Predictions based on 78+ days of historical data</p>
            <p>Models: RandomForest, XGBoost, LightGBM | Database: MongoDB Atlas</p>
            <p style="font-size: 0.8rem; margin-top: 1rem;">© 2026 | Built with ❤️ using Streamlit & Python</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()
