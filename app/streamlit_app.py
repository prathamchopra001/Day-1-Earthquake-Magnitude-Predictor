"""
Earthquake Magnitude Predictor - Streamlit Application
Layout: 3 columns - Inputs | Map+Results | LLM Text
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model.predict import EarthquakePredictor
from src.data.database import EarthquakeDatabase
from src.utils.helpers import unix_to_datetime

# Page configuration - NO SIDEBAR
st.set_page_config(
    page_title="Earthquake Magnitude Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - DARK THEME
st.markdown("""
<style>
    /* Dark/Black background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Make all text light */
    .stApp, .stMarkdown, p, span, label, .stCaption {
        color: #fafafa !important;
    }
    
    /* Input fields dark */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    
    /* Slider dark */
    .stSlider > div > div {
        background-color: #262730;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {display: none;}
    
    /* Title */
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    /* Input section labels */
    .input-label {
        font-weight: bold;
        color: #333;
        margin-bottom: 0.3rem;
        font-size: 0.9rem;
    }
    
    /* Result boxes */
    .prediction-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-box h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: bold;
    }
    
    .range-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .confidence-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .activity-box {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        border-radius: 12px;
        padding: 1rem;
        color: #fafafa;
        text-align: center;
        height: 100%;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* LLM Text box - small and simple */
    .llm-box {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        font-size: 1rem;
        line-height: 1.5;
        color: #fafafa;
    }
    
    /* Input container - dark */
    .input-container {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Reduce padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Section headers - light text */
    .section-header {
        font-size: 1rem;
        font-weight: bold;
        color: #fafafa;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the trained model."""
    try:
        predictor = EarthquakePredictor()
        predictor.load()
        return predictor, None
    except FileNotFoundError as e:
        return None, str(e)


@st.cache_data(ttl=300)
def get_recent_earthquakes(days: int = 30, min_magnitude: float = 2.5):
    """Get recent earthquakes from database."""
    try:
        db = EarthquakeDatabase()
        events = db.get_all_events(min_magnitude=min_magnitude)
        db.close()
        
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        df['datetime'] = df['time'].apply(lambda x: unix_to_datetime(x))
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df['datetime'] >= cutoff]
        
        return df.sort_values('datetime', ascending=False)
    except Exception as e:
        return pd.DataFrame()


def get_magnitude_description(mag):
    """Get description of magnitude."""
    if mag < 2.5:
        return "Very Minor", "Usually not felt"
    elif mag < 4.0:
        return "Minor", "Felt like a truck passing"
    elif mag < 5.0:
        return "Light", "Rattles windows"
    elif mag < 6.0:
        return "Moderate", "Can damage buildings"
    elif mag < 7.0:
        return "Strong", "Destructive"
    elif mag < 8.0:
        return "Major", "Serious damage"
    else:
        return "Great", "Devastating"


def check_ollama_status():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def call_ollama(prompt, model="llama3.1:8b"):
    """Call Ollama API to generate text."""
    import requests
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 200  # Increased for 5-6 sentences
                }
            },
            timeout=30  # Increased timeout for longer response
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            return None
    except Exception as e:
        return None


def generate_llm_text(result, latitude, longitude, depth):
    """Generate 5-6 sentence analysis using Ollama LLM."""
    mag = result['magnitude']
    std = result['std']
    ci_lower = result['ci_lower']
    ci_upper = result['ci_upper']
    context = result.get('context', {})
    
    severity, desc = get_magnitude_description(mag)
    count_7d = context.get('rolling_count_7d', 0)
    count_30d = context.get('rolling_count_30d', 0)
    
    # Depth type
    if depth < 70:
        depth_type = "shallow"
        depth_impact = "Shallow earthquakes typically cause more surface shaking and damage."
    elif depth < 300:
        depth_type = "intermediate"
        depth_impact = "Intermediate depth earthquakes can still be felt over wide areas."
    else:
        depth_type = "deep"
        depth_impact = "Deep earthquakes generally cause less surface damage."
    
    # Activity level
    if count_7d > 10:
        activity = "very active"
    elif count_7d > 3:
        activity = "moderately active"
    else:
        activity = "relatively quiet"
    
    # Confidence level
    if std < 0.3:
        confidence = "high confidence"
    elif std < 0.6:
        confidence = "moderate confidence"
    else:
        confidence = "lower confidence"
    
    # Create detailed prompt for Ollama
    prompt = f"""Write a 5-6 sentence summary about this earthquake prediction for a general audience. Be informative but easy to understand.

Location: {latitude:.2f}°, {longitude:.2f}°
Predicted Magnitude: M{mag:.1f} ({severity})
Possible Range: M{ci_lower:.1f} to M{ci_upper:.1f} (95% confidence)
Depth: {depth}km ({depth_type})
Model Confidence: {confidence} (±{std:.2f})
Recent Activity: {count_7d} earthquakes in past 7 days, {count_30d} in past 30 days
Area Status: {activity}

Cover these points in 5-6 sentences:
1. The predicted magnitude and what it means
2. The uncertainty range and confidence level
3. How depth affects potential impact
4. Recent seismic activity in the region
5. General safety context (educational, not a warning)

Write in a clear, informative tone. Do not use bullet points. Just write 5-6 flowing sentences:"""

    # Try to call Ollama
    llm_response = call_ollama(prompt)
    
    if llm_response:
        # Clean up response
        clean = llm_response.strip()
        # Remove any leading/trailing quotes if present
        if clean.startswith('"') and clean.endswith('"'):
            clean = clean[1:-1]
        return clean
    
    # If Ollama is not available, show error message
    return "⚠️ <b>Ollama is not running.</b><br><br>To enable AI summaries, start Ollama with:<br><code>ollama serve</code><br><br>Then pull a model:<br><code>ollama pull llama3.2</code>"


def make_prediction(predictor, latitude, longitude, depth):
    """Make prediction and store in session state."""
    try:
        result = predictor.predict(
            latitude=latitude,
            longitude=longitude,
            depth=float(depth),
            confidence_level=0.95
        )
        st.session_state['last_prediction'] = result
        st.session_state['prediction_location'] = (latitude, longitude, depth)
        return result
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


def main():
    """Main application."""
    
    # ========== TITLE ==========
    st.markdown('<div class="main-title">🌍 EARTHQUAKE MAGNITUDE PREDICTOR</div>', unsafe_allow_html=True)
    
    # Load model
    predictor, error = load_predictor()
    
    if error:
        st.error(f"⚠️ Model not loaded! Run `python setup.py` first.")
        return
    
    # Initialize session state
    if 'latitude' not in st.session_state:
        st.session_state.latitude = 35.6762
    if 'longitude' not in st.session_state:
        st.session_state.longitude = 139.6503
    if 'depth' not in st.session_state:
        st.session_state.depth = 50
    
    # ========== 3 COLUMN LAYOUT ==========
    left_col, middle_col, right_col = st.columns([1.2, 2, 1.5])
    
    # ========== LEFT COLUMN: INPUTS ==========
    with left_col:
        st.markdown('<p class="section-header">📍 SEARCH PLACE</p>', unsafe_allow_html=True)
        address = st.text_input(
            "Search",
            placeholder="e.g., Tokyo, Japan",
            label_visibility="collapsed"
        )
        
        if address:
            try:
                import requests
                geocode_url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json&limit=1"
                headers = {'User-Agent': 'EarthquakePredictor/1.0'}
                response = requests.get(geocode_url, headers=headers, timeout=5)
                if response.status_code == 200 and response.json():
                    result = response.json()[0]
                    st.session_state.latitude = float(result['lat'])
                    st.session_state.longitude = float(result['lon'])
                    st.success("📍 Found!")
            except:
                pass
        
        st.markdown('<p class="section-header">🎯 ENTER COORDINATES</p>', unsafe_allow_html=True)
        
        new_lat = st.number_input(
            "LATITUDE",
            min_value=-90.0,
            max_value=90.0,
            value=float(st.session_state.latitude),
            step=0.1,
            format="%.4f"
        )
        
        new_lon = st.number_input(
            "LONGITUDE",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.longitude),
            step=0.1,
            format="%.4f"
        )
        
        if new_lat != st.session_state.latitude or new_lon != st.session_state.longitude:
            st.session_state.latitude = new_lat
            st.session_state.longitude = new_lon
        
        st.markdown('<p class="section-header">🌍 PICK RECENT ACTIVITY</p>', unsafe_allow_html=True)
        recent_df = get_recent_earthquakes(days=7, min_magnitude=4.0)
        
        if not recent_df.empty:
            options = ["-- Select --"] + recent_df.head(10).apply(
                lambda x: f"M{x['magnitude']:.1f} | {str(x['place'])[:20]}",
                axis=1
            ).tolist()
            
            selected = st.selectbox(
                "Recent",
                options,
                label_visibility="collapsed"
            )
            
            if selected != "-- Select --":
                idx = options.index(selected) - 1
                st.session_state.latitude = float(recent_df.iloc[idx]['latitude'])
                st.session_state.longitude = float(recent_df.iloc[idx]['longitude'])
                st.session_state.depth = int(recent_df.iloc[idx]['depth'])
        else:
            st.caption("No recent M4+ earthquakes")
        
        st.markdown('<p class="section-header">📏 EARTHQUAKE DEPTH</p>', unsafe_allow_html=True)
        depth = st.slider(
            "Depth",
            min_value=0,
            max_value=700,
            value=int(st.session_state.depth),
            step=10,
            format="%d km",
            label_visibility="collapsed"
        )
        st.session_state.depth = depth
        
        if depth < 70:
            st.caption("🔵 Shallow")
        elif depth < 300:
            st.caption("🟡 Intermediate")
        else:
            st.caption("🔴 Deep")
    
    # Get current values
    latitude = st.session_state.latitude
    longitude = st.session_state.longitude
    depth = st.session_state.depth
    
    # ========== MIDDLE COLUMN: MAP + RESULTS ==========
    with middle_col:
        # MAP
        st.markdown('<p class="section-header">🗺️ MAP VIEW</p>', unsafe_allow_html=True)
        
        recent_df_all = get_recent_earthquakes(days=30, min_magnitude=2.5)
        
        try:
            import folium
            from streamlit_folium import st_folium
            
            m = folium.Map(
                location=[latitude, longitude],
                zoom_start=3,
                tiles='cartodbpositron'
            )
            
            # Add earthquake markers
            if not recent_df_all.empty:
                for _, row in recent_df_all.iterrows():
                    mag = row['magnitude']
                    eq_lat = row['latitude']
                    eq_lon = row['longitude']
                    eq_depth = row.get('depth', 50)
                    place = str(row.get('place', 'Unknown'))[:30]
                    
                    color = 'green' if mag < 3 else 'orange' if mag < 4 else 'red' if mag < 5 else 'darkred'
                    
                    popup_text = f"M{mag:.1f}|{place}|LAT:{eq_lat:.4f}|LON:{eq_lon:.4f}|DEPTH:{eq_depth:.0f}"
                    
                    folium.CircleMarker(
                        location=[eq_lat, eq_lon],
                        radius=max(3, mag * 2),
                        popup=popup_text,
                        color=color,
                        fill=True,
                        fillOpacity=0.7
                    ).add_to(m)
            
            # Selected location marker
            folium.Marker(
                location=[latitude, longitude],
                popup=f"Selected: ({latitude:.2f}, {longitude:.2f})",
                icon=folium.Icon(color='blue', icon='star')
            ).add_to(m)
            
            map_data = st_folium(m, width=None, height=300, returned_objects=["last_clicked", "last_object_clicked_popup"])
            
            # Handle clicks
            if map_data and map_data.get("last_object_clicked_popup"):
                popup = map_data["last_object_clicked_popup"]
                if "LAT:" in popup:
                    try:
                        parts = popup.split("|")
                        lat_p = [p for p in parts if "LAT:" in p][0]
                        lon_p = [p for p in parts if "LON:" in p][0]
                        dep_p = [p for p in parts if "DEPTH:" in p][0]
                        
                        c_lat = float(lat_p.split(":")[1])
                        c_lon = float(lon_p.split(":")[1])
                        c_dep = int(float(dep_p.split(":")[1]))
                        
                        if abs(c_lat - latitude) > 0.001 or abs(c_lon - longitude) > 0.001:
                            st.session_state.latitude = c_lat
                            st.session_state.longitude = c_lon
                            st.session_state.depth = c_dep
                            if 'last_prediction' in st.session_state:
                                del st.session_state['last_prediction']
                            st.rerun()
                    except:
                        pass
            elif map_data and map_data.get("last_clicked"):
                c_lat = map_data["last_clicked"]["lat"]
                c_lon = map_data["last_clicked"]["lng"]
                
                while c_lon > 180: c_lon -= 360
                while c_lon < -180: c_lon += 360
                c_lat = max(-90, min(90, c_lat))
                
                if abs(c_lat - latitude) > 0.01 or abs(c_lon - longitude) > 0.01:
                    st.session_state.latitude = c_lat
                    st.session_state.longitude = c_lon
                    if 'last_prediction' in st.session_state:
                        del st.session_state['last_prediction']
                    st.rerun()
        
        except ImportError:
            st.warning("Install: `pip install folium streamlit-folium`")
        
        st.caption("Click map or marker to select • 🟢M<3 🟠M3-4 🔴M4-5 ⚫M5+")
        
        # RESULT BOXES (2x2 grid)
        # Auto-predict
        need_pred = 'last_prediction' not in st.session_state
        if not need_pred and 'prediction_location' in st.session_state:
            old = st.session_state['prediction_location']
            if abs(old[0]-latitude) > 0.01 or abs(old[1]-longitude) > 0.01 or old[2] != depth:
                need_pred = True
        
        if need_pred:
            result = make_prediction(predictor, latitude, longitude, depth)
        else:
            result = st.session_state.get('last_prediction')
        
        if result:
            mag = result['magnitude']
            std = result['std']
            severity, _ = get_magnitude_description(mag)
            context = result.get('context', {})
            
            # Row 1: Prediction + Possible Range
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown(f"""
                <div class="prediction-box">
                    <p style="margin:0;font-size:0.85rem;opacity:0.9;">PREDICTION RESULT</p>
                    <h2>M {mag:.1f}</h2>
                    <p style="margin:0;font-size:0.9rem;">{severity}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with r1c2:
                st.markdown(f"""
                <div class="range-box">
                    <p style="margin:0;font-size:0.85rem;opacity:0.9;">POSSIBLE RANGE</p>
                    <h3 style="margin:0.3rem 0;font-size:1.8rem;">M {result['ci_lower']:.1f} - {result['ci_upper']:.1f}</h3>
                    <p style="margin:0;font-size:0.85rem;">95% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 2: Confidence + Recent Activity
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                conf_pct = max(0, min(100, int((1 - std/2) * 100)))
                st.markdown(f"""
                <div class="confidence-box">
                    <p style="margin:0;font-size:0.85rem;opacity:0.9;">CONFIDENCE</p>
                    <h3 style="margin:0.3rem 0;font-size:2rem;">{conf_pct}%</h3>
                    <p style="margin:0;font-size:0.85rem;">±{std:.2f} std</p>
                </div>
                """, unsafe_allow_html=True)
            
            with r2c2:
                c7 = context.get('rolling_count_7d', 0)
                c30 = context.get('rolling_count_30d', 0)
                st.markdown(f"""
                <div class="activity-box">
                    <p style="margin:0;font-size:0.85rem;">RECENT ACTIVITY</p>
                    <p style="margin:0.5rem 0;font-size:1.1rem;"><strong>{c7}</strong> in 7D | <strong>{c30}</strong> in 30D</p>
                </div>
                """, unsafe_allow_html=True)
    
    # ========== RIGHT COLUMN: LLM TEXT ==========
    with right_col:
        st.markdown('<p class="section-header">💡 AI SUMMARY</p>', unsafe_allow_html=True)
        
        # Show Ollama status
        ollama_running = check_ollama_status()
        if ollama_running:
            st.markdown('<span style="color: #38ef7d; font-size: 0.8rem;">🟢 Ollama Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color: #f5576c; font-size: 0.8rem;">🔴 Ollama Offline</span>', unsafe_allow_html=True)
        
        if result:
            # Use cached LLM response if location hasn't changed
            cache_key = f"{latitude:.2f}_{longitude:.2f}_{depth}"
            
            if 'llm_cache_key' not in st.session_state or st.session_state.llm_cache_key != cache_key:
                with st.spinner("✨ Generating AI summary..."):
                    llm_text = generate_llm_text(result, latitude, longitude, depth)
                    st.session_state.llm_text = llm_text
                    st.session_state.llm_cache_key = cache_key
            else:
                llm_text = st.session_state.get('llm_text', '')
            
            st.markdown(f'<div class="llm-box">{llm_text}</div>', unsafe_allow_html=True)
            
            # Small regenerate link
            if st.button("🔄 Regenerate", key="regen"):
                st.session_state.llm_cache_key = None
                st.rerun()
        else:
            st.info("Select a location")
    
    # ========== BOTTOM: RECENT EARTHQUAKES ==========
    st.markdown("---")
    st.markdown('<p class="section-header">🌍 RECENT EARTHQUAKES WORLDWIDE</p>', unsafe_allow_html=True)
    
    if not recent_df_all.empty:
        fig = px.scatter(
            recent_df_all,
            x='datetime',
            y='magnitude',
            color='magnitude',
            size='magnitude',
            hover_name='place',
            color_continuous_scale='YlOrRd',
            size_max=10
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=10, b=30),
            xaxis_title='',
            yaxis_title='Magnitude',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Data: USGS Earthquake Hazards Program | Model: Gaussian Process Regression | ⚠️ For educational purposes only")


if __name__ == "__main__":
    main()