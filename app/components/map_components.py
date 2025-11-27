import folium
from folium.plugins import MarkerCluster
import pandas as pd
from typing import Optional, Tuple
import streamlit as st
from streamlit_folium import st_folium


def create_folium_map(
    earthquakes_df: Optional[pd.DataFrame] = None,
    center: Tuple[float, float] = (0, 0),
    zoom: int = 2,
    selected_location: Optional[Tuple[float, float]] = None
) -> folium.Map:
    
    # Create base map
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='cartodbpositron'
    )
    
    # Add earthquake markers if data provided
    if earthquakes_df is not None and not earthquakes_df.empty:
        # Create marker cluster
        marker_cluster = MarkerCluster(name='Earthquakes')
        
        for _, row in earthquakes_df.iterrows():
            # Color based on magnitude
            mag = row['magnitude']
            if mag < 3:
                color = 'green'
            elif mag < 4:
                color = 'yellow'
            elif mag < 5:
                color = 'orange'
            else:
                color = 'red'
            
            # Create popup content
            popup_html = f"""
            <div style="width: 200px;">
                <h4 style="margin: 0;">M {mag:.1f}</h4>
                <p style="margin: 5px 0;">{row.get('place', 'Unknown')}</p>
                <p style="margin: 5px 0; font-size: 0.9em;">
                    Depth: {row.get('depth', 'N/A'):.1f} km<br>
                    Lat: {row['latitude']:.3f}<br>
                    Lon: {row['longitude']:.3f}
                </p>
            </div>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=mag * 2,
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=1
            ).add_to(marker_cluster)
        
        marker_cluster.add_to(m)
    
    # Add selected location marker
    if selected_location:
        folium.Marker(
            location=selected_location,
            popup='Selected Location',
            icon=folium.Icon(color='blue', icon='star', prefix='fa')
        ).add_to(m)
    
    # Add click handler marker (will be updated by user clicks)
    m.add_child(folium.LatLngPopup())
    
    return m


def get_map_click_location(map_data: dict) -> Optional[Tuple[float, float]]:
    
    if map_data and map_data.get('last_clicked'):
        click = map_data['last_clicked']
        return (click['lat'], click['lng'])
    return None


def render_interactive_map(
    earthquakes_df: Optional[pd.DataFrame] = None,
    selected_location: Optional[Tuple[float, float]] = None
) -> Optional[Tuple[float, float]]:
    
    # Create map centered on selected location or default
    center = selected_location if selected_location else (0, 0)
    
    m = create_folium_map(
        earthquakes_df=earthquakes_df,
        center=center,
        zoom=3,
        selected_location=selected_location
    )
    
    # Render map and get click data
    map_data = st_folium(
        m,
        width=None,
        height=500,
        returned_objects=['last_clicked']
    )
    
    # Return clicked location
    return get_map_click_location(map_data)


# Magnitude color scale legend
def get_magnitude_legend_html():
    """Generate HTML for magnitude color legend."""
    return """
    <div style="display: flex; justify-content: center; gap: 20px; margin: 10px 0;">
        <span>🟢 M < 3</span>
        <span>🟡 M 3-4</span>
        <span>🟠 M 4-5</span>
        <span>🔴 M 5+</span>
    </div>
    """