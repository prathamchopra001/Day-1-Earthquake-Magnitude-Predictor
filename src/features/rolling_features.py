import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.helpers import haversine_distance, load_config
from src.utils.logger import setup_logger

logger = setup_logger('rolling_features')


def compute_rolling_features(
    event: Dict[str, Any],
    historical_events: List[Dict[str, Any]],
    radius_km: float = 200.0,
    windows_days: List[int] = [7, 30],
    significant_threshold: float = 4.0
) -> Dict[str, Any]:

    event_time = event['time']
    event_lat = event['latitude']
    event_lon = event['longitude']
    
    features = {}
    
    # Filter events that occurred BEFORE this event
    prior_events = [e for e in historical_events if e['time'] < event_time]
    
    # Find events within spatial radius
    nearby_events = []
    for e in prior_events:
        distance = haversine_distance(
            event_lat, event_lon,
            e['latitude'], e['longitude']
        )
        if distance <= radius_km:
            nearby_events.append({
                **e,
                'distance_km': distance
            })
    
    # Compute features for each time window
    for window_days in windows_days:
        window_ms = window_days * 24 * 60 * 60 * 1000
        cutoff_time = event_time - window_ms
        
        # Events in this time window and spatial radius
        window_events = [e for e in nearby_events if e['time'] >= cutoff_time]
        
        # Extract magnitudes
        magnitudes = [e['magnitude'] for e in window_events if e['magnitude'] is not None]
        
        # Compute statistics
        count = len(window_events)
        mean_mag = statistics.mean(magnitudes) if magnitudes else 0.0
        max_mag = max(magnitudes) if magnitudes else 0.0
        std_mag = statistics.stdev(magnitudes) if len(magnitudes) > 1 else 0.0
        
        # Store with window suffix
        features[f'rolling_count_{window_days}d'] = count
        features[f'rolling_mean_mag_{window_days}d'] = round(mean_mag, 4)
        features[f'rolling_max_mag_{window_days}d'] = round(max_mag, 4)
        features[f'rolling_std_mag_{window_days}d'] = round(std_mag, 4)
    
    # Days since last significant event in region
    significant_events = [
        e for e in nearby_events
        if e['magnitude'] is not None and e['magnitude'] >= significant_threshold
    ]
    
    if significant_events:
        # Sort by time descending to get most recent
        significant_events.sort(key=lambda x: x['time'], reverse=True)
        last_significant = significant_events[0]
        days_since = (event_time - last_significant['time']) / (1000 * 60 * 60 * 24)
        features['days_since_last_significant'] = round(days_since, 4)
    else:
        # No significant events found, use large default
        features['days_since_last_significant'] = 365.0
    
    # Local seismic density (events per 1000 sq km in last 30 days)
    # Area of circle = π * r²
    area_sq_km = 3.14159 * (radius_km ** 2)
    count_30d = features.get('rolling_count_30d', 0)
    density = (count_30d / area_sq_km) * 1000  # per 1000 sq km
    features['local_seismic_density'] = round(density, 6)
    
    return features


def compute_quality_features(event: Dict[str, Any]) -> Dict[str, Any]:
    features = {}
    
    # Number of stations (typical range: 0-500)
    nst = event.get('nst')
    if nst is not None:
        features['nst_norm'] = min(nst / 500.0, 1.0)
    else:
        features['nst_norm'] = 0.0
    
    # Azimuthal gap (0-360 degrees, lower is better)
    gap = event.get('gap')
    if gap is not None:
        features['gap_norm'] = gap / 360.0
    else:
        features['gap_norm'] = 1.0  # Worst case if missing
    
    # Distance to nearest station (typical range: 0-10 degrees)
    dmin = event.get('dmin')
    if dmin is not None:
        features['dmin_norm'] = min(dmin / 10.0, 1.0)
    else:
        features['dmin_norm'] = 1.0
    
    # RMS travel time residual (typical range: 0-2 seconds)
    rms = event.get('rms')
    if rms is not None:
        features['rms_norm'] = min(rms / 2.0, 1.0)
    else:
        features['rms_norm'] = 0.5  # Neutral if missing
    
    return features


def compute_all_rolling_features(
    events: List[Dict[str, Any]],
    radius_km: float = 200.0,
    windows_days: List[int] = [7, 30],
    significant_threshold: float = 4.0,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    # Sort by time to ensure correct ordering
    events_sorted = sorted(events, key=lambda x: x['time'])
    
    all_features = []
    total = len(events_sorted)
    
    for i, event in enumerate(events_sorted):
        # Historical events are all events before current
        historical = events_sorted[:i]
        
        # Compute rolling features
        rolling = compute_rolling_features(
            event,
            historical,
            radius_km=radius_km,
            windows_days=windows_days,
            significant_threshold=significant_threshold
        )
        
        # Compute quality features
        quality = compute_quality_features(event)
        
        # Combine with event ID
        features = {
            'event_id': event['id'],
            **rolling,
            **quality
        }
        
        all_features.append(features)
        
        # Progress logging
        if show_progress and (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{total} events")
    
    if show_progress:
        logger.info(f"Completed rolling features for {total} events")
    
    return all_features


# Test function
def test_rolling_features():
    
    # Create sample events
    base_time = 1764189673525  # Base timestamp
    day_ms = 24 * 60 * 60 * 1000
    
    sample_events = [
        # Historical events (before target)
        {
            'id': 'event1',
            'time': base_time - 2 * day_ms,
            'latitude': -18.0,
            'longitude': -178.0,
            'depth': 100,
            'magnitude': 3.5,
            'nst': 25,
            'gap': 120,
            'dmin': 1.5,
            'rms': 0.5
        },
        {
            'id': 'event2',
            'time': base_time - 5 * day_ms,
            'latitude': -17.5,
            'longitude': -178.5,
            'depth': 200,
            'magnitude': 4.2,
            'nst': 40,
            'gap': 90,
            'dmin': 1.0,
            'rms': 0.3
        },
        {
            'id': 'event3',
            'time': base_time - 15 * day_ms,
            'latitude': -18.5,
            'longitude': -177.5,
            'depth': 150,
            'magnitude': 5.1,
            'nst': 60,
            'gap': 60,
            'dmin': 0.5,
            'rms': 0.2
        },
        # Target event
        {
            'id': 'target',
            'time': base_time,
            'latitude': -17.9244,
            'longitude': -178.371,
            'depth': 565,
            'magnitude': 4.6,
            'nst': 39,
            'gap': 104,
            'dmin': 2.594,
            'rms': 0.68
        },
    ]
    
    # Compute features for all events
    all_features = compute_all_rolling_features(
        sample_events,
        radius_km=200,
        windows_days=[7, 30],
        show_progress=False
    )
    
    # Show features for target event
    target_features = all_features[-1]
    
    print("Rolling Features for Target Event:")
    print("-" * 40)
    print(f"Event ID: {target_features['event_id']}")
    print()
    print("7-Day Window:")
    print(f"  Count: {target_features['rolling_count_7d']}")
    print(f"  Mean Mag: {target_features['rolling_mean_mag_7d']}")
    print(f"  Max Mag: {target_features['rolling_max_mag_7d']}")
    print(f"  Std Mag: {target_features['rolling_std_mag_7d']}")
    print()
    print("30-Day Window:")
    print(f"  Count: {target_features['rolling_count_30d']}")
    print(f"  Mean Mag: {target_features['rolling_mean_mag_30d']}")
    print(f"  Max Mag: {target_features['rolling_max_mag_30d']}")
    print(f"  Std Mag: {target_features['rolling_std_mag_30d']}")
    print()
    print(f"Days Since Last Significant: {target_features['days_since_last_significant']}")
    print(f"Local Seismic Density: {target_features['local_seismic_density']}")
    print()
    print("Quality Features:")
    print(f"  NST norm: {target_features['nst_norm']}")
    print(f"  Gap norm: {target_features['gap_norm']}")
    print(f"  Dmin norm: {target_features['dmin_norm']}")
    print(f"  RMS norm: {target_features['rms_norm']}")
    
    return all_features


if __name__ == "__main__":
    test_rolling_features()