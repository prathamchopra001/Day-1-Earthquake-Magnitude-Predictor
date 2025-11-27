import math
from datetime import datetime
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.helpers import unix_to_datetime


def extract_temporal_features(timestamp_ms: int) -> Dict[str, Any]:

    dt = unix_to_datetime(timestamp_ms)
    
    # Basic temporal components
    hour = dt.hour
    day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
    day_of_year = dt.timetuple().tm_yday
    month = dt.month
    
    # Cyclical encoding for hour (24-hour cycle)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    
    # Cyclical encoding for day of week (7-day cycle)
    dow_sin = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos = math.cos(2 * math.pi * day_of_week / 7)
    
    # Cyclical encoding for day of year (365-day cycle)
    doy_sin = math.sin(2 * math.pi * day_of_year / 365)
    doy_cos = math.cos(2 * math.pi * day_of_year / 365)
    
    # Cyclical encoding for month (12-month cycle)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    
    return {
        # Raw values
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'month': month,
        
        # Cyclical encodings
        'hour_sin': round(hour_sin, 6),
        'hour_cos': round(hour_cos, 6),
        'dow_sin': round(dow_sin, 6),
        'dow_cos': round(dow_cos, 6),
        'doy_sin': round(doy_sin, 6),
        'doy_cos': round(doy_cos, 6),
        'month_sin': round(month_sin, 6),
        'month_cos': round(month_cos, 6),
        
        # Additional
        'timestamp_ms': timestamp_ms,
        'datetime': dt.isoformat()
    }


def get_time_delta_days(timestamp_ms: int, reference_ms: int) -> float:

    delta_ms = timestamp_ms - reference_ms
    return delta_ms / (1000 * 60 * 60 * 24)


# Test function
def test_temporal_features():
    """Test temporal feature extraction."""
    # Example timestamp: 2025-11-26 14:30:00 UTC
    test_timestamp = 1764189673525
    
    features = extract_temporal_features(test_timestamp)
    
    print("Temporal Features:")
    print("-" * 40)
    print(f"  Datetime: {features['datetime']}")
    print(f"  Hour: {features['hour']}")
    print(f"  Day of Week: {features['day_of_week']} (0=Mon)")
    print(f"  Day of Year: {features['day_of_year']}")
    print(f"  Month: {features['month']}")
    print()
    print("Cyclical Encodings:")
    print(f"  Hour: sin={features['hour_sin']:.4f}, cos={features['hour_cos']:.4f}")
    print(f"  DoW:  sin={features['dow_sin']:.4f}, cos={features['dow_cos']:.4f}")
    print(f"  DoY:  sin={features['doy_sin']:.4f}, cos={features['doy_cos']:.4f}")
    
    return features


if __name__ == "__main__":
    test_temporal_features()