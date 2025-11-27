import math
from typing import Dict, Any, Optional, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.helpers import haversine_distance

SEISMIC_ZONES = {
    'ring_of_fire_west': {'lat': 35.0, 'lon': 140.0, 'name': 'Japan/Philippines'},
    'ring_of_fire_east': {'lat': -33.0, 'lon': -70.0, 'name': 'Chile/Peru'},
    'ring_of_fire_north': {'lat': 61.0, 'lon': -150.0, 'name': 'Alaska'},
    'indonesia': {'lat': -5.0, 'lon': 120.0, 'name': 'Indonesia'},
    'himalaya': {'lat': 28.0, 'lon': 84.0, 'name': 'Himalaya'},
    'mediterranean': {'lat': 38.0, 'lon': 20.0, 'name': 'Mediterranean'},
    'mid_atlantic': {'lat': 0.0, 'lon': -30.0, 'name': 'Mid-Atlantic Ridge'},
    'california': {'lat': 36.0, 'lon': -120.0, 'name': 'San Andreas'},
}


def normalize_latitude(lat: float) -> float:
    return lat / 90.0


def normalize_longitude(lon: float) -> float:
    return lon / 180.0


def normalize_depth(depth: float, max_depth: float = 700.0) -> float:
    if depth < 0:
        depth = 0
    return min(depth / max_depth, 1.0)


def get_depth_category(depth: float) -> str:
    if depth < 70:
        return 'shallow'
    elif depth < 300:
        return 'intermediate'
    else:
        return 'deep'


def find_nearest_seismic_zone(lat: float, lon: float) -> Tuple[str, float]:
    min_distance = float('inf')
    nearest_zone = None
    
    for zone_id, zone_info in SEISMIC_ZONES.items():
        distance = haversine_distance(lat, lon, zone_info['lat'], zone_info['lon'])
        if distance < min_distance:
            min_distance = distance
            nearest_zone = zone_id
    
    return nearest_zone, min_distance


def extract_spatial_features(
    latitude: float,
    longitude: float,
    depth: float
) -> Dict[str, Any]:
    # Normalized coordinates
    lat_norm = normalize_latitude(latitude)
    lon_norm = normalize_longitude(longitude)
    depth_norm = normalize_depth(depth)
    
    # Depth category
    depth_category = get_depth_category(depth)
    
    # Nearest seismic zone
    nearest_zone, zone_distance = find_nearest_seismic_zone(latitude, longitude)
    
    # Normalized zone distance (assuming max relevant distance ~5000km)
    zone_distance_norm = min(zone_distance / 5000.0, 1.0)
    
    # Cartesian coordinates (useful for some ML algorithms)
    # Convert lat/lon to 3D cartesian on unit sphere
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    
    cart_x = math.cos(lat_rad) * math.cos(lon_rad)
    cart_y = math.cos(lat_rad) * math.sin(lon_rad)
    cart_z = math.sin(lat_rad)
    
    return {
        # Raw values
        'latitude': latitude,
        'longitude': longitude,
        'depth': depth,
        
        # Normalized values
        'lat_norm': round(lat_norm, 6),
        'lon_norm': round(lon_norm, 6),
        'depth_norm': round(depth_norm, 6),
        
        # Depth category
        'depth_category': depth_category,
        
        # Seismic zone info
        'nearest_seismic_zone': nearest_zone,
        'zone_distance_km': round(zone_distance, 2),
        'zone_distance_norm': round(zone_distance_norm, 6),
        
        # Cartesian coordinates
        'cart_x': round(cart_x, 6),
        'cart_y': round(cart_y, 6),
        'cart_z': round(cart_z, 6),
    }


def calculate_distance_to_point(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    
    return haversine_distance(lat1, lon1, lat2, lon2)


def degrees_to_km(degrees: float, latitude: float = 0) -> float:
    
    km_per_degree = 111.0 * math.cos(math.radians(latitude))
    return degrees * km_per_degree


def km_to_degrees(km: float, latitude: float = 0) -> float:
    
    km_per_degree = 111.0 * math.cos(math.radians(latitude))
    if km_per_degree == 0:
        return 0
    return km / km_per_degree


# Test function
def test_spatial_features():
    test_lat = -17.9244
    test_lon = -178.371
    test_depth = 565.297
    
    features = extract_spatial_features(test_lat, test_lon, test_depth)
    
    print("Spatial Features:")
    print("-" * 40)
    print(f"  Location: ({test_lat}, {test_lon})")
    print(f"  Depth: {test_depth} km")
    print()
    print("Normalized:")
    print(f"  Lat norm: {features['lat_norm']}")
    print(f"  Lon norm: {features['lon_norm']}")
    print(f"  Depth norm: {features['depth_norm']}")
    print()
    print(f"Depth Category: {features['depth_category']}")
    print()
    print(f"Nearest Seismic Zone: {features['nearest_seismic_zone']}")
    print(f"  Distance: {features['zone_distance_km']} km")
    print()
    print("Cartesian Coordinates:")
    print(f"  X: {features['cart_x']}")
    print(f"  Y: {features['cart_y']}")
    print(f"  Z: {features['cart_z']}")
    
    return features


if __name__ == "__main__":
    test_spatial_features()