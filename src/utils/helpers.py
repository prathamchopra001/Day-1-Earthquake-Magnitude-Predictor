import os
import yaml
from datetime import datetime, timedelta
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the config file
    
    Returns:
        Configuration dictionary
    """
    # Handle relative paths from project root
    if not os.path.isabs(config_path):
        project_root = get_project_root()
        config_path = os.path.join(project_root, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Absolute path to project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: utils -> src -> project_root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    return project_root


def get_date_range(days_back: int = 30) -> tuple:
    """
    Get start and end dates for API queries.
    
    Args:
        days_back: Number of days to look back
    
    Returns:
        Tuple of (start_date, end_date) as ISO format strings
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    
    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def unix_to_datetime(unix_ms: int) -> datetime:
    """
    Convert Unix timestamp (milliseconds) to datetime.
    
    Args:
        unix_ms: Unix timestamp in milliseconds
    
    Returns:
        datetime object
    """
    return datetime.utcfromtimestamp(unix_ms / 1000)


def datetime_to_unix(dt: datetime) -> int:
    """
    Convert datetime to Unix timestamp (milliseconds).
    
    Args:
        dt: datetime object
    
    Returns:
        Unix timestamp in milliseconds
    """
    return int(dt.timestamp() * 1000)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of first point (degrees)
        lat2, lon2: Coordinates of second point (degrees)
    
    Returns:
        Distance in kilometers
    """
    import math
    
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (math.sin(delta_lat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c