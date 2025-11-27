import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional,List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_date_range

logger = setup_logger('api_client')

class USGSApiClient:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.base_url = self.config['api']['base_url']
        self.default_params = self.config['api']['default_params']
        
    def build_query_url(
        self,
        starttime:Optional[str]=None,
        endtime:Optional[str]=None,
        minmagnitude:Optional[float]=None,
        maxmagnitude:Optional[float]=None,
        minlatitude:Optional[float]=None,
        maxlatitude:Optional[float]=None,
        minlongitude:Optional[float]=None,
        maxlongitude:Optional[float]=None,
        mindepth:Optional[float]=None,
        maxdepth:Optional[float]=None,
        limit:Optional[int]=None,
        Oderby:Optional[str]=None
    )-> str:
        params = {'format': self.config['api']['format']}
        
        if starttime:
            params['starttime'] = starttime
        if endtime:
            params['endtime'] = endtime
        if minmagnitude is not None:
            params['minmagnitude'] = minmagnitude
        if maxmagnitude is not None:
            params['maxmagnitude'] = maxmagnitude
        if minlatitude is not None:
            params['minlatitude'] = minlatitude
        if maxlatitude is not None:
            params['maxlatitude'] = maxlatitude
        if minlongitude is not None:
            params['minlongitude'] = minlongitude    
        if maxlongitude is not None:
            params['maxlongitude'] = maxlongitude
        if mindepth is not None:
            params['mindepth'] = mindepth
        if maxdepth is not None:
            params['maxdepth'] = maxdepth
        if limit is not None:
            params['limit'] = limit
        if Oderby is not None:
            params['orderby'] = Oderby
        
        query_params = '&'.join([f"{k}={v}" for k, v in params.items()])
        
        return f"{self.base_url}?{query_params}"
    
    def fetch_data(self,
                url: str,
                max_retries: int = 3,
                retry_delay: float=2.0,
                timeout: int=30
                ) -> Optional[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                logger.info(f"fetching data (attempt {attempt + 1}/{max_retries}) from URL: {url}")
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                logger.info(f"Data fetched successfully from URL: {url} | {data['metadata']['count']} evensts")
                return data
            
            except requests.exceptions.Timeout as e:
                logger.warning(f"Request timed out (attempt {attempt + 1})")
            
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error: {e}")
                if response.status_code == 400:
                    logger.error("Bad request - check query parameters")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                
            except ValueError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return None
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        logger.error("All retry attempts failed")
        return None
    
    def parse_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not data or 'features' not in data:
            logger.warning("No features found in response")
            return []
        
        events = []
        
        for feature in data['features']:
            try:
                properties = feature['properties']
                geometry = feature['geometry']
                coordinates = geometry['coordinates']
                
                event = {
                    # Identifiers
                    'id': feature['id'],
                    'code': properties.get('code'),
                    
                    # Location
                    'longitude': coordinates[0],
                    'latitude': coordinates[1],
                    'depth': coordinates[2] if len(coordinates) > 2 else None,
                    'place': properties.get('place'),
                    
                    # Magnitude
                    'magnitude': properties.get('mag'),
                    'mag_type': properties.get('magType'),
                    
                    # Time
                    'time': properties.get('time'),  # Unix timestamp in ms
                    'updated': properties.get('updated'),
                    
                    # Event metadata
                    'type': properties.get('type'),
                    'status': properties.get('status'),
                    'tsunami': properties.get('tsunami'),
                    'sig': properties.get('sig'),  # Significance
                    
                    # Quality metrics
                    'nst': properties.get('nst'),    # Number of stations
                    'gap': properties.get('gap'),    # Azimuthal gap
                    'dmin': properties.get('dmin'),  # Min distance to station
                    'rms': properties.get('rms'),    # Travel time residual
                    
                    # Additional
                    'felt': properties.get('felt'),
                    'cdi': properties.get('cdi'),
                    'mmi': properties.get('mmi'),
                    'alert': properties.get('alert'),
                    'net': properties.get('net'),
                }
                
                events.append(event)
                
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse feature: {e}")
                continue
        
        logger.info(f"Parsed {len(events)} events")
        return events
    
    def get_earthquakes(
        self,
        days_back: int = 30,
        min_magnitude: float = 2.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        start_date, end_date = get_date_range(days_back)
        
        url = self.build_query_url(
            starttime=start_date,
            endtime=end_date,
            minmagnitude=min_magnitude,
            **kwargs
        )
        
        logger.info(f"Querying earthquakes from {start_date} to {end_date}")
        logger.debug(f"URL: {url}")
        
        raw_data = self.fetch_data(url)
        
        if raw_data is None:
            return []
        
        return self.parse_response(raw_data)
    
def test_api_client():
    client = USGSApiClient()
    
    events = client.get_earthquakes(days_back=7, min_magnitude=4.5, limit=10, Oderby='time')
    
    print(f"\nFetched {len(events)} events:\n")
    
    for event in events[:5]:
        print(f"  {event['magnitude']:.1f} - {event['place']}")
        print(f"    Lat: {event['latitude']:.4f}, Lon: {event['longitude']:.4f}, Depth: {event['depth']:.1f} km")
        print()
    
    return events


if __name__ == "__main__":
    test_api_client()