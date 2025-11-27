import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.utils.helpers import load_config, get_project_root

logger = setup_logger('database')


class EarthquakeDatabase:
    
    def __init__(self, db_path: Optional[str] = None):
        
        if db_path is None:
            config = load_config()
            db_path = config['data']['database_path']
        
        # Handle relative paths
        if not os.path.isabs(db_path):
            project_root = get_project_root()
            db_path = os.path.join(project_root, db_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        
    def connect(self) -> sqlite3.Connection:
        
        if self.conn is None:
            # check_same_thread=False allows connection to be used across threads
            # This is needed for Streamlit which runs in multiple threads
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info(f"Connected to database: {self.db_path}")
        return self.conn
    
    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def create_tables(self):
        conn = self.connect()
        cursor = conn.cursor()
        
        # Events table - stores raw earthquake data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                code TEXT,
                time INTEGER NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                depth REAL,
                magnitude REAL NOT NULL,
                mag_type TEXT,
                place TEXT,
                type TEXT,
                status TEXT,
                tsunami INTEGER,
                sig INTEGER,
                nst INTEGER,
                gap REAL,
                dmin REAL,
                rms REAL,
                felt INTEGER,
                cdi REAL,
                mmi REAL,
                alert TEXT,
                net TEXT,
                updated INTEGER,
                ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Features table - stores computed features for ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                event_id TEXT PRIMARY KEY,
                
                -- Temporal features
                hour INTEGER,
                day_of_week INTEGER,
                day_of_year INTEGER,
                hour_sin REAL,
                hour_cos REAL,
                dow_sin REAL,
                dow_cos REAL,
                
                -- Normalized spatial features
                lat_norm REAL,
                lon_norm REAL,
                depth_norm REAL,
                
                -- Rolling window features (7 days)
                rolling_count_7d INTEGER,
                rolling_mean_mag_7d REAL,
                rolling_max_mag_7d REAL,
                rolling_std_mag_7d REAL,
                
                -- Rolling window features (30 days)
                rolling_count_30d INTEGER,
                rolling_mean_mag_30d REAL,
                rolling_max_mag_30d REAL,
                rolling_std_mag_30d REAL,
                
                -- Other derived features
                days_since_last_significant REAL,
                local_seismic_density REAL,
                
                -- Quality features (from raw data)
                nst_norm REAL,
                gap_norm REAL,
                dmin_norm REAL,
                rms_norm REAL,
                
                computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (event_id) REFERENCES events(id)
            )
        ''')
        
        # Create indexes for efficient queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_time 
            ON events(time)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_location 
            ON events(latitude, longitude)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_magnitude 
            ON events(magnitude)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_events_time_location 
            ON events(time, latitude, longitude)
        ''')
        
        conn.commit()
        logger.info("Database tables created successfully")
    
    def insert_events(self, events: List[Dict[str, Any]]) -> int:
        
        if not events:
            return 0
        
        conn = self.connect()
        cursor = conn.cursor()
        
        insert_sql = '''
            INSERT OR REPLACE INTO events (
                id, code, time, latitude, longitude, depth, magnitude,
                mag_type, place, type, status, tsunami, sig, nst,
                gap, dmin, rms, felt, cdi, mmi, alert, net, updated
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        '''
        
        inserted = 0
        
        for event in events:
            try:
                cursor.execute(insert_sql, (
                    event.get('id'),
                    event.get('code'),
                    event.get('time'),
                    event.get('latitude'),
                    event.get('longitude'),
                    event.get('depth'),
                    event.get('magnitude'),
                    event.get('mag_type'),
                    event.get('place'),
                    event.get('type'),
                    event.get('status'),
                    event.get('tsunami'),
                    event.get('sig'),
                    event.get('nst'),
                    event.get('gap'),
                    event.get('dmin'),
                    event.get('rms'),
                    event.get('felt'),
                    event.get('cdi'),
                    event.get('mmi'),
                    event.get('alert'),
                    event.get('net'),
                    event.get('updated')
                ))
                inserted += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to insert event {event.get('id')}: {e}")
        
        conn.commit()
        logger.info(f"Inserted {inserted} events into database")
        
        return inserted
    
    def insert_features(self, features: List[Dict[str, Any]]) -> int:
        
        if not features:
            return 0
        
        conn = self.connect()
        cursor = conn.cursor()
        
        insert_sql = '''
            INSERT OR REPLACE INTO features (
                event_id, hour, day_of_week, day_of_year,
                hour_sin, hour_cos, dow_sin, dow_cos,
                lat_norm, lon_norm, depth_norm,
                rolling_count_7d, rolling_mean_mag_7d, rolling_max_mag_7d, rolling_std_mag_7d,
                rolling_count_30d, rolling_mean_mag_30d, rolling_max_mag_30d, rolling_std_mag_30d,
                days_since_last_significant, local_seismic_density,
                nst_norm, gap_norm, dmin_norm, rms_norm
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        '''
        
        inserted = 0
        
        for feat in features:
            try:
                cursor.execute(insert_sql, (
                    feat.get('event_id'),
                    feat.get('hour'),
                    feat.get('day_of_week'),
                    feat.get('day_of_year'),
                    feat.get('hour_sin'),
                    feat.get('hour_cos'),
                    feat.get('dow_sin'),
                    feat.get('dow_cos'),
                    feat.get('lat_norm'),
                    feat.get('lon_norm'),
                    feat.get('depth_norm'),
                    feat.get('rolling_count_7d'),
                    feat.get('rolling_mean_mag_7d'),
                    feat.get('rolling_max_mag_7d'),
                    feat.get('rolling_std_mag_7d'),
                    feat.get('rolling_count_30d'),
                    feat.get('rolling_mean_mag_30d'),
                    feat.get('rolling_max_mag_30d'),
                    feat.get('rolling_std_mag_30d'),
                    feat.get('days_since_last_significant'),
                    feat.get('local_seismic_density'),
                    feat.get('nst_norm'),
                    feat.get('gap_norm'),
                    feat.get('dmin_norm'),
                    feat.get('rms_norm')
                ))
                inserted += 1
            except sqlite3.Error as e:
                logger.warning(f"Failed to insert features for {feat.get('event_id')}: {e}")
        
        conn.commit()
        logger.info(f"Inserted {inserted} feature records")
        
        return inserted
    
    def get_all_events(
        self,
        min_magnitude: Optional[float] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        event_type: str = 'earthquake'
    ) -> List[Dict[str, Any]]:
        
        conn = self.connect()
        cursor = conn.cursor()
        
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type)
        
        if min_magnitude is not None:
            query += " AND magnitude >= ?"
            params.append(min_magnitude)
        
        if start_time is not None:
            query += " AND time >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND time <= ?"
            params.append(end_time)
        
        query += " ORDER BY time ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        events = [dict(row) for row in rows]
        logger.info(f"Retrieved {len(events)} events from database")
        
        return events
    
    def get_events_in_region(
        self,
        latitude: float,
        longitude: float,
        radius_degrees: float,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        
        conn = self.connect()
        cursor = conn.cursor()
        
        query = '''
            SELECT * FROM events
            WHERE latitude BETWEEN ? AND ?
            AND longitude BETWEEN ? AND ?
            AND type = 'earthquake'
        '''
        params = [
            latitude - radius_degrees,
            latitude + radius_degrees,
            longitude - radius_degrees,
            longitude + radius_degrees
        ]
        
        if start_time is not None:
            query += " AND time >= ?"
            params.append(start_time)
        
        if end_time is not None:
            query += " AND time <= ?"
            params.append(end_time)
        
        query += " ORDER BY time ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def get_events_with_features(self) -> List[Dict[str, Any]]:
        
        conn = self.connect()
        cursor = conn.cursor()
        
        query = '''
            SELECT e.*, f.*
            FROM events e
            INNER JOIN features f ON e.id = f.event_id
            WHERE e.type = 'earthquake'
            ORDER BY e.time ASC
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        results = [dict(row) for row in rows]
        logger.info(f"Retrieved {len(results)} events with features")
        
        return results
    
    def get_event_count(self) -> int:
        """Get total number of events in database."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM events")
        return cursor.fetchone()[0]
    
    def get_feature_count(self) -> int:
        """Get total number of feature records in database."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM features")
        return cursor.fetchone()[0]
    
    def get_date_range(self) -> Tuple[Optional[int], Optional[int]]:
        
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT MIN(time), MAX(time) FROM events")
        row = cursor.fetchone()
        return (row[0], row[1])
    
    def clear_tables(self):
        """Clear all data from tables (use with caution)."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM features")
        cursor.execute("DELETE FROM events")
        conn.commit()
        logger.warning("All tables cleared")


# Test function
def test_database():
    """Test database operations."""
    db = EarthquakeDatabase()
    
    # Create tables
    db.create_tables()
    
    # Test insert
    test_events = [
        {
            'id': 'test001',
            'code': 'test001',
            'time': 1764189673525,
            'latitude': -17.9244,
            'longitude': -178.371,
            'depth': 565.297,
            'magnitude': 4.6,
            'mag_type': 'mb',
            'place': 'Test Location',
            'type': 'earthquake',
            'status': 'reviewed',
            'tsunami': 0,
            'sig': 326,
            'nst': 39,
            'gap': 104,
            'dmin': 2.594,
            'rms': 0.68
        }
    ]
    
    db.insert_events(test_events)
    
    # Test retrieval
    events = db.get_all_events()
    print(f"\nTotal events in database: {db.get_event_count()}")
    
    for event in events:
        print(f"  {event['id']}: M{event['magnitude']} at {event['place']}")
    
    db.close()


if __name__ == "__main__":
    test_database()