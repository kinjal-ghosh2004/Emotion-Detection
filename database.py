"""
database.py - Local Database Management
Provides persistent storage for wellness data
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class WellnessDatabase:
    """SQLite database for wellness monitoring"""
    
    def __init__(self, db_path: str = "wellness_data.db"):
        """Initialize database"""
        self.db_path = db_path
        self.connection = None
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Main wellness data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS wellness_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    face_emotion TEXT,
                    heart_rate REAL,
                    voice_emotion TEXT,
                    stress_level REAL,
                    confidence REAL,
                    notes TEXT
                )
            """)
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration_minutes REAL,
                    avg_stress REAL,
                    avg_heart_rate REAL,
                    notes TEXT
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT 0
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Database initialized at {self.db_path}")
        
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def insert_wellness_data(self, data: dict) -> bool:
        """Insert wellness data record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO wellness_data 
                (timestamp, face_emotion, heart_rate, voice_emotion, stress_level, confidence, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get('timestamp'),
                data.get('face_emotion'),
                data.get('heart_rate'),
                data.get('voice_emotion'),
                data.get('stress_level'),
                data.get('confidence'),
                data.get('notes')
            ))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False
    
    def get_wellness_data(self, days: int = 7) -> pd.DataFrame:
        """Retrieve wellness data from last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM wellness_data 
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(days,))
            conn.close()
            
            return df
        
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return pd.DataFrame()
    
    def create_session(self, session_data: dict) -> bool:
        """Create session record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO sessions 
                (start_time, end_time, duration_minutes, avg_stress, avg_heart_rate, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_data.get('start_time'),
                session_data.get('end_time'),
                session_data.get('duration_minutes'),
                session_data.get('avg_stress'),
                session_data.get('avg_heart_rate'),
                session_data.get('notes')
            ))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return False
    
    def log_alert(self, alert_data: dict) -> bool:
        """Log alert event"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now(),
                alert_data.get('alert_type'),
                alert_data.get('severity'),
                alert_data.get('message')
            ))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False
    
    def get_statistics(self, days: int = 7) -> dict:
        """Get wellness statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    AVG(heart_rate) as avg_heart_rate,
                    MIN(heart_rate) as min_heart_rate,
                    MAX(heart_rate) as max_heart_rate,
                    AVG(stress_level) as avg_stress,
                    AVG(confidence) as avg_confidence
                FROM wellness_data
                WHERE timestamp >= datetime('now', '-' || ? || ' days')
            """, (days,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_records': result[0],
                    'avg_heart_rate': result[1],
                    'min_heart_rate': result[2],
                    'max_heart_rate': result[3],
                    'avg_stress': result[4],
                    'avg_confidence': result[5]
                }
            
            return {}
        
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def export_to_csv(self, filename: str = None, days: int = None) -> str:
        """Export wellness data to CSV"""
        try:
            if filename is None:
                filename = f"wellness_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            if days is None:
                df = pd.read_sql_query("SELECT * FROM wellness_data", sqlite3.connect(self.db_path))
            else:
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query(
                    "SELECT * FROM wellness_data WHERE timestamp >= datetime('now', '-' || ? || ' days')",
                    conn,
                    params=(days,)
                )
                conn.close()
            
            df.to_csv(filename, index=False)
            logger.info(f"✓ Data exported to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return None
    
    def clear_old_data(self, days: int = 90) -> bool:
        """Clear data older than N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM wellness_data 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Deleted {deleted} old records")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing old data: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")

# Global database instance
_db_instance = None

def get_database() -> WellnessDatabase:
    """Get or create database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = WellnessDatabase()
    return _db_instance
