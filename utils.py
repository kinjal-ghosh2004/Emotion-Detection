"""
utils.py - Utility functions and helpers
Provides common functionality across modules
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    """Validates and sanitizes incoming data"""
    
    @staticmethod
    def validate_heart_rate(hr: float) -> bool:
        """Check if heart rate is in physiological range"""
        return 30 <= hr <= 250
    
    @staticmethod
    def validate_emotion(emotion: str) -> bool:
        """Validate emotion string"""
        valid_emotions = {"happy", "sad", "angry", "anxious", "fear", 
                         "disgust", "surprise", "neutral"}
        return emotion.lower() in valid_emotions
    
    @staticmethod
    def validate_confidence(confidence: float) -> bool:
        """Validate confidence score"""
        return 0.0 <= confidence <= 1.0
    
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers and invalid data"""
        if df.empty:
            return df
        
        # Remove HR outliers using IQR method
        hr_col = df["heart_rate"]
        valid_hr = hr_col[(hr_col > 0) & (hr_col <= 200)]
        
        if len(valid_hr) > 0:
            Q1 = valid_hr.quantile(0.25)
            Q3 = valid_hr.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df = df[(df["heart_rate"] <= 0) | 
                   ((df["heart_rate"] >= lower_bound) & 
                    (df["heart_rate"] <= upper_bound))]
        
        return df

class MetricsCalculator:
    """Calculate various wellness metrics"""
    
    @staticmethod
    def calculate_heart_rate_variability(hr_series: pd.Series) -> float:
        """Calculate HRV (heart rate variability)"""
        if len(hr_series) < 2:
            return 0.0
        
        valid_hr = hr_series[(hr_series > 0) & (hr_series <= 200)]
        if len(valid_hr) < 2:
            return 0.0
        
        return float(valid_hr.std())
    
    @staticmethod
    def calculate_emotion_stability(emotion_series: pd.Series) -> float:
        """Calculate emotion stability (0-1, 1 = most stable)"""
        if len(emotion_series) == 0:
            return 0.0
        
        dominant_emotion = emotion_series.mode()
        if len(dominant_emotion) == 0:
            return 0.0
        
        consistency = (emotion_series == dominant_emotion.iloc[0]).sum() / len(emotion_series)
        return float(consistency)
    
    @staticmethod
    def calculate_stress_trend(stress_history: List[float]) -> str:
        """Determine if stress is increasing or decreasing"""
        if len(stress_history) < 3:
            return "insufficient_data"
        
        recent = np.array(stress_history[-3:])
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend > 5:
            return "increasing"
        elif trend < -5:
            return "decreasing"
        else:
            return "stable"
    
    @staticmethod
    def calculate_recovery_time(stress_history: List[float]) -> Optional[int]:
        """Estimate minutes until stress returns to baseline"""
        if len(stress_history) < 2:
            return None
        
        recent_stress = stress_history[-1]
        
        if recent_stress < 30:
            return 0
        
        # Simple linear regression
        x = np.arange(len(stress_history[-10:]))
        y = np.array(stress_history[-10:])
        
        if len(y) < 2 or y[-1] >= y[0]:  # Stress not decreasing
            return None
        
        slope, _ = np.polyfit(x, y, 1)
        
        if slope >= 0:
            return None
        
        # Time to reach 30% stress
        minutes_to_recovery = (30 - recent_stress) / (slope * 2)  # 2 data points per minute
        return max(0, int(minutes_to_recovery))

class TimeSeriesAnalyzer:
    """Analyze time series data"""
    
    @staticmethod
    def get_summary_window(df: pd.DataFrame, minutes: int = 5) -> pd.DataFrame:
        """Get data from last N minutes"""
        if df.empty:
            return df
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return df[df["timestamp"] >= cutoff_time]
    
    @staticmethod
    def resample_data(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
        """Resample time series data"""
        if df.empty:
            return df
        
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])
        df_copy = df_copy.set_index("timestamp")
        
        # Resample numeric columns
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_copy[col] = df_copy[col].resample(freq).mean()
        
        return df_copy.reset_index()
    
    @staticmethod
    def detect_anomalies(series: pd.Series, window: int = 5) -> List[int]:
        """Detect anomalies using rolling statistics"""
        if len(series) < window:
            return []
        
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        upper_bound = rolling_mean + 2 * rolling_std
        lower_bound = rolling_mean - 2 * rolling_std
        
        anomalies = np.where(
            (series > upper_bound) | (series < lower_bound)
        )[0].tolist()
        
        return anomalies

class NotificationGenerator:
    """Generate contextual notifications"""
    
    @staticmethod
    def generate_greeting(time_of_day: Optional[str] = None) -> str:
        """Generate time-appropriate greeting"""
        hour = datetime.now().hour
        
        if hour < 12:
            return "Good morning 🌅"
        elif hour < 17:
            return "Good afternoon ☀️"
        elif hour < 21:
            return "Good evening 🌆"
        else:
            return "Good night 🌙"
    
    @staticmethod
    def generate_wellness_message(wellness_score: float) -> str:
        """Generate wellness-based message"""
        if wellness_score >= 80:
            return "Excellent! You're doing great. Keep up your wellness routine!"
        elif wellness_score >= 60:
            return "You're doing well! Consider some relaxation techniques to improve further."
        elif wellness_score >= 40:
            return "Take a break and practice some self-care. You deserve it."
        else:
            return "Please prioritize your wellbeing. Consider reaching out for support."
    
    @staticmethod
    def generate_action_item(stress_level: float, recent_emotion: str) -> str:
        """Generate specific action item based on current state"""
        if stress_level > 70:
            return "💡 Try: Deep breathing exercise (4-7-8 breathing technique)"
        elif stress_level > 50 and recent_emotion in ["anxious", "sad"]:
            return "💡 Try: 5-minute guided meditation or grounding exercise"
        elif stress_level > 30:
            return "💡 Try: Short walk or light stretching"
        else:
            return "💡 Continue your current wellness activities"

class ExportFormatter:
    """Format data for export"""
    
    @staticmethod
    def generate_csv_report(df: pd.DataFrame) -> str:
        """Generate CSV export"""
        return df.to_csv(index=False)
    
    @staticmethod
    def generate_text_report(df: pd.DataFrame, wellness_score: float) -> str:
        """Generate human-readable text report"""
        if df.empty:
            return "No data to report."
        
        report = f"""
        POSTPARTUM WELLNESS REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        SESSION SUMMARY
        ===============
        Total Records: {len(df)}
        Duration: {(len(df) / 2):.1f} minutes
        Wellness Score: {wellness_score:.1f}/100
        
        PHYSIOLOGICAL DATA
        ==================
        Avg Heart Rate: {df[df['heart_rate'] > 0]['heart_rate'].mean():.0f} BPM
        HR Range: {df[df['heart_rate'] > 0]['heart_rate'].min():.0f} - {df[df['heart_rate'] > 0]['heart_rate'].max():.0f} BPM
        HR Stability: {df[df['heart_rate'] > 0]['heart_rate'].std():.1f} (std dev)
        
        EMOTIONAL STATE
        ===============
        Facial Emotions: {df['face_emotion'].value_counts().to_dict()}
        Vocal Tones: {df['voice_emotion'].value_counts().to_dict()}
        Emotion Stability: {MetricsCalculator.calculate_emotion_stability(df['face_emotion']):.2%}
        
        RECOMMENDATIONS
        ================
        - Maintain regular monitoring
        - Practice stress management techniques
        - Maintain healthy sleep schedule
        - Stay connected with support network
        
        IMPORTANT
        =========
        For clinical guidance, consult with your healthcare provider.
        This report is for personal wellness tracking only.
        """
        return report

# Export all utilities
__all__ = [
    'DataValidator',
    'MetricsCalculator',
    'TimeSeriesAnalyzer',
    'NotificationGenerator',
    'ExportFormatter'
]
