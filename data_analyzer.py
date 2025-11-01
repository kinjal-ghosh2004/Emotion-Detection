"""
Data analysis and alert generation utilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict

class StressDetector:
    """Analyzes multimodal data for stress patterns"""
    
    NEGATIVE_EMOTIONS = ["sad", "anxious", "angry", "fear", "disgust"]
    
    def __init__(self, window_size: int = 10, hr_threshold: int = 100):
        """
        Initialize stress detector
        window_size: number of recent data points to analyze
        hr_threshold: heart rate threshold for elevated state
        """
        self.window_size = window_size
        self.hr_threshold = hr_threshold
    
    def analyze_stress(self, df: pd.DataFrame) -> Tuple[float, List[str], List[str]]:
        """
        Analyze recent data for stress patterns
        Returns: (stress_level, risk_factors, recommendations)
        """
        if len(df) < 5:
            return 0, [], []
        
        recent = df.tail(self.window_size)
        
        stress_level = 0
        risk_factors = []
        recommendations = []
        
        # Analyze heart rate
        avg_hr = recent["heart_rate"].mean()
        if avg_hr > 0:
            if avg_hr > self.hr_threshold:
                risk_factors.append(f"🔴 Elevated heart rate: {avg_hr:.0f} BPM")
                stress_level += 30
                recommendations.append("Practice slow, deep breathing (4-7-8 technique)")
            elif avg_hr > 90:
                risk_factors.append(f"🟡 Moderate heart rate: {avg_hr:.0f} BPM")
                stress_level += 15
        
        # Analyze facial emotion
        face_emotions = recent["face_emotion"].value_counts()
        negative_face = face_emotions.get([e for e in self.NEGATIVE_EMOTIONS 
                                          if e in face_emotions.index], 0)
        
        if negative_face > len(recent) * 0.3:  # >30% negative
            dominant_neg = recent[recent["face_emotion"].isin(self.NEGATIVE_EMOTIONS)]["face_emotion"].mode()
            if len(dominant_neg) > 0:
                risk_factors.append(f"🟡 Facial expression: {dominant_neg[0]}")
                stress_level += 25
                recommendations.append("Take a 5-minute break and do a grounding exercise")
        
        # Analyze voice emotion
        voice_emotions = recent["voice_emotion"].value_counts()
        negative_voice = voice_emotions.get([e for e in self.NEGATIVE_EMOTIONS 
                                           if e in voice_emotions.index], 0)
        
        if negative_voice > len(recent) * 0.3:  # >30% negative
            dominant_voice = recent[recent["voice_emotion"].isin(self.NEGATIVE_EMOTIONS)]["voice_emotion"].mode()
            if len(dominant_voice) > 0:
                risk_factors.append(f"🟡 Vocal tone: {dominant_voice[0]}")
                stress_level += 25
                recommendations.append("Try humming or singing to regulate mood")
        
        # Multimodal alert (2+ factors)
        if len(risk_factors) >= 2:
            stress_level = min(100, stress_level)
            if recommendations:
                recommendations.insert(0, "💡 **Multiple stress indicators detected**")
        else:
            stress_level = 0
        
        return stress_level, risk_factors, recommendations
    
    def get_wellness_score(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive wellness score
        Returns: dict with various metrics
        """
        if len(df) == 0:
            return {
                "overall_score": 0,
                "emotional_score": 0,
                "physiological_score": 0,
                "consistency_score": 0
            }
        
        # Emotional score (inverse of negative emotions)
        negative_count = len(df[df["face_emotion"].isin(self.NEGATIVE_EMOTIONS)])
        emotional_score = max(0, 100 - (negative_count / len(df) * 100))
        
        # Physiological score (HR consistency)
        hr_data = df["heart_rate"][df["heart_rate"] > 0]
        if len(hr_data) > 0:
            hr_std = hr_data.std()
            # Lower std = more consistent = higher score
            physiological_score = max(0, 100 - (hr_std / 3))
        else:
            physiological_score = 50
        
        # Consistency score (steady mood indicators)
        recent = df.tail(10)
        dominant_emotion = recent["face_emotion"].mode()
        if len(dominant_emotion) > 0 and dominant_emotion[0] not in self.NEGATIVE_EMOTIONS:
            consistency_score = 80
        else:
            consistency_score = 40
        
        overall_score = (emotional_score + physiological_score + consistency_score) / 3
        
        return {
            "overall_score": min(100, max(0, overall_score)),
            "emotional_score": min(100, max(0, emotional_score)),
            "physiological_score": min(100, max(0, physiological_score)),
            "consistency_score": min(100, max(0, consistency_score))
        }

class AlertManager:
    """Manages alerts and recommendations"""
    
    def __init__(self, alert_cooldown: int = 300):
        """
        Initialize alert manager
        alert_cooldown: seconds between repeated alerts
        """
        self.last_alert_time = {}
        self.alert_cooldown = alert_cooldown
    
    def should_alert(self, alert_type: str) -> bool:
        """Check if enough time has passed to show alert again"""
        now = datetime.now()
        last_time = self.last_alert_time.get(alert_type, None)
        
        if last_time is None:
            self.last_alert_time[alert_type] = now
            return True
        
        if (now - last_time).total_seconds() > self.alert_cooldown:
            self.last_alert_time[alert_type] = now
            return True
        
        return False
    
    def get_personalized_recommendation(self, stress_level: float, risk_factors: List[str]) -> str:
        """Get personalized recommendation based on stress level"""
        if stress_level >= 70:
            return "🚨 **High stress detected**: Please reach out to a healthcare provider or counselor"
        elif stress_level >= 50:
            return "💡 Try: Box breathing (4-4-4-4), meditation, or talking to a support person"
        elif stress_level >= 30:
            return "🌿 Consider: A short walk, hydration, or a calming activity like reading"
        else:
            return "✨ You're doing well! Keep maintaining your wellness routine"

class DataAnalyzer:
    """Comprehensive data analysis"""
    
    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> Dict:
        """Get summary statistics"""
        if len(df) == 0:
            return {}
        
        stats = {
            "total_records": len(df),
            "session_duration": (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 60,
            "avg_heart_rate": df[df["heart_rate"] > 0]["heart_rate"].mean() if "heart_rate" in df.columns else 0,
            "max_heart_rate": df["heart_rate"].max() if "heart_rate" in df.columns else 0,
            "min_heart_rate": df[df["heart_rate"] > 0]["heart_rate"].min() if "heart_rate" in df.columns else 0,
        }
        
        return stats
    
    @staticmethod
    def get_emotion_distribution(df: pd.DataFrame) -> Dict:
        """Get emotion distribution"""
        face_dist = df["face_emotion"].value_counts().to_dict()
        voice_dist = df["voice_emotion"].value_counts().to_dict()
        
        return {
            "facial": face_dist,
            "vocal": voice_dist
        }
    
    @staticmethod
    def export_report(df: pd.DataFrame) -> str:
        """Generate text report"""
        if len(df) == 0:
            return "No data collected yet."
        
        stats = DataAnalyzer.get_summary_stats(df)
        emotions = DataAnalyzer.get_emotion_distribution(df)
        
        report = f"""
POSTPARTUM WELLNESS REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SESSION SUMMARY
- Total Records: {stats.get('total_records', 0)}
- Duration: {stats.get('session_duration', 0):.1f} minutes
- Avg Heart Rate: {stats.get('avg_heart_rate', 0):.0f} BPM
- HR Range: {stats.get('min_heart_rate', 0):.0f} - {stats.get('max_heart_rate', 0):.0f} BPM

EMOTIONAL STATE
Facial Emotions: {emotions['facial']}
Vocal Emotions: {emotions['vocal']}

RECOMMENDATIONS
- Regular monitoring helps track wellness trends
- Maintain a consistent sleep schedule
- Stay hydrated and take regular breaks
- Connect with support networks when needed

========================================
For detailed analysis, consult with your healthcare provider.
"""
        return report
