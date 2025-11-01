"""
config.py - Centralized Configuration Management
Improves maintainability and makes tuning easy
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class VideoConfig:
    """Video processing configuration"""
    BUFFER_SIZE: int = 150  # frames for PPG signal
    FPS: int = 30
    PROCESS_EVERY_N_FRAMES: int = 3  # Process every 3rd frame for performance
    BPM_MIN: int = 40
    BPM_MAX: int = 200
    BPM_LOW_FREQ: float = 0.83  # Hz (50 BPM)
    BPM_HIGH_FREQ: float = 3.0  # Hz (180 BPM)
    FILTER_ORDER: int = 4
    EMOTION_CACHE_DURATION: float = 0.5  # seconds
    ROI_X_RATIO: float = 1/3
    ROI_Y_RATIO: float = 1/4
    ROI_W_RATIO: float = 1/3
    ROI_H_RATIO: float = 1/6

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    CHUNK_SIZE: int = 480  # 30ms at 16kHz
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    ENERGY_THRESHOLD: float = 0.01  # For speech detection
    SILENCE_THRESHOLD: int = 10  # ~300ms silence
    MFCC_COEFF: int = 13
    VAD_MODE: int = 2  # 0-3, lower = more aggressive

@dataclass
class WellnessConfig:
    """Wellness analysis configuration"""
    ANALYSIS_WINDOW: int = 20  # Recent data points to analyze
    HR_ELEVATED_THRESHOLD: int = 110
    HR_MODERATE_THRESHOLD: int = 95
    NEGATIVE_EMOTION_THRESHOLD: float = 0.4  # 40% of data points
    VOICE_NEGATIVE_EMOTION_THRESHOLD: float = 0.3
    MAX_STRESS_LEVEL: int = 100
    MIN_STRESS_LEVEL: int = 0
    WELLNESS_HISTORY_MAX: int = 100
    
    NEGATIVE_EMOTIONS: List[str] = None
    
    def __post_init__(self):
        if self.NEGATIVE_EMOTIONS is None:
            self.NEGATIVE_EMOTIONS = ["sad", "anxious", "angry", "fear", "disgust"]

@dataclass
class AlertConfig:
    """Alert configuration"""
    HIGH_STRESS_THRESHOLD: int = 60
    MODERATE_STRESS_THRESHOLD: int = 40
    ALERT_COOLDOWN: int = 300  # seconds
    MIN_DATA_POINTS_FOR_ANALYSIS: int = 5
    MIN_DATA_POINTS_FOR_FL: int = 10

@dataclass
class FLConfig:
    """Federated Learning configuration"""
    NUM_ROUNDS: int = 3
    MIN_CLIENTS: int = 1
    MIN_AVAILABLE_CLIENTS: int = 1
    SERVER_ADDRESS: str = "0.0.0.0:8080"
    CLIENT_ADDRESS: str = "localhost:8080"
    MODEL_TYPE: str = "random_forest"
    N_ESTIMATORS: int = 5
    MAX_DEPTH: int = 3

@dataclass
class UIConfig:
    """UI configuration"""
    PAGE_TITLE: str = "Postpartum Wellness Monitor v2.0"
    PAGE_LAYOUT: str = "wide"
    SIDEBAR_INITIAL_STATE: str = "expanded"
    THEME: str = "light"
    PRIMARY_COLOR: str = "#E94B3C"  # Health-related pink/red

class Config:
    """Master configuration class"""
    
    video = VideoConfig()
    audio = AudioConfig()
    wellness = WellnessConfig()
    alert = AlertConfig()
    fl = FLConfig()
    ui = UIConfig()
    
    # Emotion colors for visualization
    EMOTION_COLORS: Dict[str, tuple] = {
        "happy": (0, 255, 0),      # Green
        "neutral": (200, 200, 200), # Gray
        "sad": (0, 0, 255),         # Red
        "anxious": (255, 0, 0),     # Blue
        "angry": (0, 165, 255),     # Orange
        "fear": (255, 0, 255),      # Magenta
        "disgust": (50, 205, 50),   # Lime
        "surprise": (0, 255, 255)   # Yellow
    }
    
    # Recommendations
    STRESS_RECOMMENDATIONS: Dict[str, List[str]] = {
        "high": [
            "🚨 Contact healthcare provider immediately",
            "Try grounding techniques (5-4-3-2-1 senses)",
            "Gentle progressive muscle relaxation",
            "Consider crisis support hotline"
        ],
        "moderate": [
            "Practice 5-minute breathing exercise (4-7-8 technique)",
            "Take a short walk outdoors",
            "Listen to calming music",
            "Gentle stretching or yoga",
            "Reach out to support person"
        ],
        "low": [
            "Continue current wellness routine",
            "Stay hydrated and maintain sleep schedule",
            "Engage in enjoyable activities",
            "Regular check-ins with support network"
        ]
    }
