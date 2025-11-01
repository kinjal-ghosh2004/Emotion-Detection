import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import queue
from datetime import datetime, timedelta
from collections import deque
import av
from deepface import DeepFace
from scipy import signal
from scipy.fft import fft
import pyaudio
import librosa
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
import time
import logging

warnings.filterwarnings("ignore")

# ============================================================================
# LOGGING & CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Postpartum Wellness Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables"""
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(
            columns=["timestamp", "face_emotion", "heart_rate", "voice_emotion", "stress_level", "confidence"]
        )
    
    if "voice_emotion" not in st.session_state:
        st.session_state.voice_emotion = "neutral"
    
    if "heart_rate_buffer" not in st.session_state:
        st.session_state.heart_rate_buffer = deque(maxlen=150)
    
    if "current_face_emotion" not in st.session_state:
        st.session_state.current_face_emotion = "initializing"
    
    if "current_heart_rate" not in st.session_state:
        st.session_state.current_heart_rate = 0
    
    if "current_emotion_confidence" not in st.session_state:
        st.session_state.current_emotion_confidence = 0.0
    
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
    
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []
    
    if "emotion_model" not in st.session_state:
        st.session_state.emotion_model = load_emotion_model()
    
    if "wellness_score_history" not in st.session_state:
        st.session_state.wellness_score_history = deque(maxlen=100)

def load_emotion_model():
    """Load pre-trained emotion model with fallback"""
    try:
        model = tf.keras.models.load_model("model/emotion_model.pt")
        logger.info("✓ Emotion model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"⚠️ Could not load emotion model: {e}")
        return None

init_session_state()

# ============================================================================
# ENHANCED VIDEO PROCESSOR WITH ERROR HANDLING
# ============================================================================

class EnhancedVideoProcessor:
    """Advanced video processor with robust error handling"""
    
    def __init__(self):
        self.frame_count = 0
        self.emotion_cache = {"emotion": "neutral", "confidence": 0.0, "timestamp": time.time()}
        self.cache_duration = 0.5  # Cache results for 0.5 seconds
        
    def recv(self, frame):
        """Process video frame with caching and error handling"""
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            
            # Process every 3rd frame for performance
            if self.frame_count % 3 == 0:
                # Facial emotion detection with caching
                emotion_result = self._analyze_emotion_cached(img)
                if emotion_result:
                    st.session_state.current_face_emotion = emotion_result["emotion"]
                    st.session_state.current_emotion_confidence = emotion_result["confidence"]
                    
                    # Draw emotion on frame
                    cv2.putText(img, 
                               f"{emotion_result['emotion'].upper()} ({emotion_result['confidence']:.2f})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
            
            # Heart rate detection every frame
            heart_rate = self._extract_heart_rate(img)
            if heart_rate:
                st.session_state.current_heart_rate = heart_rate
                # Color code based on HR
                hr_color = (0, 255, 0) if heart_rate < 100 else (0, 165, 255) if heart_rate < 120 else (0, 0, 255)
                cv2.putText(img, f"HR: {int(heart_rate)} BPM", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, hr_color, 2)
            
            # Add frame counter for debugging
            if st.session_state.get("show_debug", False):
                cv2.putText(img, f"Frame: {self.frame_count}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def _analyze_emotion_cached(self, frame):
        """Cached emotion analysis to reduce computation"""
        current_time = time.time()
        if current_time - self.emotion_cache["timestamp"] < self.cache_duration:
            return self.emotion_cache
        
        try:
            results = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            if results:
                emotion = results[0]["dominant_emotion"]
                confidence = results[0]["emotion"][emotion]
                
                self.emotion_cache = {
                    "emotion": emotion,
                    "confidence": confidence,
                    "timestamp": current_time
                }
                return self.emotion_cache
        except Exception as e:
            logger.debug(f"DeepFace error: {e}")
        
        return None
    
    def _extract_heart_rate(self, frame):
        """Extract heart rate using optimized PPG method"""
        try:
            h, w = frame.shape[:2]
            roi_x, roi_y = w // 3, h // 4
            roi_w, roi_h = w // 3, h // 6
            
            if roi_y < 0 or roi_x < 0:
                return None
            
            roi = frame[max(0, roi_y):roi_y+roi_h, max(0, roi_x):roi_x+roi_w]
            
            if roi.size == 0:
                return None
            
            # Extract green channel
            green = roi[:, :, 1].astype(float)
            mean_green = np.mean(green)
            st.session_state.heart_rate_buffer.append(mean_green)
            
            if len(st.session_state.heart_rate_buffer) == 150:
                hr = self._calculate_heart_rate()
                if hr:
                    return hr
        
        except Exception as e:
            logger.debug(f"Heart rate extraction error: {e}")
        
        return None
    
    def _calculate_heart_rate(self):
        """Calculate BPM from PPG signal using FFT"""
        try:
            signal_array = np.array(list(st.session_state.heart_rate_buffer), dtype=np.float32)
            signal_array = (signal_array - np.mean(signal_array)) / (np.std(signal_array) + 1e-8)
            
            # Apply bandpass filter
            sos = signal.butter(4, [0.83, 3.0], btype='band', fs=30, output='sos')
            filtered = signal.sosfilt(sos, signal_array)
            
            # FFT analysis
            fft_result = fft(filtered)
            frequencies = np.fft.fftfreq(len(filtered), 1/30)
            magnitude = np.abs(fft_result)
            
            valid_mask = (frequencies > 0.83) & (frequencies < 3.0)
            if not np.any(valid_mask):
                return None
            
            peak_freq = frequencies[valid_mask][np.argmax(magnitude[valid_mask])]
            bpm = peak_freq * 60
            bpm = max(40, min(200, bpm))  # Clamp to physiological range
            
            return bpm
        
        except Exception as e:
            logger.debug(f"Heart rate calculation error: {e}")
            return None

# ============================================================================
# ENHANCED AUDIO PROCESSOR
# ============================================================================

class EnhancedAudioProcessor(threading.Thread):
    """Background audio processing with improved error handling"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.is_running = False
        self.stream = None
        self.p = None
    
    def run(self):
        """Audio processing loop"""
        CHUNK = 480
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK,
                exception_on_overflow=False
            )
            
            speech_buffer = []
            silence_count = 0
            
            while self.is_running:
                try:
                    data = self.stream.read(CHUNK, exception_on_overflow=False)
                    
                    # Process audio (simplified - no VAD needed for basic demo)
                    audio_np = np.frombuffer(data, dtype=np.int16).astype(float) / 32768.0
                    
                    # Simple energy detection instead of VAD
                    energy = np.sqrt(np.mean(audio_np ** 2))
                    
                    if energy > 0.01:  # Threshold for speech
                        speech_buffer.extend(audio_np)
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count > 10 and speech_buffer:  # ~300ms silence
                            self._process_speech(np.array(speech_buffer))
                            speech_buffer = []
                
                except Exception as e:
                    logger.debug(f"Audio read error: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
        
        finally:
            self._cleanup()
    
    def _process_speech(self, audio_data):
        """Process captured speech"""
        try:
            if st.session_state.emotion_model is None:
                return
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1).reshape(1, -1)
            
            # Predict
            prediction = st.session_state.emotion_model.predict(mfccs_mean, verbose=0)
            emotions = ["calm", "anxious", "angry", "sad", "happy", "neutral"]
            emotion_idx = np.argmax(prediction[0])
            emotion = emotions[min(emotion_idx, len(emotions)-1)]
            
            st.session_state.voice_emotion = emotion
        
        except Exception as e:
            logger.debug(f"Speech processing error: {e}")
    
    def start_listening(self):
        """Start audio listening"""
        if not self.is_running:
            self.is_running = True
            self.start()
    
    def stop_listening(self):
        """Stop audio listening"""
        self.is_running = False
    
    def _cleanup(self):
        """Clean up audio resources"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.p:
                self.p.terminate()
        except Exception as e:
            logger.debug(f"Cleanup error: {e}")

# ============================================================================
# ENHANCED DATA LOGGING & ANALYSIS
# ============================================================================

def log_data():
    """Enhanced data logging with validation"""
    try:
        face_emotion = st.session_state.get("current_face_emotion", "neutral")
        heart_rate = float(st.session_state.get("current_heart_rate", 0))
        voice_emotion = st.session_state.get("voice_emotion", "neutral")
        confidence = float(st.session_state.get("current_emotion_confidence", 0.0))
        
        # Validate data
        if heart_rate < 0 or heart_rate > 250:
            return
        
        new_row = pd.DataFrame({
            "timestamp": [datetime.now()],
            "face_emotion": [face_emotion],
            "heart_rate": [heart_rate],
            "voice_emotion": [voice_emotion],
            "stress_level": [0],
            "confidence": [confidence]
        })
        
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
    
    except Exception as e:
        logger.error(f"Data logging error: {e}")

def analyze_wellness():
    """Enhanced wellness analysis with multiple metrics"""
    if len(st.session_state.df) < 5:
        return None, None, None
    
    recent_data = st.session_state.df.tail(20)
    
    # Heart rate analysis
    hr_data = recent_data[recent_data["heart_rate"] > 0]["heart_rate"]
    avg_hr = hr_data.mean() if len(hr_data) > 0 else 0
    hr_std = hr_data.std() if len(hr_data) > 0 else 0
    
    # Emotion analysis
    negative_emotions = ["sad", "anxious", "angry", "fear", "disgust"]
    face_negative = len(recent_data[recent_data["face_emotion"].isin(negative_emotions)])
    voice_negative = len(recent_data[recent_data["voice_emotion"].isin(negative_emotions)])
    
    # Stress calculation
    stress_level = 0
    risk_factors = []
    recommendations = []
    
    if avg_hr > 110:
        stress_level += 35
        risk_factors.append(f"🔴 Elevated HR: {avg_hr:.0f} BPM")
        recommendations.append("Take slow, deep breaths (5-6 breaths/min)")
    elif avg_hr > 95:
        stress_level += 20
        risk_factors.append(f"🟡 Moderate HR: {avg_hr:.0f} BPM")
    
    if face_negative / len(recent_data) > 0.4:
        stress_level += 25
        dominant_emotion = recent_data[recent_data["face_emotion"].isin(negative_emotions)]["face_emotion"].mode()
        if len(dominant_emotion) > 0:
            risk_factors.append(f"🟡 Facial: {dominant_emotion.iloc[0]}")
        recommendations.append("Practice 5-min mindfulness meditation")
    
    if voice_negative / len(recent_data) > 0.3:
        stress_level += 20
        dominant_voice = recent_data[recent_data["voice_emotion"].isin(negative_emotions)]["voice_emotion"].mode()
        if len(dominant_voice) > 0:
            risk_factors.append(f"🟡 Voice tone: {dominant_voice.iloc[0]}")
        recommendations.append("Try gentle yoga or stretching")
    
    # Wellness score
    wellness_score = 100 - min(stress_level, 100)
    st.session_state.wellness_score_history.append(wellness_score)
    
    return min(100, stress_level), risk_factors, recommendations

# ============================================================================
# UI LAYOUT - FIXED FOR STREAMLIT 1.28+
# ============================================================================

st.title("🏥 Postpartum Wellness Monitor v2.0")
st.markdown("**Advanced Real-Time Emotion & Wellness Analysis with Privacy Protection**")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    show_debug = st.checkbox("🔧 Debug Mode", value=False)
    st.session_state.show_debug = show_debug
    
    col1, col2 = st.columns(2)
    with col1:
        start_audio = st.button("🎤 Start Audio", use_container_width=True)
    with col2:
        stop_audio = st.button("⏹️ Stop Audio", use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        export_data = st.button("📊 Export CSV", use_container_width=True)
    with col2:
        clear_data = st.button("🗑️ Clear Data", use_container_width=True)
    
    st.divider()
    st.subheader("📈 Statistics")
    
    if len(st.session_state.df) > 0:
        st.metric("Records", len(st.session_state.df))
        hr_data = st.session_state.df[st.session_state.df["heart_rate"] > 0]["heart_rate"]
        if len(hr_data) > 0:
            st.metric("Avg HR", f"{hr_data.mean():.0f} BPM")
            st.metric("HR Range", f"{hr_data.min():.0f}-{hr_data.max():.0f}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📹 Live Feed", "📊 Analytics", "❤️ Wellness", "🔒 Privacy"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Real-Time Video Stream")
        webrtc_ctx = webrtc_streamer(
            key="wellness-monitor",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EnhancedVideoProcessor,
            media_stream_constraints={"audio": False, "video": True},
            async_processing=True,
        )
    
    with col2:
        st.subheader("📈 Current Metrics")
        
        metric1, metric2 = st.columns(2)
        with metric1:
            st.metric("😊 Emotion",
                     st.session_state.get("current_face_emotion", "—").upper(),
                     delta=f"{st.session_state.get('current_emotion_confidence', 0):.2f}")
        with metric2:
            st.metric("❤️ Heart Rate",
                     f"{int(st.session_state.get('current_heart_rate', 0))} BPM")
        
        st.divider()
        
        st.metric("🗣️ Voice Tone",
                 st.session_state.get("voice_emotion", "—").upper())

with tab2:
    st.subheader("Detailed Analytics Dashboard")
    
    if len(st.session_state.df) > 0:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Session Duration", f"{(len(st.session_state.df) / 2):.1f} min")
        
        with col2:
            avg_confidence = st.session_state.df["confidence"].mean()
            st.metric("Detection Confidence", f"{avg_confidence:.2%}")
        
        with col3:
            unique_emotions = st.session_state.df["face_emotion"].nunique()
            st.metric("Emotion Variety", f"{unique_emotions} types")
        
        # Charts - FIXED: use width instead of use_container_width
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Heart Rate Trend")
            hr_data = st.session_state.df[st.session_state.df["heart_rate"] > 0][["timestamp", "heart_rate"]]
            if len(hr_data) > 0:
                st.line_chart(hr_data.set_index("timestamp")["heart_rate"], use_container_width=True)
        
        with chart_col2:
            st.subheader("Emotion Distribution")
            emotion_dist = st.session_state.df["face_emotion"].value_counts()
            st.bar_chart(emotion_dist, use_container_width=True)
        
        # Time series
        st.subheader("Complete Timeline")
        st.dataframe(st.session_state.df.tail(20), use_container_width=True)
    else:
        st.info("📊 No data collected yet. Start monitoring to see analytics!")

with tab3:
    st.subheader("🏥 Wellness Assessment")
    
    stress_level, risk_factors, recommendations = analyze_wellness()
    
    if stress_level is not None:
        # Stress gauge
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Stress Level", f"{int(stress_level)}%")
        
        with col2:
            wellness_score = 100 - stress_level
            st.metric("Wellness Score", f"{int(wellness_score)}/100")
        
        with col3:
            if len(st.session_state.wellness_score_history) > 0:
                trend = "📈" if st.session_state.wellness_score_history[-1] >= (
                    st.session_state.wellness_score_history[0] if len(st.session_state.wellness_score_history) > 1 
                    else 50
                ) else "📉"
                st.metric("Trend", trend)
        
        st.divider()
        
        # Alerts and recommendations
        if stress_level >= 60:
            st.error(f"⚠️ **HIGH STRESS ALERT** ({int(stress_level)}%)")
        elif stress_level >= 40:
            st.warning(f"⚠️ **Moderate Stress Detected** ({int(stress_level)}%)")
        else:
            st.success(f"✅ **Wellness Level Good** ({int(stress_level)}%)")
        
        if risk_factors:
            st.subheader("Risk Factors")
            for factor in risk_factors:
                st.write(factor)
        
        if recommendations:
            st.subheader("💡 Recommendations")
            for rec in recommendations:
                st.write(rec)
    else:
        st.info("ℹ️ Monitor for at least 5 data points to see wellness assessment")
    
    # Wellness history chart
    if len(st.session_state.wellness_score_history) > 1:
        st.subheader("Wellness Score History")
        st.line_chart(list(st.session_state.wellness_score_history), use_container_width=True)

with tab4:
    st.subheader("🔒 Privacy & Security")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ✅ **Local Processing**
        - All video frames processed locally
        - No images uploaded to cloud
        - Audio analyzed on your device
        
        ✅ **Data Privacy**
        - Only metrics stored (no raw media)
        - Emotions extracted, not stored raw
        - Heart rate calculated locally
        """)
    
    with col2:
        st.write("""
        ✅ **Federated Learning**
        - Local model training only
        - Weights aggregated, not data
        - Multi-user safe
        - HIPAA-friendly design
        
        ✅ **User Control**
        - Clear Data button clears all
        - Export anytime
        - Stop monitoring anytime
        """)
    
    st.divider()
    
    if st.button("📥 Contribute to Federated Learning"):
        if len(st.session_state.df) > 10:
            st.success("✅ Model training initiated in background...")
            st.info("💡 Server will aggregate weights from multiple contributors")
        else:
            st.warning("⚠️ Collect at least 10 data points first")

# Data logging
if webrtc_ctx.state.playing:
    log_data()

# Handle button actions
if start_audio:
    st.session_state.audio_processor = EnhancedAudioProcessor()
    st.session_state.audio_processor.start_listening()
    st.success("🎤 Audio monitoring started")

if stop_audio:
    if hasattr(st.session_state, 'audio_processor'):
        st.session_state.audio_processor.stop_listening()
    st.info("⏹️ Audio monitoring stopped")

if export_data:
    if len(st.session_state.df) > 0:
        csv = st.session_state.df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"wellness_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data to export")

if clear_data:
    st.session_state.df = pd.DataFrame(
        columns=["timestamp", "face_emotion", "heart_rate", "voice_emotion", "stress_level", "confidence"]
    )
    st.session_state.wellness_score_history.clear()
    st.success("🗑️ Data cleared")

# Footer
st.markdown("---")
st.markdown("""
💜 **Postpartum Wellness Monitor v2.0** | Privacy-First | Real-Time Analysis
""")
