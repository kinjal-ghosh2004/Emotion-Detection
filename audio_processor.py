"""
Audio processing utilities for emotion detection
"""

import numpy as np
import pyaudio
import threading
import webrtcvad
from collections import deque
import librosa
import tensorflow as tf
from typing import Optional, Callable

class AudioEmotionDetector:
    """Detects emotion from audio using pre-trained model"""
    
    EMOTIONS = ["calm", "anxious", "angry", "sad", "happy", "neutral"]
    
    def __init__(self, model_path: str = "models/emotion_model.h5"):
        """Initialize audio emotion detector"""
        self.model = None
        self.last_emotion = "neutral"
        
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"⚠️ Could not load emotion model: {e}")
    
    def extract_mfcc_features(self, audio_data: np.ndarray, sr: int = 16000, n_mfcc: int = 13):
        """Extract MFCC features from audio"""
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
            mfccs_mean = np.mean(mfccs, axis=1)
            return mfccs_mean
        except Exception as e:
            return None
    
    def predict_emotion(self, audio_data: np.ndarray) -> str:
        """Predict emotion from audio data"""
        if self.model is None:
            return self.last_emotion
        
        try:
            # Extract features
            features = self.extract_mfcc_features(audio_data)
            
            if features is None or len(features) == 0:
                return self.last_emotion
            
            # Ensure correct shape
            features = features.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(features, verbose=0)
            emotion_idx = np.argmax(prediction[0])
            emotion = self.EMOTIONS[emotion_idx] if emotion_idx < len(self.EMOTIONS) else "neutral"
            
            self.last_emotion = emotion
            return emotion
        except Exception as e:
            return self.last_emotion

class VoiceActivityDetector:
    """Detects voice activity in audio stream"""
    
    def __init__(self, vad_mode: int = 2):
        """
        Initialize VAD
        vad_mode: 0-3 (0=most aggressive, 3=least aggressive)
        """
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(vad_mode)
        self.sample_rate = 16000
        self.frame_duration = 30  # milliseconds
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
    
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech"""
        try:
            return self.vad.is_speech(audio_chunk, self.sample_rate)
        except:
            return False

class AudioStreamProcessor:
    """Background processor for audio streams"""
    
    def __init__(self, emotion_model_path: str = "models/emotion_model.h5",
                 callback: Optional[Callable] = None):
        """
        Initialize audio processor
        callback: function called with emotion when speech ends
        """
        self.emotion_detector = AudioEmotionDetector(emotion_model_path)
        self.vad = VoiceActivityDetector(vad_mode=2)
        self.callback = callback
        self.is_running = False
        self.thread = None
        
        # Audio parameters
        self.CHUNK = self.vad.frame_size
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = self.vad.sample_rate
    
    def start(self):
        """Start audio processing in background thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop audio processing"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _process_audio(self):
        """Main audio processing loop"""
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
                exception_on_overflow=False
            )
            
            speech_buffer = []
            
            while self.is_running:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    is_speech = self.vad.is_speech(data)
                    
                    if is_speech:
                        speech_buffer.append(data)
                    elif speech_buffer:
                        # Process accumulated speech
                        audio_data = b''.join(speech_buffer)
                        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(float) / 32768.0
                        
                        emotion = self.emotion_detector.predict_emotion(audio_np)
                        
                        if self.callback:
                            self.callback(emotion)
                        
                        speech_buffer = []
                
                except Exception as e:
                    pass
            
            stream.stop_stream()
            stream.close()
        
        except Exception as e:
            print(f"Audio stream error: {e}")
        
        finally:
            p.terminate()

class AudioBuffer:
    """Circular buffer for audio data"""
    
    def __init__(self, max_seconds: int = 60, sample_rate: int = 16000):
        """Initialize buffer"""
        self.max_size = max_seconds * sample_rate
        self.buffer = deque(maxlen=self.max_size)
        self.sample_rate = sample_rate
    
    def append(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.buffer.extend(audio_chunk)
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio"""
        return np.array(list(self.buffer), dtype=np.float32)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) == self.max_size
