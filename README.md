# Postpartum Wellness Monitor - Setup Guide & Quick Start

This directory contains a complete federated learning system for postpartum wellness monitoring.
The system analyzes facial emotion, voice tone, and heart rate in real-time while preserving privacy.

DIRECTORY STRUCTURE:
=====================================
```
.
├── app.py                      # Main Streamlit application
├── server.py                   # Federated Learning Server (Flower)
├── dummyclient.py             # Dummy client for testing FL
├── requirements.txt           # Python dependencies
├── models/
│   └── emotion_model.h5       # (Download pre-trained Keras model)
└── README.md
```
QUICK START:
=====================================

1. CREATE VIRTUAL ENVIRONMENT:
   python -m venv venv
   
   On MacOS/Linux:
   source venv/bin/activate
   
   On Windows:
   venv\Scripts\activate

2. INSTALL DEPENDENCIES:
   pip install -r requirements.txt

3. DOWNLOAD PRE-TRAINED AUDIO EMOTION MODEL:
   - Go to: https://www.kaggle.com or GitHub
   - Search: "audio emotion classification keras .h5"
   - Download a pre-trained .h5 model file
   - Create folder: mkdir models
   - Place model file in: models/emotion_model.h5

4. RUN THE SYSTEM:

   TERMINAL 1 - Start Federated Learning Server:
   python server.py
   
   Expected output:
   ========================================
   🏥 POSTPARTUM WELLNESS MONITOR - FEDERATED LEARNING SERVER
   ========================================
   Server is waiting for clients to connect...
   
   TERMINAL 2 - Start Streamlit App:
   streamlit run app.py
   
   Expected output:
   - Browser opens to http://localhost:8501
   - Video feed displays with emotion labels
   - Heart rate displayed in real-time
   
   TERMINAL 3 (Optional) - Start Dummy Client:
   python dummyclient.py
   
   Expected output:
   ========================================
   🏥 POSTPARTUM WELLNESS MONITOR - DUMMY FEDERATED CLIENT
   ========================================
   Connecting to Federated Learning Server at localhost:8080...

5. INTERACT WITH APP:
   - Click "Start Audio Analysis" to analyze voice emotion
   - Click "Contribute Local Model to Server" to test federated learning
   - Watch the server terminal show weights aggregation
   - Click "Export Data" to download collected wellness metrics

ARCHITECTURE OVERVIEW:
=====================================

MODULE 1: VIDEO PROCESSING
  - Real-time facial emotion detection via DeepFace
  - Heart rate extraction using PPG (Photoplethysmography)
  - Green channel analysis + FFT-based frequency detection

MODULE 2: AUDIO PROCESSING
  - Voice Activity Detection (VAD) for privacy
  - MFCC feature extraction with librosa
  - Emotion classification via pre-trained TensorFlow model

MODULE 3: DATA LOGGING
  - Pandas DataFrame for time-series metrics
  - Streamlit session state for persistence within session
  - Live-updating dashboard charts

MODULE 4: MULTIMODAL ALERTS
  - Combines facial, vocal, and physiological signals
  - Detects high-stress patterns
  - Provides personalized recommendations

MODULE 5: FEDERATED LEARNING
  - Privacy-preserving model training via Flower (FLWR)
  - Local model training on each user's device
  - Only model weights aggregated (no raw data shared)
  - Supports multiple concurrent clients

REQUIREMENTS IN DETAIL:
=====================================

Core Libraries:
- streamlit: Web application framework
- streamlit-webrtc: Real-time video/audio streaming
- opencv-python-headless: Computer vision (video processing)
- deepface: Facial emotion recognition
- tensorflow: Deep learning framework
- pandas: Data manipulation and analysis
- librosa: Audio signal processing
- numpy: Numerical computing
- scipy: Scientific computing (signal processing, FFT)
- pyaudio: Audio input/output
- webrtcvad: Voice Activity Detection
- scikit-learn: Machine learning (Random Forest)
- flwr: Federated Learning framework


PRIVACY STATEMENT:
=====================================
```
✅ All processing happens on your local device
✅ Raw video, audio, and health data NEVER leave your computer
✅ Only anonymized model weights are shared to federated server
✅ No personal data is stored centrally
✅ Compliant with healthcare privacy standards
```
