"""
Federated Learning utilities for privacy-preserving model training
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
from typing import Tuple, Optional
import flwr as fl

class FederatedLearningClient:
    """Client for federated learning"""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize FL client with local model"""
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.y_train = None
    
    def prepare_data(self, df: pd.DataFrame) -> bool:
        """
        Prepare data from pandas DataFrame for training
        Returns: True if successful
        """
        if len(df) < 10:
            return False
        
        try:
            # Encode emotions
            emotion_map = {
                "happy": 0, "neutral": 1, "sad": 2,
                "anxious": 3, "angry": 4, "fear": 5, "disgust": 6
            }
            
            df_clean = df.copy()
            df_clean["face_emotion_encoded"] = df_clean["face_emotion"].map(
                lambda x: emotion_map.get(x, 1)
            )
            df_clean["voice_emotion_encoded"] = df_clean["voice_emotion"].map(
                lambda x: emotion_map.get(x, 1)
            )
            
            # Features and target
            feature_cols = ["heart_rate", "face_emotion_encoded", "voice_emotion_encoded"]
            self.X_train = df_clean[feature_cols].values
            
            # Target: high stress (HR > 100)
            self.y_train = (df_clean["heart_rate"] > 100).astype(int).values
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            
            return True
        
        except Exception as e:
            print(f"Error preparing data: {e}")
            return False
    
    def train_local_model(self) -> bool:
        """Train local model on client's data"""
        if self.X_train is None or len(self.X_train) < 5:
            return False
        
        try:
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(
                    n_estimators=5,
                    max_depth=3,
                    random_state=42
                )
            
            self.model.fit(self.X_train, self.y_train)
            return True
        
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def get_weights(self) -> np.ndarray:
        """Extract model weights for transmission"""
        if self.model is None:
            return np.array([])
        
        try:
            # For RandomForest, extract tree weights
            weights = []
            for tree in self.model.estimators_:
                weights.extend(tree.tree_.feature)
            return np.array(weights)
        except:
            return np.array([])
    
    def evaluate(self) -> float:
        """Evaluate local model"""
        if self.model is None or self.X_train is None:
            return 0.0
        
        return self.model.score(self.X_train, self.y_train)

class FLNumPyClient(fl.client.NumPyClient):
    """Flower NumPy client for federated learning"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize Flower client with data"""
        self.fl_client = FederatedLearningClient()
        self.df = df
        
        # Prepare and train initial model
        if self.fl_client.prepare_data(df):
            self.fl_client.train_local_model()
    
    def get_parameters(self, config: dict) -> Tuple:
        """Get model parameters"""
        if self.fl_client.model is None:
            return (np.array([]),)
        
        try:
            params = []
            
            # Extract weights from decision trees
            for tree in self.fl_client.model.estimators_:
                tree_weights = []
                tree_weights.append(tree.tree_.feature)
                tree_weights.append(tree.tree_.threshold)
                params.append(np.concatenate([tree_weights[0].astype(float), tree_weights[1]]))
            
            return tuple(params) if params else (np.array([]),)
        except:
            return (np.array([]),)
    
    def fit(self, parameters: Tuple, config: dict) -> Tuple:
        """Fit model locally"""
        # Train model
        self.fl_client.train_local_model()
        
        # Return updated parameters
        return self.get_parameters(config), len(self.fl_client.X_train), {}
    
    def evaluate(self, parameters: Tuple, config: dict) -> Tuple:
        """Evaluate model"""
        accuracy = self.fl_client.evaluate()
        loss = 1 - accuracy
        
        return loss, len(self.fl_client.X_train), {"accuracy": accuracy}

class ModelAggregator:
    """Aggregate models from multiple clients"""
    
    @staticmethod
    def average_weights(weights_list: list) -> np.ndarray:
        """Average weights from multiple clients"""
        if not weights_list:
            return np.array([])
        
        return np.mean(weights_list, axis=0)
    
    @staticmethod
    def save_aggregated_model(weights: np.ndarray, path: str):
        """Save aggregated weights"""
        try:
            with open(path, "wb") as f:
                pickle.dump(weights, f)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_aggregated_model(path: str) -> Optional[np.ndarray]:
        """Load aggregated weights"""
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

class FLConfig:
    """Configuration for federated learning"""
    
    NUM_ROUNDS = 3
    MIN_CLIENTS = 1
    MIN_AVAILABLE_CLIENTS = 1
    SERVER_ADDRESS = "0.0.0.0:8080"
    CLIENT_ADDRESS = "localhost:8080"
