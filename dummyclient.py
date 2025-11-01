import flwr as fl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pickle
from typing import Dict, Tuple

# ============================================================================
# DUMMY FEDERATED LEARNING CLIENT
# ============================================================================

class WellnessClient(fl.client.NumPyClient):
    """Simulated federated client for testing"""
    
    def __init__(self, model):
        self.model = model
        # Synthetic training data (simulate Mother A's data)
        X_train, y_train = make_classification(
            n_samples=100, n_features=3, n_informative=2,
            n_redundant=0, random_state=42
        )
        self.X_train = X_train
        self.y_train = y_train
    
    def get_parameters(self, config: Dict) -> Tuple:
        """Get model parameters"""
        params = [
            self.model.n_estimators,
            self.model.max_depth,
        ]
        return params
    
    def fit(self, parameters: Tuple, config: Dict) -> Tuple:
        """Train model locally"""
        print("🔄 Dummy Client: Training model on local synthetic data...")
        
        # Train on synthetic data
        self.model.fit(self.X_train, self.y_train)
        
        print("✅ Dummy Client: Local training complete")
        
        return self.get_parameters(config), len(self.X_train), {}
    
    def evaluate(self, parameters: Tuple, config: Dict) -> Tuple:
        """Evaluate model"""
        loss = float(1 - self.model.score(self.X_train, self.y_train))
        accuracy = self.model.score(self.X_train, self.y_train)
        
        return loss, len(self.X_train), {"accuracy": accuracy}

def main():
    """Start dummy federated client"""
    
    print("=" * 70)
    print("🏥 POSTPARTUM WELLNESS MONITOR - DUMMY FEDERATED CLIENT")
    print("=" * 70)
    print("\nConnecting to Federated Learning Server at localhost:8080...")
    print("This simulates 'Mother A' contributing to privacy-preserving training")
    print("=" * 70 + "\n")
    
    # Create dummy model
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    
    # Create and start client
    client = WellnessClient(model)
    
    fl.client.start_client(
        server_address="localhost:8080",
        client=client,
    )

if __name__ == "__main__":
    main()
