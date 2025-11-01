import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
from typing import Dict, Tuple, List
import pickle

# ============================================================================
# FEDERATED LEARNING SERVER
# ============================================================================

class SaveModelStrategy(FedAvg):
    """Custom Flower strategy that saves aggregated models"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_count = 0
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple],
        failures: List,
    ):
        """Aggregate model weights and save"""
        print(f"\n🔄 Federated Round {server_round} - Received {len(results)} client(s)")
        
        for i, (client_proxy, fit_res) in enumerate(results):
            print(f"   Client {i+1}: Contributed {len(fit_res.parameters.tensors)} tensors")
        
        # Call parent aggregation
        weights_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Save aggregated weights
        if weights_aggregated is not None:
            weights, metrics = weights_aggregated
            with open(f"aggregated_model_round_{server_round}.pkl", "wb") as f:
                pickle.dump(weights, f)
            print(f"✅ Aggregated model saved for round {server_round}")
        
        return weights_aggregated

def main():
    """Start Flower federated learning server"""
    
    print("=" * 70)
    print("🏥 POSTPARTUM WELLNESS MONITOR - FEDERATED LEARNING SERVER")
    print("=" * 70)
    print("\nServer is waiting for clients to connect...")
    print("Start your Streamlit app: streamlit run app.py")
    print("Or run dummy client: python dummyclient.py")
    print("\n" + "=" * 70)
    
    # Define server strategy
    strategy = SaveModelStrategy(
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=0,
    )
    
    # Server configuration
    config = ServerConfig(num_rounds=3)
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
