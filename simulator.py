import numpy as np

class FederatedNode:
    def __init__(self, node_id, data_size):
        self.node_id = node_id
        # Local model weights (W) and biases (b)
        self.weights = np.random.randn(2)
        self.bias = np.random.randn(1)
        self.data_size = data_size
        
    def local_train(self, global_weights, global_bias):
        # Download global weights
        self.weights = global_weights
        self.bias = global_bias
        
        # Simulate local training on private data (adding slight gradients)
        grad_w = np.random.normal(0, 0.01, size=self.weights.shape)
        grad_b = np.random.normal(0, 0.01, size=self.bias.shape)
        
        self.weights -= grad_w
        self.bias -= grad_b
        
        # Return ONLY the updated weights, NOT the private data
        return self.weights, self.bias, self.data_size

def federated_averaging():
    print("[*] Initializing Federated Learning Decentralized Network...")
    
    nodes = [FederatedNode(i, data_size=np.random.randint(100, 500)) for i in range(5)]
    global_weights = np.zeros(2)
    global_bias = np.zeros(1)
    
    rounds = 3
    for r in range(rounds):
        print(f"\n--- Communication Round {r+1} ---")
        updates = []
        
        # 1. Distribute model & train locally
        for node in nodes:
            w_updated, b_updated, count = node.local_train(np.copy(global_weights), np.copy(global_bias))
            updates.append((w_updated, b_updated, count))
            print(f"[Node {node.node_id}] Trained on {count} private samples.")
            
        # 2. Secure Aggregation (Weighted Average) at the Central Server
        total_data = sum([u[2] for u in updates])
        
        new_global_w = np.zeros_like(global_weights)
        new_global_b = np.zeros_like(global_bias)
        
        for w, b, count in updates:
            weight = count / total_data
            new_global_w += w * weight
            new_global_b += b * weight
            
        global_weights = new_global_w
        global_bias = new_global_b
        print("[+] Server aggregated local updates via FedAvg without seeing raw data.")
        
    print("\n[SUCCESS] Global Model trained preserving edge privacy.")

if __name__ == "__main__":
    federated_averaging()
