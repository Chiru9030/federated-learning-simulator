# Federated Learning Simulator

A Python-based simulation of a Federated Learning environment demonstrating privacy-preserving decentralized AI model training.

## 📖 Complete Brief
Federated Learning is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging them. This repository simulates a central server coordinating with several edge nodes. In this simulation, each node trains a local model on its private data and only transmits the updated weights/gradients back to the central server. The server then uses the Federated Averaging (FedAvg) algorithm to aggregate these updates into a global model. This approach proves that a robust AI model can be trained collaboratively without ever exposing sensitive raw data to the network.

## ✨ Features
- **Decentralized Training Simulation:** Models edge nodes training independently on varying amounts of local dummy data.
- **Privacy-Preserving:** Demonstrates how raw data never leaves the edge node; only model gradients are shared.
- **Federated Averaging (FedAvg):** Implements the fundamental aggregation algorithm used in real-world federated systems.
- **Multi-Round Communication:** Simulates the iterative process of downloading global weights, running local epochs, and uploading updates.

## 🛠 Prerequisites
- Python 3.x
- `numpy`

## 🚀 How to Run

1. **Navigate to the simulator directory:**
   ```bash
   cd ./federated-learning-simulator/
   ```

2. **Install dependencies:**
   ```bash
   pip3 install -r requirements.txt
   ```
   *(Alternatively: `pip install numpy`)*

3. **Run the simulation:**
   ```bash
   python3 simulator.py
   ```

4. **Expected Output:**
   The CLI will output the progress of multiple communication rounds. You will see individual nodes reporting the size of their private data and the server confirming successful aggregation of weights:
   ```text
   [*] Initializing Federated Learning Decentralized Network...

   --- Communication Round 1 ---
   [Node 0] Trained on 342 private samples.
   [Node 1] Trained on 128 private samples.
   ...
   [+] Server aggregated local updates via FedAvg without seeing raw data.

   [SUCCESS] Global Model trained preserving edge privacy.
   ```
