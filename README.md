# Federated Learning Privacy Simulator

A Python demonstration of Privacy-Preserving Machine Learning using the **Federated Averaging (FedAvg)** algorithm.

## Overview
Traditional AI requires centralizing massive amounts of user data, creating massive security and privacy risks (e.g., GDPR violations). 

Federated Learning solves this by deploying an initial model to distributed edge devices (like smartphones or hospitals). The devices train the model locally on their private data. Then, they send **only the mathematical gradient updates** back to the central server. The server aggregates these updates to refine the global model.

The raw data never leaves the device.

## Usage
```bash
pip install -r requirements.txt
python simulator.py
```
