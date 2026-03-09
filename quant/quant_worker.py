import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import time
import uuid
from typing import List, Optional
from quant.quant_model import TimeSeriesFoundationModel
from quentin.basilica_training import HeterogeneousSparseLoCo

class QuantWorker:
    """
    Decentralized worker for training the quant model.
    Uses SparseLoCo for communication-efficient updates.
    """
    def __init__(self, aggregator_url: str, worker_id: str = None, is_compressed: bool = False):
        self.aggregator_url = aggregator_url
        self.worker_id = worker_id or f"quant-worker-{uuid.uuid4().hex[:8]}"
        self.model = TimeSeriesFoundationModel()
        # Using the advanced optimizer from basilica_training
        self.optimizer = HeterogeneousSparseLoCo(self.model, is_compressed=is_compressed, d_model=256)
        self.version = 0

    def pull_weights(self):
        """Pulls global weights from the aggregator."""
        try:
            response = requests.get(f"{self.aggregator_url}/weights")
            if response.status_code == 200:
                data = response.json()
                weights = [torch.tensor(w) for w in data['weights']]
                self.version = data['version']
                self.optimizer.synchronize(weights, global_version=self.version)
                print(f"[{self.worker_id}] Synced to version {self.version}")
            else:
                print(f"[{self.worker_id}] Failed to pull weights: {response.status_code}")
        except Exception as e:
            print(f"[{self.worker_id}] Error pulling weights: {e}")

    def train_step(self, inputs, targets, lr=1e-4, use_fedsam=True):
        """Performs a local training step."""
        loss = self.optimizer.local_step(inputs, targets, lr, use_fedsam=use_fedsam)
        return loss

    def push_update(self, inputs, targets, h_steps=10, lr=1e-4):
        """Pushes a sparse update to the aggregator."""
        update_package = self.optimizer.get_sparse_update()
        updates = update_package['updates']
        
        payload = {
            "worker_id": self.worker_id,
            "version": self.version,
            "bits": [u['bits'].tolist() for u in updates if u],
            "indices": [u['indices'].tolist() for u in updates if u],
            "scales": [u['scale'].tolist() for u in updates if u],
            "verification_data": {
                "initial_weights": [p.tolist() for p in self.optimizer.initial_weights],
                "data_shard": (inputs.tolist(), targets.tolist()),
                "h_steps": h_steps,
                "lr": lr,
                "seed": 42,
                "is_compressed": self.optimizer.is_compressed,
                "layer_norms": update_package['layer_norms']
            }
        }
        
        try:
            response = requests.post(f"{self.aggregator_url}/push_update", json=payload)
            if response.status_code == 200:
                print(f"[{self.worker_id}] Update accepted. New version: {response.json()['global_version']}")
                return True
            else:
                print(f"[{self.worker_id}] Update rejected: {response.text}")
                return False
        except Exception as e:
            print(f"[{self.worker_id}] Error pushing update: {e}")
            return False

if __name__ == "__main__":
    worker = QuantWorker("http://localhost:8000")
