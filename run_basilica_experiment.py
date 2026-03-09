import os
import torch
import torch.nn as nn
from basilica import BasilicaClient
from basilica_training import HeterogeneousSparseLoCo
import time
import requests
import json
import numpy as np

# Basilica API Key from environment
API_KEY = os.getenv("BASILICA_API_TOKEN")

class BasilicaTrainer:
    def __init__(self, model_class, config):
        self.client = BasilicaClient(api_key=API_KEY)
        self.model_class = model_class
        self.config = config
        self.deployments = []

    def launch_workers(self, num_workers: int, compressed_ratio: float = 0.5):
        """
        Launches heterogeneous workers on Basilica.
        """
        num_compressed = int(num_workers * compressed_ratio)
        
        for i in range(num_workers):
            is_compressed = i < num_compressed
            name = f"worker-{i}-{'compressed' if is_compressed else 'full'}"
            
            print(f"Deploying {name} to Basilica...")
            
            worker_code = self._generate_worker_code(is_compressed)
            
            # Use Basilica SDK to deploy
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    deployment = self.client.deploy(
                        name=name,
                        source=worker_code,
                        gpu_count=1,
                        gpu_models=["A10G"],
                        pip_packages=["torch", "numpy", "basilica-sdk", "requests", "uv"],
                        env_vars={
                            "BASILICA_API_TOKEN": API_KEY,
                            "CITADEL_URL": self.config.get("CITADEL_URL", "http://localhost:8000"),
                            "WORKER_ID": name,
                            "IS_COMPRESSED": str(is_compressed),
                            "SPARSE_DENSITY": "0.03" # Initial density
                        }
                    )
                    self.deployments.append(deployment)
                    print(f"Worker {i} live at: {deployment.url}")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {name}: {e}")
                    if attempt == max_retries - 1:
                        print(f"Failed to deploy {name} after {max_retries} attempts.")
                    else:
                        time.sleep(5)

    def _generate_worker_code(self, is_compressed: bool):
        # Read the required files to bundle them
        with open("basilica_training.py", "r") as f:
            basilica_training_code = f.read()
        with open("sparseloco.py", "r") as f:
            sparseloco_code = f.read()
        with open("main_model.py", "r") as f:
            main_model_code = f.read()

        return f"""
import os
import sys

# Write bundled modules to disk so they can be imported
with open("basilica_training.py", "w") as f:
    f.write({repr(basilica_training_code)})
with open("sparseloco.py", "w") as f:
    f.write({repr(sparseloco_code)})
with open("main_model.py", "w") as f:
    f.write({repr(main_model_code)})

import torch
import torch.nn as nn
from basilica_training import HeterogeneousSparseLoCo
import requests
import json
import time

# Config from environment
API_KEY = os.getenv("BASILICA_API_TOKEN")
CITADEL_URL = os.getenv("CITADEL_URL")
WORKER_ID = os.getenv("WORKER_ID")
IS_COMPRESSED = os.getenv("IS_COMPRESSED") == "True"
SPARSE_DENSITY = float(os.getenv("SPARSE_DENSITY", "0.03"))

class Worker:
    def __init__(self, model_class, is_compressed, density):
        self.model = model_class()
        self.optimizer = HeterogeneousSparseLoCo(
            self.model, 
            is_compressed=is_compressed,
            density=density
        )
        self.version = 0

    def push_update(self, update):
        # Convert tensors to lists for JSON
        serializable_update = {{
            "worker_id": WORKER_ID,
            "version": self.version,
            "bits": [u['bits'].tolist() if u else [] for u in update['updates']],
            "indices": [u['indices'].tolist() if u else [] for u in update['updates']],
            "scales": [u['scale'].item() if u else 0.0 for u in update['updates']]
        }}
        
        max_retries = 10
        for attempt in range(max_retries):
            try:
                resp = requests.post(f"{{CITADEL_URL}}/push_update", json=serializable_update, timeout=60)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"Push update attempt {{attempt + 1}} failed: {{e}}")
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (torch.rand(1).item() * 2)
                time.sleep(wait_time)

    def pull_weights(self):
        max_retries = 10
        for attempt in range(max_retries):
            try:
                resp = requests.get(f"{{CITADEL_URL}}/weights", timeout=60)
                resp.raise_for_status()
                data = resp.json()
                weights = [torch.tensor(w) for w in data['weights']]
                return weights
            except Exception as e:
                print(f"Pull weights attempt {{attempt + 1}} failed: {{e}}")
                if attempt == max_retries - 1:
                    raise
                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + (torch.rand(1).item() * 2)
                time.sleep(wait_time)

    def train(self):
        print(f"Worker {WORKER_ID} starting training loop...")
        criterion = nn.MSELoss()
        # Use a smaller LR for stability in decentralized setting
        local_opt = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        
        while True:
            start_time = time.time()
            # Generate synthetic data for experiment
            data = torch.randn(16, 128)
            target = torch.randn(16, 128)
            
            self.model.train()
            local_opt.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            local_opt.step()
            
            step_time = time.time() - start_time
            
            if self.version % 10 == 0: # More frequent updates for testing
                print(f"Step {self.version}, Loss: {loss.item():.4f}, Step Time: {step_time:.4f}s")
                update = self.optimizer.get_sparse_update()
                try:
                    sync_start = time.time()
                    self.push_update(update)
                    global_weights = self.pull_weights()
                    self.optimizer.synchronize(global_weights)
                    sync_time = time.time() - sync_start
                    print(f"Sync successful in {sync_time:.4f}s | Step Time: {step_time:.4f}s")
                    
                    # Dynamic density adjustment based on sync time
                    self.optimizer.adjust_density(sync_time)
                    
                    # Reset local optimizer for new weights
                    local_opt = torch.optim.SGD(self.model.parameters(), lr=1e-4)
                except Exception as e:
                    print(f"Sync failed: {e}")
            
            self.version += 1

if __name__ == "__main__":
    from main_model import GlobalModel
    worker = Worker(GlobalModel, IS_COMPRESSED, SPARSE_DENSITY)
    worker.train()
"""

    def run_experiment(self, citadel_url: str):
        # 1. Update config with live Citadel URL
        self.config["CITADEL_URL"] = citadel_url
        
        # 2. Launch workers
        self.launch_workers(num_workers=10) # Scaled up to 10+ workers
        
        # 3. Monitor and Aggregate (The Citadel/Aggregator logic)
        print(f"Experiment running with Citadel at {citadel_url}. Monitoring workers...")
        
        start_time = time.time()
        last_version = 0
        
        while True:
            try:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                resp = requests.get(f"{self.config['CITADEL_URL']}/status", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    current_version = data['global_version']
                    elapsed = time.time() - start_time
                    
                    # Calculate throughput: updates per second
                    if elapsed > 0:
                        throughput = current_version / elapsed
                        print(f"Citadel Status: Version={current_version}, Throughput={throughput:.2f} updates/s, Active Workers={len(data['active_workers'])}")
                    
                    if current_version > last_version:
                        print(f"New updates detected! Progressing at {throughput:.2f} updates/s")
                        last_version = current_version
                else:
                    print(f"Citadel Status Error: {resp.status_code} - {resp.text}")
                time.sleep(10)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(10)
        
    def cleanup(self):
        for d in self.deployments:
            try:
                self.client.delete(d.id)
            except:
                pass

if __name__ == "__main__":
    from main_model import GlobalModel
    from deploy_citadel import deploy_citadel
    
    # Deploy Citadel first
    citadel_url = deploy_citadel()

    trainer = BasilicaTrainer(GlobalModel, {"CITADEL_URL": citadel_url})
    try:
        trainer.run_experiment(citadel_url)
    except KeyboardInterrupt:
        print("Cleaning up...")
        trainer.cleanup()
