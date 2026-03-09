import os
import torch
import torch.nn as nn
from basilica import BasilicaClient
from quentin.basilica_training import HeterogeneousSparseLoCo
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
            
            worker_code = self._generate_worker_code(is_compressed, name)
            
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
                        timeout=600 # Increased timeout for worker deployment
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

    def _generate_worker_code(self, is_compressed: bool, name: str):
        # Read the required files to bundle them from the quentin/ and shared/ directories
        with open("quentin/basilica_training.py", "r") as f:
            basilica_training_code = f.read()
        with open("quentin/sparseloco.py", "r") as f:
            sparseloco_code = f.read()
        with open("shared/main_model.py", "r") as f:
            main_model_code = f.read()
        with open("quentin/zk_spot.py", "r") as f:
            zk_spot_code = f.read()

        return f"""
import os
import sys

# Set environment variables manually for the worker
os.environ["BASILICA_API_TOKEN"] = {repr(API_KEY)}
os.environ["CITADEL_URL"] = {repr(self.config.get("CITADEL_URL"))}
os.environ["WORKER_ID"] = {repr(name)}
os.environ["IS_COMPRESSED"] = {repr(str(is_compressed))}
os.environ["SPARSE_DENSITY"] = "0.03"
with open("basilica_training.py", "w") as f:
    f.write({repr(basilica_training_code)})
with open("sparseloco.py", "w") as f:
    f.write({repr(sparseloco_code)})
with open("main_model.py", "w") as f:
    f.write({repr(main_model_code)})
with open("zk_spot.py", "w") as f:
    f.write({repr(zk_spot_code)})

import torch
import torch.nn as nn
from quentin.basilica_training import HeterogeneousSparseLoCo
from zk_spot import PrivacyAwareWorker
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
        # Initialize PrivacyAwareWorker for ZK-SPoT
        # In a real scenario, data_shard would be real data
        self.privacy_worker = PrivacyAwareWorker(
            WORKER_ID, 
            model_class, 
            (torch.randn(16, 128), torch.randn(16, 128)) 
        )

    def push_update(self, update, verification_data=None, zk_proof=None):
        # Convert tensors to lists for JSON
        serializable_update = {
            "worker_id": WORKER_ID,
            "version": self.version,
            "bits": [u['bits'].tolist() if u else [] for u in update['updates']],
            "indices": [u['indices'].tolist() if u else [] for u in update['updates']],
            "scales": [u['scale'].tolist() if u else [0.0, 0.0] for u in update['updates']],
            "layer_norms": update.get('layer_norms', [])
        }

        if zk_proof:
            # Serialize ZK proof
            serializable_update["zk_proof"] = {
                "proof": {
                    "commitment": zk_proof['proof']['commitment'],
                    "sparse_bits": [b.tolist() for b in zk_proof['proof']['sparse_bits']],
                    "sparse_indices": [i.tolist() for i in zk_proof['proof']['sparse_indices']],
                    "sparse_scales": [s.tolist() for s in zk_proof['proof']['sparse_scales']],
                    "data_shard": [zk_proof['proof']['data_shard'][0].tolist(), zk_proof['proof']['data_shard'][1].tolist()]
                },
                "public_inputs": {
                    "initial_weights": [w.tolist() for w in zk_proof['public_inputs']['initial_weights']],
                    "data_hash": zk_proof['public_inputs']['data_hash'],
                    "h_steps": zk_proof['public_inputs']['h_steps'],
                    "lr": zk_proof['public_inputs']['lr'],
                    "seed": zk_proof['public_inputs']['seed']
                }
            }
        elif verification_data:
            serializable_update["verification_data"] = {
                "initial_weights": [w.tolist() for w in verification_data['initial_weights']],
                "data_shard": [verification_data['data_shard'][0].tolist(), verification_data['data_shard'][1].tolist()],
                "h_steps": verification_data['h_steps'],
                "lr": verification_data['lr'],
                "seed": verification_data['seed']
            }
        
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
                # Use Polyak weights for smoother convergence if available
                resp = requests.get(f"{{CITADEL_URL}}/weights?use_polyak=true", timeout=60)
                resp.raise_for_status()
                data = resp.json()
                weights = [torch.tensor(w) for w in data['weights']]
                version = data.get('version', 0)
                return weights, version
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
        local_lr = 1e-4
        
        import threading
        import queue

        # Queue for background synchronization
        sync_queue = queue.Queue(maxsize=1)
        
        def background_sync():
            while True:
                update_task = sync_queue.get()
                if update_task is None: break
                
                update, zk_proof = update_task
                try:
                    sync_start = time.time()
                    
                    # Gradient-Aware Communication Scheduling (GACS):
                    # Sort layers by gradient norm and only send those above a threshold
                    # or limit the number of layers sent to save bandwidth.
                    if 'layer_norms' in update:
                        norms = torch.tensor(update['layer_norms'])
                        # Only send layers in the top 50% of gradient norms
                        threshold = torch.median(norms)
                        for i, norm in enumerate(update['layer_norms']):
                            if norm < threshold:
                                update['updates'][i] = None
                    
                    self.push_update(update, zk_proof=zk_proof)
                    global_weights, global_version = self.pull_weights()
                    sync_time = time.time() - sync_start
                    
                    # We can't apply weights directly here as it might interfere with training
                    # Instead, we store them for the main thread to pick up
                    self.pending_weights = (global_weights, global_version, sync_time)
                    print(f"[BackgroundSync] Completed in {sync_time:.4f}s")
                except Exception as e:
                    print(f"[BackgroundSync] Failed: {e}")
                finally:
                    sync_queue.task_done()

        self.pending_weights = None
        sync_thread = threading.Thread(target=background_sync, daemon=True)
        sync_thread.start()

        while True:
            start_time = time.time()
            # Generate synthetic data for experiment
            data = torch.randn(16, 128)
            target = torch.randn(16, 128)
            
            # Check for pending weights from background sync
            if self.pending_weights:
                weights, version, sync_time = self.pending_weights
                self.optimizer.synchronize(weights, network_latency=sync_time, global_version=version)
                self.pending_weights = None
                print(f"Applied background weights | Version: {version}")

            # For SPoT, we need to capture the state before local steps
            initial_weights = [p.data.clone().detach() for p in self.model.parameters()]
            seed = int(time.time() * 1000) % 100000
            torch.manual_seed(seed)

            self.model.train()
            # Use local_step which implements SAM
            loss = self.optimizer.local_step(data, target, local_lr)
            
            step_time = time.time() - start_time
            
            if self.version % 10 == 0: # More frequent updates for testing
                print(f"Step {self.version}, Loss: {loss:.4f}, Step Time: {step_time:.4f}s")
                update = self.optimizer.get_sparse_update()
                
                # Generate ZK-SPoT proof (Privacy-Preserving)
                zk_proof = self.privacy_worker.generate_proof(
                    initial_weights,
                    [u['bits'] if u else torch.tensor([], dtype=torch.int8) for u in update['updates']],
                    [u['indices'] if u else torch.tensor([], dtype=torch.long) for u in update['updates']],
                    [u['scale'] if u else torch.tensor([0.0, 0.0], dtype=torch.float32) for u in update['updates']],
                    h_steps=1,
                    lr=local_lr,
                    seed=seed
                )

                # Try to put in sync queue, if full, we skip (still training)
                try:
                    sync_queue.put_nowait((update, zk_proof))
                except queue.Full:
                    print("Sync queue full, skipping this update to maintain compute throughput.")
            
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
        self.launch_workers(num_workers=12) # Scaled to 12 heterogeneous workers
        
        # 3. Monitor and Aggregate (The Citadel/Aggregator logic)
        print(f"Experiment running with Citadel at {citadel_url}. Monitoring workers...")
        
        start_time = time.time()
        last_version = 0
        
        while True:
            try:
                # Query the new /metrics endpoint
                resp = requests.get(f"{self.config['CITADEL_URL']}/metrics", timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        print(f"Waiting for data from Citadel...")
                    else:
                        current_version = data['global_version']
                        throughput = data['throughput_ups']
                        active = data['active_workers']
                        print(f"[Monitor] Version: {current_version} | UPS: {throughput:.2f} | Workers: {active}")
                        
                        if current_version > last_version:
                            last_version = current_version
                else:
                    print(f"Citadel Metrics Error: {resp.status_code}")
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
    import sys
    import os
    # Add parent directory to path to import shared modules
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from shared.main_model import GlobalModel
    from quentin.deploy_citadel import deploy_citadel
    
    # Deploy Citadel first
    citadel_url = deploy_citadel()

    trainer = BasilicaTrainer(GlobalModel, {"CITADEL_URL": citadel_url})
    try:
        trainer.run_experiment(citadel_url)
    except KeyboardInterrupt:
        print("Cleaning up...")
        trainer.cleanup()
