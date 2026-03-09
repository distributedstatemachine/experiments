from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from sparseloco import BasilicaAggregator, SPoTVerifier
from zk_spot import ZKSPoTVerifier
from main_model import GlobalModel
from typing import List, Dict, Any
import os

import time
from typing import List, Dict, Any
import torch

class ConvergenceTracker:
    """
    Tracks convergence and worker rewards for decentralized training.
    """
    def __init__(self):
        self.history = []
        self.start_time = time.time()
        self.total_updates = 0

    def log_step(self, global_version: int, worker_rewards: Dict[str, float], worker_slashes: Dict[str, int], active_workers: List[str]):
        elapsed = time.time() - self.start_time
        entry = {
            "timestamp": elapsed,
            "version": global_version,
            "rewards": {k: float(v) for k, v in worker_rewards.items()},
            "slashes": worker_slashes.copy(),
            "num_workers": len(active_workers)
        }
        self.history.append(entry)
        self.total_updates += 1
        
        # Print a summary to console
        print(f"[Citadel Dashboard] T+{elapsed:.1f}s | Version: {global_version} | Workers: {len(active_workers)}")
        for wid, reward in worker_rewards.items():
            slashes = worker_slashes.get(wid, 0)
            print(f"  - {wid}: Rewards={reward:.2f}, Slashes={slashes}")

app = FastAPI()

model = GlobalModel()
aggregator = BasilicaAggregator(model, verifier=SPoTVerifier(GlobalModel, density=0.03))
zk_verifier = ZKSPoTVerifier(GlobalModel, density=0.03)
tracker = ConvergenceTracker()

@app.get("/metrics")
async def get_metrics():
    """Returns real-time convergence metrics for the dashboard."""
    if not tracker.history:
        return {"error": "No data yet"}
    
    latest = tracker.history[-1]
    # Calculate throughput (updates/sec)
    throughput = latest['version'] / latest['timestamp'] if latest['timestamp'] > 0 else 0
    
    return {
        "elapsed_seconds": latest['timestamp'],
        "global_version": latest['version'],
        "throughput_ups": throughput,
        "active_workers": latest['num_workers'],
        "total_rewards": sum(latest['rewards'].values()),
        "total_slashes": sum(latest['slashes'].values()),
        "worker_details": [
            {
                "worker_id": wid,
                "reward": latest['rewards'].get(wid, 0),
                "slashes": latest['slashes'].get(wid, 0)
            } for wid in latest['rewards'].keys()
        ]
    }

@app.get("/weights")
async def get_weights():
    """Returns the current global model weights."""
    weights = aggregator.get_global_weights()
    # Convert tensors to lists for JSON serialization
    return {"weights": [w.tolist() for w in weights], "version": aggregator.global_version}

@app.post("/push_update")
async def push_update(update_data: Dict[str, Any]):
    """
    Receives sparse updates from workers.
    Expects: worker_id, version, bits, indices, scales, verification_data (optional)
    """
    try:
        # Convert lists back to tensors
        bits = [torch.tensor(b, dtype=torch.int8) for b in update_data['bits']]
        indices = [torch.tensor(i, dtype=torch.long) for i in update_data['indices']]
        scales = [torch.tensor(s, dtype=torch.float32) for s in update_data['scales']]
        
        verification_data = update_data.get('verification_data')
        zk_proof = update_data.get('zk_proof')
        
        if zk_proof:
            # ZK-SPoT Verification
            # Reconstruct tensors in zk_proof
            zk_proof['proof']['sparse_bits'] = [torch.tensor(b, dtype=torch.int8) for b in zk_proof['proof']['sparse_bits']]
            zk_proof['proof']['sparse_indices'] = [torch.tensor(i, dtype=torch.long) for i in zk_proof['proof']['sparse_indices']]
            zk_proof['proof']['sparse_scales'] = [torch.tensor(s, dtype=torch.float32) for s in zk_proof['proof']['sparse_scales']]
            zk_proof['proof']['data_shard'] = (
                torch.tensor(zk_proof['proof']['data_shard'][0], dtype=torch.float32),
                torch.tensor(zk_proof['proof']['data_shard'][1], dtype=torch.float32)
            )
            zk_proof['public_inputs']['initial_weights'] = [torch.tensor(w, dtype=torch.float32) for w in zk_proof['public_inputs']['initial_weights']]
            
            is_valid = zk_verifier.verify_proof(zk_proof['proof'], zk_proof['public_inputs'])
            if not is_valid:
                # Log failure and slash
                aggregator.slash_worker(update_data['worker_id'])
                raise HTTPException(status_code=403, detail="ZK-SPoT verification failed")
            
            # If ZK-SPoT passes, we proceed with the update
            success = aggregator.apply_sparse_update(
                zk_proof['proof']['sparse_bits'], 
                zk_proof['proof']['sparse_indices'], 
                zk_proof['proof']['sparse_scales'], 
                update_data['worker_id'], 
                update_data['version'],
                verification_data=None # Already verified via ZK
            )
        elif verification_data:
            # Reconstruct tensors in verification_data
            verification_data['initial_weights'] = [torch.tensor(w) for w in verification_data['initial_weights']]
            verification_data['data_shard'] = (
                torch.tensor(verification_data['data_shard'][0]),
                torch.tensor(verification_data['data_shard'][1])
            )

            success = aggregator.apply_sparse_update(
                bits, 
                indices, 
                scales, 
                update_data['worker_id'], 
                update_data['version'],
                verification_data=verification_data
            )
        else:
            # Standard update without verification (less reward)
            success = aggregator.apply_sparse_update(
                bits, 
                indices, 
                scales, 
                update_data['worker_id'], 
                update_data['version']
            )
        
        if not success:
            raise HTTPException(status_code=403, detail="Update rejected by verification/incentive logic")
            
        # Log to tracker
        tracker.log_step(
            aggregator.global_version,
            aggregator.worker_rewards,
            aggregator.worker_slashes,
            list(aggregator.worker_versions.keys())
        )
            
        return {"status": "success", "global_version": aggregator.global_version}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/slash")
async def slash_worker(data: Dict[str, str]):
    """Slashes a worker manually."""
    worker_id = data.get("worker_id")
    if not worker_id:
        raise HTTPException(status_code=400, detail="worker_id required")
    
    aggregator.slash_worker(worker_id)
    
    # Log to tracker
    tracker.log_step(
        aggregator.global_version,
        aggregator.worker_rewards,
        aggregator.worker_slashes,
        list(aggregator.worker_versions.keys())
    )
    
    return {"status": "success", "worker_id": worker_id, "total_slashes": aggregator.worker_slashes.get(worker_id, 0)}

@app.get("/status")
async def get_status():
    """Returns the status of the training run."""
    return {
        "global_version": aggregator.global_version,
        "worker_rewards": aggregator.worker_rewards,
        "worker_slashes": aggregator.worker_slashes,
        "active_workers": list(aggregator.worker_versions.keys()),
        "history": tracker.history[-10:] # Return last 10 entries
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
