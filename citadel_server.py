from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from sparseloco import BasilicaAggregator, SPoTVerifier
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

    def log_step(self, global_version: int, worker_rewards: Dict[str, int], worker_slashes: Dict[str, int], active_workers: List[str]):
        elapsed = time.time() - self.start_time
        entry = {
            "timestamp": elapsed,
            "version": global_version,
            "rewards": worker_rewards.copy(),
            "slashes": worker_slashes.copy(),
            "num_workers": len(active_workers)
        }
        self.history.append(entry)
        self.total_updates += 1
        
        # Print a summary to console
        print(f"[Citadel Dashboard] T+{elapsed:.1f}s | Version: {global_version} | Workers: {len(active_workers)}")
        for wid, reward in worker_rewards.items():
            slashes = worker_slashes.get(wid, 0)
            print(f"  - {wid}: Rewards={reward}, Slashes={slashes}")

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

app = FastAPI()

model = GlobalModel()
aggregator = BasilicaAggregator(model)
tracker = ConvergenceTracker()

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
    Expects: worker_id, version, bits, indices, scales
    """
    try:
        # Convert lists back to tensors
        bits = [torch.tensor(b, dtype=torch.int8) for b in update_data['bits']]
        indices = [torch.tensor(i, dtype=torch.long) for i in update_data['indices']]
        scales = [torch.tensor(s, dtype=torch.float32) for s in update_data['scales']]
        
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
