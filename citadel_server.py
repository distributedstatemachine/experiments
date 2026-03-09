from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from sparseloco import BasilicaAggregator, SPoTVerifier
from main_model import GlobalModel
from typing import List, Dict, Any
import os

app = FastAPI()

model = GlobalModel()
aggregator = BasilicaAggregator(model)
# verifier = SPoTVerifier(GlobalModel, density=0.03) # Optional: enable for strict mode

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
        "active_workers": list(aggregator.worker_versions.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
