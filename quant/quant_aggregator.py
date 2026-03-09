from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from quentin.sparseloco import BasilicaAggregator, SPoTVerifier
from quant.quant_model import TimeSeriesFoundationModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI()

# Initialize the SOTA Quant Model
model = TimeSeriesFoundationModel()
# Aggregator with SPoT verification
aggregator = BasilicaAggregator(
    model, 
    verifier=SPoTVerifier(TimeSeriesFoundationModel, density=0.03),
    use_lookahead=True,
    use_polyak=True
)

@app.get("/weights")
async def get_weights():
    """Returns the current global model weights."""
    weights = aggregator.get_global_weights()
    return {"weights": [w.tolist() for w in weights], "version": aggregator.global_version}

@app.post("/push_update")
async def push_update(update_data: Dict[str, Any]):
    """Receives and applies sparse updates from workers."""
    try:
        bits = [torch.tensor(b, dtype=torch.int8) for b in update_data['bits']]
        indices = [torch.tensor(i, dtype=torch.long) for i in update_data['indices']]
        scales = [torch.tensor(s, dtype=torch.float32) for s in update_data['scales']]
        
        verification_data = update_data.get('verification_data')
        if verification_data:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
