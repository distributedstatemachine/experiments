I've audited the implementation and moved from simulation to a real-world deployment architecture for Basilica.

### Key Updates

1.  **Incentive & Slashing Logic**: Updated `BasilicaAggregator` in `sparseloco.py` to track worker rewards and slashes. Updates that fail SPoT (Sparse Proof of Training) verification are now rejected, and the worker is penalized.
2.  **Citadel Server**: Created `citadel_server.py`, a FastAPI-based aggregator that acts as the central coordination point for decentralized workers. It handles asynchronous sparse updates with staleness compensation and provides global weights to workers.
3.  **Basilica Orchestration**: Refined `run_basilica_experiment.py` to handle heterogeneous deployments (compressed vs. full replicas) and injected robust synchronization logic into the worker code.

### Next Steps

- [ ] **Integrate `basilica-python` SDK**: Finalize the rental and management logic to automate machine provisioning.
- [ ] **Live Experiment**: Conduct the first decentralized training run on real Basilica hardware.
- [ ] **Complex SPoT**: Implement random layer verification to reduce the aggregator's computational overhead during SPoT checks.

```python:citadel_server.py
@app.post("/push_update")
async def push_update(update_data: Dict[str, Any]):
    # ...
    success = aggregator.apply_sparse_update(
        bits, 
        indices, 
        scales, 
        update_data['worker_id'], 
        update_data['version']
    )
    # ...
```

I've updated `context/Quentin/STATE.md` to reflect these advancements. Ready to move toward live execution once you provide the signal.