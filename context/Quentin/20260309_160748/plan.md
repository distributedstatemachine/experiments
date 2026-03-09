I've prepared the system for a live decentralized training experiment on Basilica.

### Progress Update
1.  **Citadel Aggregator**: Launched the `citadel_server.py` using `uv` and `uvicorn`. It's now listening for sparse updates and serving global weights.
2.  **Worker Orchestration**: Hardened `run_basilica_experiment.py` to bundle all necessary modules (`sparseloco.py`, `basilica_training.py`, `main_model.py`) into the worker deployment. This ensures Basilica nodes have the full context needed for **Heterogeneous SparseLoCo**.
3.  **Model Definition**: Created `main_model.py` to provide a consistent architecture across the aggregator and all remote workers.
4.  **Verification**: The system is now ready to move from simulation to real-world execution.

### Next Steps
I am ready to run the experiment. Since this will involve renting real GPUs on Basilica and consuming credits, I'll proceed with `uv run run_basilica_experiment.py http://<your-public-ip>:8000` once you confirm the aggregator is reachable from the internet (e.g., via ngrok or a public IP).

```bash
# To start the experiment (ensure CITADEL_URL is reachable by Basilica nodes)
uv run run_basilica_experiment.py http://your-public-ip:8000
```

I'll monitor the `Citadel` status to track worker joins, sparse update frequency, and convergence.