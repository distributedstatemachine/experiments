I've launched the live decentralized training experiment on Basilica.

### Actions Taken
1.  **Aggregator Started**: Launched `citadel_server.py` (FastAPI) to act as the central parameter server. It handles asynchronous sparse updates with staleness compensation and SPoT verification.
2.  **Workers Deployed**: Executed `run_basilica_experiment.py`, which is currently deploying heterogeneous workers (compressed and full-precision) to Basilica's GPU nodes.
3.  **Authentication**: Configured the system to use your `BASILICA_API_TOKEN` for both machine rental and secure communication between workers and the Citadel.
4.  **Bug Fix**: Resolved a `NameError` in `sparseloco.py` regarding the `Optional` type hint that was blocking the server start.

### Current Status
The experiment is running in the background. You can monitor the progress by checking the Citadel status endpoint:

```bash
curl -H "Authorization: Bearer $BASILICA_API_TOKEN" http://localhost:8000/status
```

I'll continue to monitor the worker joins and convergence metrics. `STATE.md` has been updated to reflect the transition from simulation to live experiment.