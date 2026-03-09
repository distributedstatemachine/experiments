I've updated the system to include real-time convergence tracking and benchmarking for the live Basilica experiment.

### Key Updates

1.  **Citadel Dashboard**: Implemented `ConvergenceTracker` in `citadel_server.py` to log global model versions, worker rewards, and active participant counts. This provides a live view of the training progress.
2.  **Live Benchmarking**: Updated `run_basilica_experiment.py` to track throughput (updates per second) and synchronization latency. Workers now report step times and sync durations, allowing us to measure the overhead of decentralized communication.
3.  **State Management**: Updated `context/Quentin/STATE.md` to reflect the completion of the dashboard and benchmarking tasks.

### Next Steps
- **Scale Up**: Increase the worker count to 10+ to test the limits of the asynchronous aggregator.
- **Dynamic Density**: Implement logic to adjust `SparseLoCo` density based on observed network congestion and worker staleness.

```python:citadel_server.py
class ConvergenceTracker:
    def log_step(self, global_version: int, worker_rewards: Dict[str, int], worker_slashes: Dict[str, int], active_workers: List[str]):
        # ... logs elapsed time, version, rewards, and worker counts ...
        print(f"[Citadel Dashboard] T+{elapsed:.1f}s | Version: {global_version} | Workers: {len(active_workers)}")
```

```python:run_basilica_experiment.py
# Monitoring throughput in real-time
if elapsed > 0:
    throughput = current_version / elapsed
    print(f"Citadel Status: Version={current_version}, Throughput={throughput:.2f} updates/s")
```