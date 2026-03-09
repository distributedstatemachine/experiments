I've implemented the real-time convergence tracking and benchmarking plan for the Basilica experiment.

### Key Changes

1.  **Citadel Dashboard**: Integrated `ConvergenceTracker` into `citadel_server.py`. It now logs global model versions, worker rewards, and active participant counts, providing a live view of training progress.
2.  **Live Benchmarking**: Updated `run_basilica_experiment.py` to track throughput (updates per second) and synchronization latency. Workers now report step times and sync durations.
3.  **Scale Up**: Increased the default worker count to 10 in `run_basilica_experiment.py` to test the asynchronous aggregator's performance under higher load.
4.  **State Management**: Updated `context/Quentin/STATE.md` to reflect the completion of these tasks and the transition to scaling the experiment.

### Implementation Details

```python:citadel_server.py
class ConvergenceTracker:
    def log_step(self, global_version: int, worker_rewards: Dict[str, int], worker_slashes: Dict[str, int], active_workers: List[str]):
        elapsed = time.time() - self.start_time
        # ... logs version, rewards, and worker counts ...
        print(f"[Citadel Dashboard] T+{elapsed:.1f}s | Version: {global_version} | Workers: {len(active_workers)}")
```

```python:run_basilica_experiment.py
# Monitoring throughput in real-time
if elapsed > 0:
    throughput = current_version / elapsed
    print(f"Citadel Status: Version={current_version}, Throughput={throughput:.2f} updates/s, Active Workers={len(data['active_workers'])}")
```

### Next Steps
- **Dynamic Density**: Implement logic to adjust `SparseLoCo` density based on observed network congestion and worker staleness.
- **Heterogeneous Optimization**: Fine-tune the subspace projection ratios for different Basilica GPU models (e.g., A10G vs. H100) to maximize global convergence speed.