I've implemented the plan to enhance the decentralized training system on Basilica.

### Key Implementation Details

1.  **Dynamic Density Adjustment (`basilica_training.py`)**:
    Refined the `adjust_density` method in `HeterogeneousSparseLoCo`. Workers now monitor synchronization latency and adjust their `SparseLoCo` density:
    - **Congestion (>2s)**: Reduces density by 20% (min 0.1%) to shrink payload size.
    - **Bandwidth (<0.5s)**: Increases density by 10% (max 10%) to accelerate convergence.

2.  **Real-Time Metrics Dashboard (`citadel_server.py`)**:
    Integrated the `/metrics` endpoint into the FastAPI Citadel server. It provides live tracking of:
    - **Throughput**: Updates per second (UPS).
    - **Worker Health**: Real-time rewards, slashes, and active worker counts.
    - **Global State**: Current model version and elapsed training time.

3.  **Heterogeneous Scaling (`run_basilica_experiment.py`)**:
    Updated the orchestration script to scale the experiment to **12 heterogeneous workers** (mixing Subspace Projection and full-precision replicas). The monitoring loop now consumes the new `/metrics` endpoint for cleaner status reporting.

### Code References

```python:137:147:basilica_training.py
    def adjust_density(self, network_latency: float):
        """
        Dynamically adjusts sparsity based on network conditions.
        If latency is high (> 2s), we decrease density (increase sparsity) to reduce payload size.
        If latency is low (< 0.5s), we increase density to speed up convergence.
        """
        if network_latency > 2.0: 
            self.density = max(0.001, self.density * 0.8)
            print(f"[DynamicDensity] High latency ({network_latency:.2f}s). Density -> {self.density:.4f}")
        elif network_latency < 0.5:
            self.density = min(0.1, self.density * 1.1)
            print(f"[DynamicDensity] Low latency ({network_latency:.2f}s). Density -> {self.density:.4f}")
```

```python:40:64:citadel_server.py
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
```

### Next Steps
You can monitor the live experiment by querying the Citadel's metrics:
```bash
curl <CITADEL_URL>/metrics
```
I have also updated `context/Quentin/STATE.md` to reflect these completions.