I've updated the system to handle real-world internet constraints by implementing **Dynamic Density Adjustment** and a live **Metrics Dashboard** for the Citadel aggregator. The experiment has been scaled to 10+ heterogeneous workers on Basilica.

### Key Enhancements

1.  **Dynamic Density Adjustment (`basilica_training.py`)**:
    Workers now monitor synchronization latency. If network congestion increases (latency > 2s), the `SparseLoCo` density is automatically reduced (up to 0.1%) to minimize payload size. Conversely, it increases density when bandwidth allows, optimizing for convergence speed.

2.  **Real-Time Metrics Dashboard (`citadel_server.py`)**:
    Added a `/metrics` endpoint to the Citadel aggregator to track:
    - **Throughput**: Updates per second (UPS).
    - **Worker Health**: Real-time rewards and slashes per worker.
    - **Convergence**: Global model versioning and active worker counts.

3.  **Heterogeneous Scaling (`run_basilica_experiment.py`)**:
    The orchestration script now launches 10+ workers on Basilica GPUs, mixing compressed (Subspace Projection) and full-precision replicas to anchor the global model while maximizing participation from bandwidth-constrained nodes.

### Code Changes

```python:basilica_training.py
    def adjust_density(self, network_latency: float):
        """Dynamically adjusts sparsity based on network conditions."""
        if network_latency > 2.0: 
            self.density = max(0.001, self.density * 0.8)
        elif network_latency < 0.5:
            self.density = min(0.1, self.density * 1.1)
```

```python:citadel_server.py
@app.get("/metrics")
async def get_metrics():
    """Returns real-time convergence metrics for the dashboard."""
    # ... returns throughput, worker rewards, and active counts ...
```

The latest codebase has been pushed to `https://github.com/distributedstatemachine/experiments` via SSH. You can monitor the live experiment by querying the Citadel's `/metrics` endpoint.

```bash
# Monitor live convergence
curl <CITADEL_URL>/metrics
```