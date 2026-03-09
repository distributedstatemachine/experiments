I've implemented the optimized decentralized training plan. The system now features **Dynamic Density Adjustment** to handle network congestion and **Adaptive SPoT Verification** to minimize aggregator overhead.

### Key Changes

1.  **Dynamic Density Adjustment (`basilica_training.py` & `run_basilica_experiment.py`)**:
    Workers now track synchronization latency. If sync takes >2s, density is reduced (down to 0.1%) to shrink payloads. If <0.5s, density increases (up to 10%) to accelerate convergence.
    
```python:137:148:basilica_training.py
    def adjust_density(self, network_latency: float):
        """
        Dynamically adjusts sparsity based on network conditions.
        """
        if network_latency > 2.0: 
            self.density = max(0.001, self.density * 0.8)
            print(f"[DynamicDensity] High latency ({network_latency:.2f}s). Density -> {self.density:.4f}")
        elif network_latency < 0.5:
            self.density = min(0.1, self.density * 1.1)
            print(f"[DynamicDensity] Low latency ({network_latency:.2f}s). Density -> {self.density:.4f}")
```

2.  **Adaptive SPoT Verification (`sparseloco.py`)**:
    The `BasilicaAggregator` now randomly selects ~10% of layers for verification. This maintains a strong probabilistic deterrent against cheating while reducing the aggregator's computational load by 90%.

3.  **Real-Time Metrics & SPoT Integration (`citadel_server.py`)**:
    Added a `/metrics` endpoint for tracking UPS, worker health, and slashes. The `/push_update` endpoint now handles the reconstruction of tensors for SPoT verification.

4.  **Heterogeneous Scaling (`run_basilica_experiment.py`)**:
    Updated the worker orchestration to support 12 heterogeneous workers. Workers now generate and transmit the necessary deterministic state (seeds/initial weights) for the aggregator to perform SPoT.

The experiment is live. You can monitor the decentralized training progress:
```bash
curl <CITADEL_URL>/metrics
```

I've also updated `context/Quentin/STATE.md` to reflect these advancements.