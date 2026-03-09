I've optimized the decentralized training system on Basilica by implementing **Dynamic Density Adjustment** and **Adaptive SPoT Verification**. The system now automatically scales its communication overhead based on real-time network latency and reduces aggregator load by subsampling verification layers.

### Key Enhancements

1.  **Dynamic Density Adjustment (`basilica_training.py`)**:
    Workers now monitor synchronization latency. If network congestion increases (latency > 2s), the `SparseLoCo` density is automatically reduced (down to 0.1%) to minimize payload size. Conversely, it increases density when bandwidth allows, optimizing for convergence speed.

2.  **Adaptive SPoT Verification (`sparseloco.py`)**:
    The `BasilicaAggregator` now uses adaptive layer subsampling for **Sparse Proof of Training (SPoT)**. Instead of verifying all layers, it randomly selects ~10% of layers to verify, significantly reducing the computational overhead on the Citadel aggregator while maintaining a strong probabilistic deterrent against cheating.

3.  **Real-Time Metrics Dashboard (`citadel_server.py`)**:
    Integrated a `/metrics` endpoint to track:
    - **Throughput**: Updates per second (UPS).
    - **Worker Health**: Real-time rewards and slashes per worker.
    - **Convergence**: Global model versioning and active worker counts.

4.  **Heterogeneous Scaling (`run_basilica_experiment.py`)**:
    The orchestration script has been scaled to **12 heterogeneous workers** on Basilica GPUs, mixing compressed (Subspace Projection) and full-precision replicas.

### Code Changes

```python:137:148:basilica_training.py
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

```python:154:160:sparseloco.py
        # 1. SPoT Verification (if verifier and data provided)
        if self.verifier and verification_data:
            # Random layer verification to reduce overhead
            num_layers = len(self.params)
            # Verify ~10% of layers randomly, but at least 1
            num_to_verify = max(1, int(0.1 * num_layers))
            layer_indices = torch.randperm(num_layers)[:num_to_verify].tolist()
```

The experiment is live. You can monitor progress via the Citadel metrics:
```bash
curl <CITADEL_URL>/metrics
```