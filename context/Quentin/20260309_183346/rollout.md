I've implemented the optimized decentralized training plan, focusing on convergence stability and communication efficiency for heterogeneous environments.

### Key Optimizations Implemented

1.  **Gradient-Informed Sparsity (GIS)**:
    - Updated `HeterogeneousSparseLoCo` to use a magnitude-weighted selection criteria.
    - Instead of simple Top-K on pseudo-gradients, it now prioritizes updates that represent significant directional changes by weighting `delta` with the raw gradient magnitude.
    
2.  **Federated SAM (FedSAM)**:
    - Enhanced the local training step with an anchored version of Sharpness-Aware Minimization.
    - It scales the perturbation radius ($\rho$) based on the local model's drift from the global anchor, ensuring that local sharpness optimization doesn't diverge from the global consensus in heterogeneous environments.

3.  **Communication Overlapping (Background Sync)**:
    - Refactored the training loop in `run_basilica_experiment.py` to use a background thread for pushing updates and pulling weights.
    - This allows workers to continue local computation (training) while waiting for network I/O, effectively hiding communication latency.

4.  **ZK-SPoT Benchmarking**:
    - Added high-resolution timing to the `ZKSPoTVerifier` to track verification overhead.
    - This provides the metrics needed to balance privacy-preserving verification with aggregator throughput.

### Implementation Details

```python:basilica_training.py
# GIS Implementation in get_sparse_update
g_max = torch.max(torch.abs(g_flat)) + 1e-8
gis_score = torch.abs(flat) * (1.0 + torch.abs(g_flat) / g_max)
_, indices = torch.topk(gis_score, k, sorted=False)
```

```python:run_basilica_experiment.py
# Background Sync Thread
def background_sync():
    while True:
        update_task = sync_queue.get()
        # ... push_update and pull_weights ...
        self.pending_weights = (global_weights, sync_time)
```

I have pushed these changes to the `experiments` remote on the `main` branch.

```bash
git push experiments main:main
```

Progress has been recorded in `context/Quentin/STATE.md`.