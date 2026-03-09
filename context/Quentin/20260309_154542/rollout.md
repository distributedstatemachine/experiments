I've implemented the **Heterogeneous SparseLoCo** strategy, integrating the subspace compression and drift correction mechanisms from arXiv:2601.02360 into our Basilica-native training stack.

### Key Changes:

1.  **Heterogeneous SparseLoCo (`basilica_training.py`):**
    *   Added `SubspaceCompressor` for orthonormal basis projection.
    *   Implemented `handle_embedding_drift` using Equation 3 & 4 from the paper. This ensures that even when mixing high-bandwidth (Citadel) and low-bandwidth (Bourse) nodes, the embeddings don't diverge by projecting them back to the shared subspace after global synchronization.
    *   Updated `synchronize` to automatically trigger drift correction.

2.  **Real Orchestration (`run_basilica_experiment.py`):**
    *   Refined the `BasilicaTrainer` to deploy actual worker nodes using the `basilica-sdk`.
    *   Workers are now configured with environment variables (`BASILICA_API_TOKEN`, `CITADEL_URL`) and a template training loop that handles sparse updates and global weight pulling.
    *   Supported heterogeneous deployment: you can specify a `compressed_ratio` to mix full-precision and subspace-compressed workers.

3.  **Dependency Management:**
    *   Verified `pyproject.toml` includes `basilica-sdk>=0.25.0` and `torch>=2.10.0`.

### Implementation Highlights:

```python:basilica_training.py
    def handle_embedding_drift(self):
        """
        Implements Equation 3 & 4 from arXiv:2601.02360
        """
        if not self.is_compressed:
            return

        for name, module in self.model.named_modules():
            if name in self.t_perp and name in self.compressors:
                U = self.compressors[name].U
                ts = module.weight.data
                t_perp = self.t_perp[name]
                
                # Project back to subspace: Π_S(TS) = TS U U^T
                proj_ts = torch.matmul(torch.matmul(ts, U), U.t())
                
                # Accumulate drift in T_perp
                t_perp.add_(ts - proj_ts)
                ts.copy_(proj_ts)
```

### Next Steps:
*   **Launch Experiment:** Run `python run_basilica_experiment.py` once you've set your `BASILICA_API_TOKEN` and configured the `CITADEL_URL`.
*   **Incentives:** Implement the slashing/reward logic in the aggregator based on SPoT (Sparse Proof of Training) results to punish dishonest miners.
*   **Subspace Tuning:** Experiment with different `k_ratio` values for the subspace projection to find the optimal bandwidth/convergence trade-off.

I've updated `context/Quentin/STATE.md` to reflect these completions. Ready for the first real run on Basilica.