I have implemented the advanced performance and convergence optimizations for the decentralized training system on Basilica.

### Key Optimizations Implemented:

1.  **Gradient-Aware Communication Scheduling (GACS):**
    *   Workers now calculate layer-wise gradient norms during the `get_sparse_update` phase in `HeterogeneousSparseLoCo`.
    *   In the `run_basilica_experiment.py` background sync thread, GACS prioritizes layers with higher directional changes. Layers below the median gradient norm are filtered out of the communication payload, significantly reducing bandwidth usage.

2.  **Adaptive Local Steps (ALS):**
    *   The `synchronize` method in `HeterogeneousSparseLoCo` (`basilica_training.py`) now tracks `global_version` divergence.
    *   If a worker's local state diverges significantly (>50 versions), the system triggers re-anchoring logic, which dampens the error feedback buffer to prevent stale updates from becoming destructive.

3.  **Polyak-style Weight Averaging:**
    *   `BasilicaAggregator` in `sparseloco.py` now maintains `polyak_weights` using an exponential moving average ($\alpha=0.999$).
    *   The `/weights` endpoint in `citadel_server.py` now supports `?use_polyak=true`, allowing workers to pull smoothed weights for a more stable optimization trajectory.

4.  **Enhanced SPoT (Sparse Proof of Training) Verification:**
    *   Updated `SPoTVerifier` in `sparseloco.py` to include statistical checks for `layer_norms`.
    *   The aggregator now verifies that reported gradient norms match the deterministic replay, preventing "norm-inflation" attacks where workers might unfairly prioritize their updates.

5.  **Lookahead & FedSAM Refinements:**
    *   Deepened the integration of **Lookahead** in `BasilicaAggregator` to stabilize the outer trajectory.
    *   Refined **FedSAM** anchoring in the worker loop (`basilica_training.py`) to ensure local sharpness optimization remains tethered to the global weights by scaling the perturbation radius $\rho$ based on drift from the anchor.

### Codebase Changes:
- **`sparseloco.py`**: Added Polyak Averaging, Lookahead state management, and enhanced SPoT statistical checks.
- **`basilica_training.py`**: Updated `get_sparse_update` for GACS, `synchronize` for ALS re-anchoring, and `local_step` for FedSAM.
- **`citadel_server.py`**: Updated `/weights` and `/push_update` to handle Polyak weights and GACS metadata.
- **`run_basilica_experiment.py`**: Refactored the background sync thread to implement GACS and handle version-aware synchronization.

The changes are now fully integrated and ready for the next phase of **Dynamic Density Adjustment** refinement.