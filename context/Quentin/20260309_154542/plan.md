I've analyzed the new research (arXiv:2601.02360) and integrated its insights into our Basilica-native training strategy. The core advancement is **Heterogeneous SparseLoCo**, which allows us to mix high-bandwidth nodes (The Citadel) with low-bandwidth, resource-constrained nodes (The Bourse) using selective subspace compression.

### Key Improvements:
- **Heterogeneous Compression:** We now support mixing full-precision replicas with subspace-compressed ones. This anchors the global aggregation with unbiased gradients from high-bandwidth nodes while allowing bandwidth-limited nodes to participate without stalling the system.
- **Embedding Drift Correction:** Implemented the mechanism to project token embeddings back to the subspace after global synchronization, preventing optimization divergence in mixed settings.
- **Basilica Integration:** Shifted from simulation to real orchestration using `basilica-sdk`. I've added `uv` dependencies and created a prototype for remote deployment.

### New Components:
- `basilica_training.py`: Implements `SubspaceCompressor` and the `HeterogeneousSparseLoCo` optimizer logic.
- `run_basilica_experiment.py`: Orchestrator for launching heterogeneous worker nodes on Basilica.

### Updated State:
```38:59:context/Quentin/STATE.md
## Proposed Strategy: "Heterogeneous Basilica SparseLoCo"
To make decentralized training real on Basilica, we adapt SparseLoCo for its specific constraints:
1. **Heterogeneous Compression:** Use Subspace Projection (arXiv:2601.02360) for resource-limited or bandwidth-constrained Basilica nodes (The Bourse), while keeping "The Citadel" nodes uncompressed.
2. **Asynchronous Aggregation:** Basilica nodes (miners) may have varying performance. We use an asynchronous version of SparseLoCo where the "Parameter Server" or "Aggregator" handles stale sparse updates with staleness compensation.
3. **Verification of Work (SPoT):** Since Basilica miners are self-interested, we use Sparse Proof of Training (SPoT) to verify that the sparse pseudo-gradients are actually computed from the data using deterministic replay.
4. **Dynamic Topology:** Nodes may join/leave. The algorithm handles a dynamic $R$ (number of replicas) via the asynchronous aggregator.

## Implementation Progress
- [x] Initial project structure defined.
- [x] Implement core SparseLoCo components (TopK, Error Feedback, Quantization).
- [x] Implement Asynchronous Aggregator for Basilica-style coordination.
- [x] Implement SPoT (Sparse Proof of Training) verification mechanism.
- [x] Create simulation script for benchmarking under churn and latency.
- [x] Implement 2-bit quantization for sparse updates (Sign + 1-bit Magnitude).
- [x] Implement Staleness Compensation in `BasilicaAggregator`.
- [x] Refine SPoT with deterministic replay and verify against simulation.
- [x] Analyze arXiv:2601.02360 for heterogeneous improvements.
- [x] Prototype `HeterogeneousSparseLoCo` with Subspace Compression.
- [x] Integrate `basilica-sdk` for real machine rental and orchestration.
```

I'm ready to launch the first real experiment on Basilica once you confirm the budget/compute targets.