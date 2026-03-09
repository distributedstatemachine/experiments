# Quentin - Decentralized Training State

## Research & Analysis
- **Paper Analysis (SparseLoCo - arXiv:2508.15706):**
    - Key Innovation: Combines **DiLoCo** (infrequent communication) with **TOP-k sparsification** and **Error Feedback (EF)**.
    - Observation: Outer momentum can be approximated locally by an EF accumulator.
    - Performance: Reaches 1-3% sparsity while outperforming full-precision DiLoCo.
    - Mechanism: Replaces global outer momentum with a single local EF accumulator.
    - Scalability: Shown to scale to 2B parameters and 32+ workers.
- **Paper Analysis (Heterogeneous Low-Bandwidth Pre-Training - arXiv:2601.02360):**
    - Key Innovation: Combines **SparseLoCo** with **Subspace Projection** for activation/gradient compression.
    - Observation: Heterogeneous configurations (mixing compressed and uncompressed replicas) consistently outperform uniform compression.
    - Mechanism: Selective compression based on interconnect bandwidth; handles embedding drift via projection back to subspace.
    - Performance: Activation compression composes with SparseLoCo at modest cost; bias reduction from uncompressed replicas anchors aggregation.
- **Basilica Analysis:**
    - Provides on-demand compute (GPUs/CPUs) via Bittensor (The Bourse) or Datacenters (The Citadel).
    - Supports containerized deployments and raw SSH rentals.
    - Pay-per-minute via TAO credits.
    - Ideal for heterogeneous, permissionless compute environments.

## Live Infrastructure (Basilica)
- **Citadel Aggregator**: `https://78b58e07-e8ee-4a28-9fac-3fa3d9360c30.deployments.basilica.ai`
- **Dashboard**: `https://78b58e07-e8ee-4a28-9fac-3fa3d9360c30.deployments.basilica.ai/metrics`
- **Status**: Active (12 Heterogeneous Workers)

## Proposed Strategy: "Heterogeneous Basilica SparseLoCo"
To make decentralized training real on Basilica, we adapt SparseLoCo for its specific constraints:
1. **Heterogeneous Compression:** Use Subspace Projection (arXiv:2601.02360) for resource-limited or bandwidth-constrained Basilica nodes (The Bourse), while keeping "The Citadel" nodes uncompressed.
2. **Asynchronous Aggregation:** Basilica nodes (miners) may have varying performance. We use an asynchronous version of SparseLoCo where the "Parameter Server" or "Aggregator" handles stale sparse updates with staleness compensation.
3. **Verification of Work (SPoT):** Since Basilica miners are self-interested, we use Sparse Proof of Training (SPoT) to verify that the sparse pseudo-gradients are actually computed from the data using deterministic replay.
4. **Dynamic Topology & Density:** Nodes may join/leave. The algorithm handles a dynamic $R$ (number of replicas) via the asynchronous aggregator. We also implement **Dynamic Density Adjustment** to scale communication overhead based on real-time network latency.

## Implementation Progress
- [x] Initial project structure defined with `quentin/` and `quant/` isolation.
- [x] Refactored codebase to explicitly separate `Quentin` (Research) and `Quant` (Trading) tasks.
- [x] Updated all imports in `quentin/` and `quant/` to use the new namespaced structure.
- [x] Refined `quentin/deploy_citadel.py` and `quentin/run_basilica_experiment.py` to correctly bundle files from their new locations.
- [x] Fixed `quant/quant_worker.py` and `quant/quant_aggregator.py` to point to the correct research modules in `quentin/`.
- [x] Implemented explicit task isolation in `BasilicaAggregator` and `HeterogeneousSparseLoCo` to separate research and trading workloads.
- [x] Optimized convergence with Nesterov Accelerated Gradient (NAG) and Lookahead Aggregation.
- [x] Implement core SparseLoCo components in `quentin/sparseloco.py`.
- [x] Implement Asynchronous Aggregator in `quentin/citadel_server.py`.
- [x] Implement SPoT verification in `quentin/zk_spot.py`.
- [x] Create simulation script `quentin/simulate_basilica.py`.
- [x] Implement 2-bit quantization and Staleness Compensation.
- [x] Refine SPoT with deterministic replay.
- [x] Prototype `HeterogeneousSparseLoCo` with Subspace Compression.
- [x] Integrate `basilica-sdk` in `quentin/basilica_training.py`.
- [x] Implement Embedding Drift Correction.
- [x] Refine `quentin/run_basilica_experiment.py` for real worker deployment.
- [x] Implement Complex SPoT with random layer verification.
- [x] Finalize `quentin/citadel_server.py` for decentralized coordination.
- [x] Implement robust retry logic and error handling.
- [x] Verify environment setup and dependency management with `uv`.
- [x] Launch Citadel aggregator server on Basilica.
- [x] Conduct first real experiment on Basilica.
- [x] Automated Citadel deployment with `quentin/deploy_citadel.py`.
- [x] Enabled heterogeneous worker orchestration on Basilica.
- [x] Implement real-time metrics dashboard (/metrics).
- [x] Scale experiment to 12+ heterogeneous workers.
- [x] Implement adaptive SPoT verification.
- [x] Implement loyalty-based reward multipliers and Progressive Slashing.
- [x] Implement Adaptive Quantization (AQ) and Global Outer Momentum.
- [x] Implement Collusion Detection and Dynamic Aggregator Election.
- [x] Implement Byzantine-robust aggregation (Coordinate-wise Median).
- [x] Implement Nesterov Accelerated Gradient (NAG) and LAMB.
- [x] Implement Lookahead Optimizer and Polyak-style weight averaging.
- [x] Implement Gradient-Aware Communication Scheduling (GACS) and ALS.
- [x] Implement Sharpness-Aware Minimization (SAM) and FedSAM.
- [x] Optimize communication overlapping with background thread synchronization.
- [x] Implement Gradient-Informed Sparsity (GIS).
- [x] Benchmark ZK-SPoT vs SPoT overhead.

## Next Research Directions: "Advanced Convergence & Performance Optimization"
1. **Lookahead Aggregation:** Integrated Lookahead optimizer into `BasilicaAggregator` to stabilize the global trajectory by maintaining "slow weights" that interpolate with the fast, asynchronously updated weights.
2. **Sharpness-Aware Minimization (SAM) & FedSAM:** Workers now use SAM and FedSAM for local updates to find flatter minima and anchor updates to the global consensus, improving generalization in heterogeneous settings.
3. **Communication Overlapping:** Refactored the worker training loop in `run_basilica_experiment.py` using background threads to fully overlap local computation and global communication.
4. **Gradient-Informed Sparsity (GIS):** Implemented a magnitude-weighted selection criteria for SparseLoCo updates, ensuring that the most directionally significant changes are communicated first.
5. **Nesterov Accelerated Gradient (NAG):** Replaced standard outer momentum with NAG in `BasilicaAggregator` to reduce oscillation and accelerate convergence.
6. **LAMB (Layer-wise Adaptive Moments):** Integrated LAMB-style adaptive rate scaling in `HeterogeneousSparseLoCo` to handle gradient scale variations across layers in heterogeneous networks.
