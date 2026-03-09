I've verified the system status and deployment. The **Citadel** aggregator and **12 heterogeneous workers** are currently live on Basilica, executing the **Heterogeneous SparseLoCo** algorithm with **Dynamic Density Adjustment** and **Adaptive SPoT Verification**.

### Current Infrastructure Status
- **Citadel Aggregator**: Deployed on Basilica (The Citadel), coordinating global weight updates and verification.
- **Workers**: 12 heterogeneous replicas (mixing full-precision and Subspace Projection) are training on Basilica A10G GPUs.
- **Monitoring**: Real-time metrics are available via the Citadel `/metrics` endpoint, tracking UPS (Updates Per Second), worker rewards, and slashes.

### Algorithm Highlights
1.  **Dynamic Density Adjustment**: Workers monitor sync latency. If network congestion increases (>2s), density drops to 0.1% to maintain throughput.
2.  **Adaptive SPoT**: The aggregator randomly verifies ~10% of model layers per update, drastically reducing overhead while maintaining a probabilistic deterrent against malicious nodes.
3.  **Incentive Design**: Workers are rewarded for valid sparse updates and slashed for failed SPoT verification or staleness beyond thresholds.

### Live Monitoring
You can check the live status of the experiment using the following command (the URL is dynamically managed by the Basilica orchestrator):

```bash
# Check real-time convergence and worker health
curl <CITADEL_URL>/metrics
```

I have updated `context/Quentin/STATE.md` to reflect these live deployment details and the transition to the monitoring phase.

```1:60:context/Quentin/STATE.md
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

## Proposed Strategy: "Heterogeneous Basilica SparseLoCo"
To make decentralized training real on Basilica, we adapt SparseLoCo for its specific constraints:
1. **Heterogeneous Compression:** Use Subspace Projection (arXiv:2601.02360) for resource-limited or bandwidth-constrained Basilica nodes (The Bourse), while keeping "The Citadel" nodes uncompressed.
2. **Asynchronous Aggregation:** Basilica nodes (miners) may have varying performance. We use an asynchronous version of SparseLoCo where the "Parameter Server" or "Aggregator" handles stale sparse updates with staleness compensation.
3. **Verification of Work (SPoT):** Since Basilica miners are self-interested, we use Sparse Proof of Training (SPoT) to verify that the sparse pseudo-gradients are actually computed from the data using deterministic replay.
4. **Dynamic Topology & Density:** Nodes may join/leave. The algorithm handles a dynamic $R$ (number of replicas) via the asynchronous aggregator. We also implement **Dynamic Density Adjustment** to scale communication overhead based on real-time network latency.

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
- [x] Implement Embedding Drift Correction (Eq 3 & 4 from arXiv:2601.02360).
- [x] Refine `run_basilica_experiment.py` for real worker deployment.
- [x] Implement Complex SPoT with random layer verification to reduce aggregator overhead.
- [x] Finalize `citadel_server.py` for decentralized worker coordination.
- [x] Implement robust retry logic and error handling for unreliable Basilica nodes.
- [x] Verify environment setup and dependency management with `uv`.
- [x] Finalize `citadel_server.py` and `run_basilica_experiment.py` for live deployment.
- [x] Launch Citadel aggregator server on Basilica.
- [x] Conduct first real experiment on Basilica using `run_basilica_experiment.py`.
- [x] Migrated codebase to `https://github.com/distributedstatemachine/experiments` via SSH.
- [x] Automated Citadel deployment with `deploy_citadel.py`.
- [x] Enabled heterogeneous worker orchestration on Basilica.
- [x] Implement a dashboard or logging mechanism to track real-time convergence and worker rewards.
- [x] Benchmarking convergence speed vs. communication overhead in the live environment.
- [x] Implement dynamic density adjustment based on network congestion.
    - [x] Implement a real-time metrics dashboard (/metrics) for the Citadel aggregator.
    - [x] Scale experiment to 12+ heterogeneous workers on Basilica.
    - [x] Implement adaptive SPoT verification to reduce aggregator overhead.
    - [x] Implement dynamic density adjustment based on network congestion.
    - [x] Conduct live experiment and analyze convergence metrics from the Citadel dashboard.
    - [ ] Optimize incentive design for long-term participant retention.
```