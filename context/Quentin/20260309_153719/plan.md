I've implemented a state-of-the-art decentralized training algorithm based on **SparseLoCo** (arXiv:2508.15706), specifically adapted for the **Basilica** compute environment.

### Key Innovations Implemented:
1.  **2-bit Quantized SparseLoCo**: Beyond the paper's Top-K sparsification, I've added a 2-bit quantization layer (Sign + 1-bit Magnitude) to the sparse updates. This reduces communication overhead by another ~16x compared to FP32 sparse updates, while maintaining error feedback to preserve convergence.
2.  **SPoT (Sparse Proof of Training)**: A verification mechanism that uses deterministic replay of local training steps. The aggregator can verify a worker's sparse update by re-running the computation on a data shard. This is critical for Basilica's permissionless environment where nodes are self-interested.
3.  **Asynchronous Basilica Aggregator**: Designed to handle heterogeneous nodes and churn. The simulation now handles 10% node churn and verifies updates before applying them to the global model.

### Codebase Updates:
- `sparseloco.py`: Updated with 2-bit quantization and the `SPoTVerifier` class.
- `simulate_basilica.py`: Refined to demonstrate deterministic replay and convergence under churn.

### Simulation Results:
The simulation confirms that workers can pass SPoT verification consistently when honest, and the global model continues to update even with node dropouts.

```bash
# To run the simulation yourself:
/opt/homebrew/opt/python@3.10/bin/python3.10 simulate_basilica.py
```

### Next Steps for Decentralized Pretraining:
- **Staleness Compensation**: Adjusting the outer learning rate based on how many versions behind a worker's update is.
- **1-bit Quantization**: Pushing the limits of communication efficiency for extremely bandwidth-constrained nodes.
- **Basilica API Integration**: Moving from simulation to actual remote compute orchestration.

`STATE.md` has been updated to reflect these advancements.