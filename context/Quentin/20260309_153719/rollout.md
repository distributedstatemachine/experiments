I've implemented the **Basilica-Native SparseLoCo** plan, introducing 2-bit quantization, asynchronous aggregation with staleness compensation, and the SPoT (Sparse Proof of Training) verification mechanism.

### Key Implementation Details:

1.  **2-bit Quantized SparseLoCo**:
    - Updated `SparseLoCoOptimizer` to compress sparse updates into 2 bits (Sign + 1-bit Magnitude).
    - Values are mapped to `{-scale, -scale/3, scale/3, scale}`, where `scale` is the maximum absolute value in the Top-K set.
    - Error feedback (EF) now accounts for both sparsification and quantization error, ensuring convergence.

2.  **Asynchronous Basilica Aggregator**:
    - `BasilicaAggregator` now supports asynchronous updates from workers.
    - **Staleness Compensation**: Implemented a decay factor for the outer learning rate based on how many versions a worker is behind the global model.

3.  **SPoT (Sparse Proof of Training)**:
    - `SPoTVerifier` performs deterministic replay of local training steps to verify worker honesty.
    - It checks for index overlap (Jaccard similarity > 80%) and scale consistency to prevent magnitude cheating.

### Codebase Updates:

```python:sparseloco.py
# ... 2-bit quantization logic ...
quantized_bits[(signs > 0) & mask_high] = 3
quantized_bits[(signs > 0) & (~mask_high)] = 2
quantized_bits[(signs <= 0) & (~mask_high)] = 1
quantized_bits[(signs <= 0) & mask_high] = 0

# ... Staleness compensation ...
staleness = self.global_version - worker_version
effective_lr = self.outer_lr / (1.0 + 0.1 * staleness)
```

### Simulation Results:
Running the simulation confirms that the system handles node churn (10%) and asynchronous updates while maintaining SPoT verification integrity.

```bash
# Run the simulation
/opt/homebrew/opt/python@3.10/bin/python3.10 simulate_basilica.py
```

`context/Quentin/STATE.md` has been updated to reflect these advancements. Next steps include exploring 1-bit quantization and integrating with the actual Basilica API for remote orchestration.