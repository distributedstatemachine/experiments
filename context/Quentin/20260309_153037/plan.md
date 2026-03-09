I've analyzed the SparseLoCo paper and Basilica's infrastructure. Here's the state of the art for decentralized training on Basilica:

### 1. Algorithm: SparseLoCo (Sparse Local Communication)
SparseLoCo is the foundation for training across internet-native, bandwidth-constrained environments. It achieves **1-3% communication density** while outperforming full-precision DiLoCo.

**Key Mechanism:**
- **Infrequent Communication ($H$ steps):** Each node (Basilica worker) runs $H$ local AdamW steps (e.g., $H=500$).
- **TOP-k Sparsification:** Instead of sending the full model, only the $k$ most significant pseudo-gradient components are sent.
- **Error Feedback (EF):** Local accumulators track the unsent components, ensuring no information is lost over time. This EF naturally approximates the outer momentum needed for convergence.
- **2-bit Quantization:** The sparse values are further compressed to 2 bits without performance loss.

### 2. Basilica Integration Strategy
Basilica's "Bourse" (Bittensor-backed marketplace) is the ideal environment for this.

**Proposed Architecture:**
- **Coordinator (The Citadel):** A reliable node (e.g., a Basilica Managed Deployment) acts as the aggregator. It doesn't need high compute, just high availability.
- **Workers (The Bourse):** Independent miners rent H100/A100 nodes. They pull the current model, train for $H$ steps, and push sparse updates.
- **Asynchronous Sparse Aggregation:** To handle node churn and heterogeneous hardware, we move from synchronous to **Asynchronous SparseLoCo**. The coordinator applies sparse updates as they arrive, using a version-tracking mechanism to handle staleness.

### 3. Verification & Anti-Cheating
In a permissionless environment like Basilica, we cannot trust workers.

**Mechanism: Sparse Proof of Training (SPoT)**
- **Index Verification:** The coordinator randomly selects a subset of the TOP-k indices reported by a worker and asks for the local gradients at those specific indices for a specific batch.
- **Deterministic Replay:** Since the worker's seed and data shard are known, the coordinator (or a validator) can re-run a tiny fraction of the computation to verify the reported pseudo-gradient.
- **Incentive Alignment:** Workers are slashed if their sparse updates deviate significantly from the verified subset.

### 4. Next Steps for Quentin
I've updated `context/Quentin/STATE.md` with these findings. The next phase is to implement a simulation of this "Basilica-Native SparseLoCo" to benchmark convergence under 10% node churn and 100ms+ latency.

```python:implementation_sketch.py
# Conceptual SparseLoCo Aggregator for Basilica
class BasilicaAggregator:
    def __init__(self, model, density=0.03):
        self.global_model = model
        self.density = density
        self.error_buffer = torch.zeros_like(model.parameters())

    def apply_update(self, sparse_delta, indices):
        # Asynchronous update: apply sparse delta to global model
        with torch.no_grad():
            for p, d, idx in zip(self.global_model.parameters(), sparse_delta, indices):
                p.view(-1)[idx] -= d 
```

I'm ready to move into the implementation phase. Should I prioritize the verification mechanism or the asynchronous aggregation logic?