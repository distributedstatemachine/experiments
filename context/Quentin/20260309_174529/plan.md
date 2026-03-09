I've upgraded the **SparseLoCo** implementation to include **Adaptive Quantization (AQ)** and **Global Outer Momentum**, specifically optimized for the heterogeneous and bandwidth-constrained Basilica environment.

### Key Improvements

1.  **Adaptive Quantization (AQ)**:
    - Replaced fixed-level 2-bit quantization with a dynamic scheme that uses $\mu \pm \sigma$ buckets. This captures the distribution of pseudo-gradients more accurately than simple sign-magnitude scaling, reducing quantization error in the Error Feedback (EF) loop.
    - Updated `SPoTVerifier` to verify updates using statistical checks ($\mu, \sigma$ consistency) instead of just max-magnitude, making it harder for workers to cheat by inflating gradients.

2.  **Global Outer Momentum**:
    - Implemented a global momentum buffer in `BasilicaAggregator`. This accelerates convergence by smoothing out the sparse, asynchronous updates from heterogeneous workers.
    - The aggregator now applies updates as $W_{t+1} = W_t - \eta_{outer} \cdot m_t$, where $m_t$ is the momentum-tracked sparse update.

3.  **Heterogeneous Sync**:
    - Updated `HeterogeneousSparseLoCo` to support AQ, ensuring that even workers using **Subspace Projection** (arXiv:2601.02360) benefit from more accurate gradient compression.

### Updated Codebase

```python:sparseloco.py
# ... Adaptive Quantization Logic ...
mu = torch.mean(topk_values)
std = torch.std(topk_values) + 1e-8
levels = torch.tensor([mu - 1.5*std, mu - 0.5*std, mu + 0.5*std, mu + 1.5*std])
# ... Momentum Aggregation ...
m.view(-1)[idx].mul_(0.9).add_(val, alpha=0.1)
p.data.view(-1)[idx] -= effective_lr * m.view(-1)[idx]
```

### Next Steps
- Monitor the `/metrics` dashboard to compare convergence speed with the new momentum-based aggregation.
- TODO: Implement second-order momentum (Adam-style) for the outer optimizer to handle heterogeneous learning rates across Basilica nodes.
- TODO: Explore "Layer-wise Adaptive Density" to prioritize communication for layers with higher gradient variance.

`STATE.md` has been updated to reflect these advancements.