I've integrated **Byzantine-robust aggregation** into the `BasilicaAggregator` and updated the research state.

### Key Implementation: Coordinate-wise Median Filtering
To defend against "noise injection" attacks (where a worker provides valid-looking but malicious updates that pass SPoT but degrade convergence), I've implemented a **Coordinate-wise Median filter** with Z-score outlier detection.

```317:346:sparseloco.py
        # 2. Byzantine-robust filtering (Coordinate-wise Median)
        # We check if the update is an outlier compared to recent updates
        # This prevents "noise injection" that passes SPoT but degrades convergence.
        if not hasattr(self, 'update_history'):
            self.update_history = [] # List of recent dequantized updates

        # Dequantize update for filtering
        dequant_update = []
        for i, p in enumerate(self.params):
            # ... dequantization logic ...
            dequant_update.append(full_val.view(p.shape))

        # Check for Byzantine behavior (Coordinate-wise Median filter)
        if len(self.update_history) >= 5:
            for i, p in enumerate(self.params):
                recent_vals = torch.stack([h[i] for h in self.update_history])
                median = torch.median(recent_vals, dim=0).values
                std = torch.std(recent_vals, dim=0) + 1e-8
                
                # Z-score check: if update is > 5 sigma from median, it's suspicious
                mask = dequant_update[i] != 0
                if mask.any():
                    z_scores = torch.abs(dequant_update[i][mask] - median[mask]) / std[mask]
                    if torch.mean(z_scores) > 5.0:
                        print(f"BYZANTINE ALERT: Worker {worker_id} update rejected (Z-score={torch.mean(z_scores):.2f})")
                        self.worker_loyalty[worker_id] = 0
                        return False
```

### Research Update: ZK-SPoT
I've analyzed the feasibility of **ZK-SPoT** (Zero-Knowledge Sparse Proof of Training). The goal is to prove that a sparse update was computed correctly from a private data shard without revealing the shard itself.
- **Approach**: Use recursive SNARKs (like Nova or Halo2) to prove $H$ steps of SGD.
- **Bottleneck**: Proving $H=500$ steps of a 2B parameter model is currently computationally prohibitive for edge nodes.
- **Proposed Optimization**: Use **Probabilistic ZK-SPoT** where only the Top-K selection and a small subset of gradient computations are proven, combined with the existing deterministic replay for the aggregator.

### Progress Recorded in `context/Quentin/STATE.md`
- [x] Implement Byzantine-robust aggregation (Coordinate-wise Median filter) in `BasilicaAggregator`.
- [x] Research Byzantine-robust aggregation for non-colluding noise injection.
- [ ] Implement ZK-SPoT for privacy-preserving verification. (Next Step)