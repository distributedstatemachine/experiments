I've updated the decentralized training system to be more resilient and incentive-aligned for heterogeneous Basilica nodes.

### Key Innovations

1.  **Probabilistic SPoT Audits**: To deter sophisticated cheaters without the overhead of full verification, I implemented a 1% chance for a **Full Audit** on every update. This forces workers to remain honest even when they think only 10% of their layers are being checked.
2.  **Progressive Slashing**: Malicious behavior is now penalized exponentially. The `slash_penalty` increases with each consecutive failure, quickly draining the rewards of persistent bad actors.
3.  **Heterogeneity-Aware Rewards**: I introduced a `HeterogeneityFactor` (1.2x multiplier) for workers using **Subspace Projection** (arXiv:2601.02360). This incentivizes participants to use bandwidth-efficient compression, which is crucial for scaling across the open internet.
4.  **Loyalty-Based Incentives**: Valid updates build a `worker_loyalty` score, which provides a compounding bonus to rewards, encouraging long-term participation and stability in the network.

### Code Updates

```310:327:sparseloco.py
            # We also check the scale to prevent magnitude cheating
            # In AQ, scale is [mu, std]
            if expected_topk.numel() > 1:
                expected_mu = torch.mean(expected_topk)
                expected_std = torch.std(expected_topk) + 1e-8
            elif expected_topk.numel() == 1:
                expected_mu = expected_topk[0]
                expected_std = 0.0
            else:
                expected_mu = 0.0
                expected_std = 0.0
            
            mu_diff = torch.abs(expected_mu - sparse_scales[i][0]) / (torch.abs(expected_mu) + 1e-8)
            std_diff = torch.abs(expected_std - sparse_scales[i][1]) / (expected_std + 1e-8)
            
            # Verification threshold: 80% index overlap and <15% stat difference
            # We use a tighter threshold for full audits if needed, but 80% is robust for FP32 noise
            if len(intersection) / k < 0.8:
                print(f"SPoT Fail: Index overlap {len(intersection)/k:.2f} < 0.8 for layer {i}")
                return False
            
            if mu_diff > 0.15 or std_diff > 0.15:
                print(f"SPoT Fail: Stat diff (mu={mu_diff:.2f}, std={std_diff:.2f}) > 0.15 for layer {i}")
                return False
                
        return True
```

### Next Steps
- **Collusion Detection**: Investigating mechanisms to detect workers sharing the same seed to bypass SPoT.
- **ZK-SPoT**: Researching zero-knowledge proofs for training verification to protect data privacy.
- **Dynamic Aggregator Migration**: Moving the "Citadel" role to high-reputation workers to decentralize the coordination layer.

Check `context/Quentin/STATE.md` for the full updated research state.