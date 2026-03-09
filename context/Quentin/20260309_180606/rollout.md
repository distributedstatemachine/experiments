I've implemented the **Byzantine-robust aggregation** in `BasilicaAggregator` and updated the research state.

### Implementation: Robust Coordinate-wise Median Filtering
I've upgraded the proposed filter to use **Median Absolute Deviation (MAD)** instead of standard deviation. MAD is significantly more robust to the very outliers we are trying to detect, as a single malicious update can heavily skew the standard deviation but has minimal impact on the median and MAD.

```python:317:346:sparseloco.py
        # Check for Byzantine behavior (Coordinate-wise Median filter)
        # If we have enough history, check if this update is too far from the median
        if len(self.update_history) >= 5:
            # Stack recent updates for each parameter
            for i, p in enumerate(self.params):
                recent_vals = torch.stack([h[i] for h in self.update_history])
                # Compute median and MAD (Median Absolute Deviation) for robustness
                # Standard deviation is sensitive to the very outliers we're trying to detect
                median = torch.median(recent_vals, dim=0).values
                mad = torch.median(torch.abs(recent_vals - median), dim=0).values + 1e-8
                
                # Z-score check (using MAD): if update is > 5 sigma from median, it's suspicious
                # We only check non-zero entries in the sparse update
                mask = dequant_update[i] != 0
                if mask.any():
                    # Robust Z-score = 0.6745 * (x - median) / MAD
                    z_scores = 0.6745 * torch.abs(dequant_update[i][mask] - median[mask]) / mad[mask]
                    if torch.mean(z_scores) > 5.0:
                        print(f"BYZANTINE ALERT: Worker {worker_id} update rejected (Robust Z-score={torch.mean(z_scores):.2f})")
                        self.worker_loyalty[worker_id] = 0
                        # Progressive Slashing for Byzantine behavior
                        self.worker_slashes[worker_id] = self.worker_slashes.get(worker_id, 0) + 1
                        slash_penalty = 20.0 * (2.0 ** (self.worker_slashes[worker_id] - 1))
                        self.worker_rewards[worker_id] = max(0, self.worker_rewards.get(worker_id, 0.0) - slash_penalty)
                        return False
```

### Key Enhancements:
1.  **Robust Statistics**: Replaced `std` with `MAD` for outlier detection.
2.  **Robust Z-Score**: Implemented the standard robust Z-score formula ($0.6745 \times \frac{x - \text{median}}{\text{MAD}}$) to align with Gaussian assumptions while remaining outlier-resistant.
3.  **Byzantine Slashing**: Integrated the rejection with the existing **Progressive Slashing** mechanism. Byzantine behavior now triggers a higher base penalty (20.0) compared to simple SPoT failures.
4.  **Loyalty Reset**: Malicious behavior immediately resets the worker's loyalty multiplier.

### Progress Recorded in `context/Quentin/STATE.md`
- [x] Implement Byzantine-robust aggregation (Robust Coordinate-wise Median filter with MAD).
- [x] Research Byzantine-robust aggregation for non-colluding noise injection.
- [ ] Implement ZK-SPoT for privacy-preserving verification. (Next Step)

The system is now hardened against both deterministic cheating (via SPoT) and statistical noise injection (via Robust Median Filtering). Next, I'll focus on the ZK-SPoT research to enable privacy-preserving verification.