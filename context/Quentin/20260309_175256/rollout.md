I've implemented the resilient and incentive-aligned training mechanisms in `sparseloco.py` and updated the research state in `context/Quentin/STATE.md`.

### Key Implementations

1.  **Probabilistic SPoT Audits**: Added a 1% chance for a `FULL Audit` on every update, bypassing the 10% layer subsampling to deter sophisticated cheaters.
2.  **Progressive Slashing**: Implemented an exponential penalty formula: $10 \times 2^{(\text{slashes}-1)}$. This ensures that persistent bad actors are quickly drained of rewards.
3.  **Heterogeneity-Aware Rewards**: Introduced a $1.2\times$ `HeterogeneityFactor` for workers using compression (Subspace Projection), incentivizing bandwidth efficiency as per arXiv:2601.02360.
4.  **Compounding Loyalty**: Replaced the linear bonus with a compounding logarithmic bonus $0.1 \times \log_2(1 + \text{loyalty})$ to encourage long-term participation.

### Code Updates

```python:sparseloco.py
            # Check if we should perform a full verification (probabilistic audit)
            # 1% chance of full verification to deter sophisticated cheaters
            is_full_audit = torch.rand(1).item() < 0.01
            if is_full_audit:
                print(f"AUDIT: Performing FULL SPoT verification for worker {worker_id}")
                layer_indices = None 

            # ... verification logic ...
            
            if not is_valid:
                print(f"ALERT: Worker {worker_id} failed SPoT verification! Slashing rewards.")
                # Progressive Slashing: penalty increases exponentially with consecutive failures
                self.worker_slashes[worker_id] = self.worker_slashes.get(worker_id, 0) + 1
                self.worker_loyalty[worker_id] = 0 # Reset loyalty on failure
                
                # Formula: 10 * 2^(slashes - 1)
                slash_penalty = 10.0 * (2.0 ** (self.worker_slashes[worker_id] - 1))
                self.worker_rewards[worker_id] = max(0, self.worker_rewards.get(worker_id, 0.0) - slash_penalty)
                return False

        # ... aggregation logic ...
        
        # 3. Incentive Design: Loyalty Bonus & Heterogeneity Factor
        loyalty = self.worker_loyalty.get(worker_id, 0)
        # Compounding loyalty bonus: 0.1 * log2(1 + loyalty)
        loyalty_bonus = 0.1 * torch.log2(torch.tensor(loyalty + 1.0)).item()
        
        # arXiv:2601.02360: Heterogeneity-Aware Rewards
        is_compressed = verification_data.get('is_compressed', False) if verification_data else False
        hetero_factor = 1.2 if is_compressed else 1.0 # 20% bonus for being bandwidth-efficient
        
        reward = (1.0 + loyalty_bonus) * hetero_factor
```

Check `context/Quentin/STATE.md` for the updated research roadmap, including upcoming work on **Collusion Detection** and **ZK-SPoT**.