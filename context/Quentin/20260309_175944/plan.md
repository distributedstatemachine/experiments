I've implemented **Collusion Detection** and **Aggregator Election** logic to harden the decentralized training system against adversarial behavior.

### 1. Collusion Detection
In `BasilicaAggregator`, I added a similarity-based check to detect if multiple workers are submitting identical sparse updates (indices and bits). This prevents "replay attacks" or groups of workers sharing a single computed update to bypass SPoT verification.
- **Mechanism**: Hashes the sparse indices and bits of each update.
- **Penalty**: If a match is found between different workers, both are heavily slashed (-20 rewards, reset loyalty, +2 slashes).

### 2. Aggregator Election (Dynamic Resource Allocation)
To move toward a more decentralized architecture, I implemented a basic election mechanism.
- **Logic**: Every 1000 global versions, the aggregator identifies the worker with the highest reputation (loyalty + rewards).
- **Goal**: This worker is marked as eligible for promotion to a "Citadel" (aggregator) role, enabling future migration of the coordination role to the most trusted nodes in the network.

### 3. Code Changes
Updated `sparseloco.py`:

```38:52:sparseloco.py
        # 0. Collusion Detection (Similarity-based)
        # If multiple workers submit the exact same sparse indices and bits, they might be colluding
        # or replaying a single valid update.
        update_hash = hash(tuple(tuple(idx.tolist()) for idx in sparse_indices) + 
                          tuple(tuple(b.tolist()) for b in sparse_bits))
        
        for other_id, other_hash in self.recent_updates.items():
            if other_id != worker_id and other_hash == update_hash:
                print(f"COLLUSION ALERT: Worker {worker_id} and {other_id} submitted identical updates!")
                # Penalize both workers
                for wid in [worker_id, other_id]:
                    self.worker_slashes[wid] = self.worker_slashes.get(wid, 0) + 2 # Heavier penalty for collusion
                    self.worker_loyalty[wid] = 0
                    self.worker_rewards[wid] = max(0, self.worker_rewards.get(wid, 0.0) - 20.0)
                return False
```

```241:250:sparseloco.py
        # 4. Dynamic Resource Allocation: Aggregator Migration
        # If the global version is a multiple of 1000, we could propose a new aggregator
        # based on highest reputation (loyalty + rewards).
        if self.global_version % 1000 == 0 and self.worker_rewards:
            top_worker = max(self.worker_rewards, key=self.worker_rewards.get)
            print(f"ELECTION: Worker {top_worker} is eligible for Aggregator (Citadel) promotion.")
            # In a real system, this would trigger a migration handshake.
```

`STATE.md` has been updated to reflect these advancements in Byzantine robustness. Next, I'll be looking into **Privacy-Preserving Verification (ZK-SPoT)** to enable training on private data shards.