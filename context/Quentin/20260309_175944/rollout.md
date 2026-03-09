I've implemented the **Collusion Detection** and **Aggregator Election** logic in `sparseloco.py` and updated the research state in `STATE.md`.

### 1. Collusion Detection
I've hardened the `BasilicaAggregator` against replay attacks and worker collusion by hashing sparse updates (indices and bits). If two different workers submit identical updates, they are flagged and heavily penalized.

```154:170:sparseloco.py
        # 0. Collusion Detection (Similarity-based)
        # If multiple workers submit the exact same sparse indices and bits, they might be colluding
        # or replaying a single valid update.
        # We use a hash of the indices and bits to detect exact duplicates.
        update_data = []
        for idx, bits in zip(sparse_indices, sparse_bits):
            update_data.append(tuple(idx.tolist()))
            update_data.append(tuple(bits.tolist()))
        update_hash = hash(tuple(update_data))
        
        for other_id, other_hash in self.recent_updates.items():
            if other_id != worker_id and other_hash == update_hash:
                print(f"COLLUSION ALERT: Worker {worker_id} and {other_id} submitted identical updates!")
                # Penalize both workers heavily for collusion
                for wid in [worker_id, other_id]:
                    self.worker_slashes[wid] = self.worker_slashes.get(wid, 0) + 2
                    self.worker_loyalty[wid] = 0
                    self.worker_rewards[wid] = max(0, self.worker_rewards.get(wid, 0.0) - 20.0)
                return False
        
        self.recent_updates[worker_id] = update_hash
```

### 2. Aggregator Election
To facilitate future decentralization, I've added an election mechanism that identifies high-reputation workers (based on loyalty and rewards) every 1000 global versions for potential promotion to a "Citadel" role.

```269:275:sparseloco.py
        # 4. Dynamic Resource Allocation: Aggregator Migration
        # If the global version is a multiple of 1000, we could propose a new aggregator
        # based on highest reputation (loyalty + rewards).
        if self.global_version % 1000 == 0 and self.worker_rewards:
            top_worker = max(self.worker_rewards, key=self.worker_rewards.get)
            print(f"ELECTION: Worker {top_worker} is eligible for Aggregator (Citadel) promotion.")
            # In a real system, this would trigger a migration handshake.
```

### 3. State Update
`context/Quentin/STATE.md` now reflects these changes as completed milestones.

Next, I'll be researching **ZK-SPoT** to enable training verification without exposing private data shards.