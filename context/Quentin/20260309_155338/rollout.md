I've implemented the plan to move toward a live decentralized training deployment on Basilica.

### Key Changes

1.  **Complex SPoT (Sparse Proof of Training)**: Updated `BasilicaAggregator` and `SPoTVerifier` in `sparseloco.py` to support **random layer verification**. This reduces the aggregator's computational overhead by verifying only a subset (~20%) of layers per update, while maintaining a strong probabilistic deterrent against cheating.
2.  **SDK Integration & Orchestration**: Refined `run_basilica_experiment.py` to use the `basilica-python` SDK for deploying heterogeneous workers. The worker code now includes logic for `push_update` and `pull_weights` via the Citadel server.
3.  **Citadel Server**: Finalized `citadel_server.py` as the central coordination point. It handles asynchronous sparse updates, staleness compensation, and serves global weights to workers.

### Implementation Details

**Random Layer Verification in `sparseloco.py`**:
```python:227:246:sparseloco.py
        # 2. Clone model and run local training
        test_model = self.model_fn()
        for i, p in enumerate(test_model.parameters()):
            p.data.copy_(initial_weights[i])
            
        # Re-run the exact same training steps
        optimizer = torch.optim.SGD(test_model.parameters(), lr=lr)
        
        inputs, targets = data_shard
        for _ in range(h_steps):
            optimizer.zero_grad()
            outputs = test_model(inputs)
            loss = torch.nn.functional.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # 3. Compute what the sparse update SHOULD have been
        # If layer_indices is provided, only verify those layers
        check_indices = layer_indices if layer_indices is not None else range(len(list(test_model.parameters())))
```

**Worker Synchronization Logic in `run_basilica_experiment.py`**:
```python:90:108:run_basilica_experiment.py
    def push_update(self, update):
        # Convert tensors to lists for JSON
        serializable_update = {
            "worker_id": WORKER_ID,
            "version": self.version,
            "bits": [u['bits'].tolist() if u else [] for u in update['updates']],
            "indices": [u['indices'].tolist() if u else [] for u in update['updates']],
            "scales": [u['scale'].item() if u else 0.0 for u in update['updates']]
        }
        resp = requests.post(f"{CITADEL_URL}/push_update", json=serializable_update)
        return resp.json()

    def pull_weights(self):
        resp = requests.get(f"{CITADEL_URL}/weights")
        data = resp.json()
        weights = [torch.tensor(w) for w in data['weights']]
        return weights
```

I've updated `context/Quentin/STATE.md`. We are now ready for the **Live Experiment** on real Basilica hardware. Provide the signal to proceed with the first decentralized training run.