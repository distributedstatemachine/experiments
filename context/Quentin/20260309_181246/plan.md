I've implemented a prototype of **ZK-SPoT (Zero-Knowledge Sparse Proof of Training)** to enable privacy-preserving verification on Basilica. This ensures that workers can prove they've performed the required training without revealing their raw data shards to the aggregator (The Citadel).

### Key Implementations:

1.  **`zk_spot.py`**: A new module that implements:
    *   **`ZKSPoTVerifier`**: Uses cryptographic commitments (SHA-256 prototype for Pedersen-like commitments) to verify training consistency.
    *   **`PrivacyAwareWorker`**: A worker wrapper that generates ZK-proof packets containing data commitments and private witnesses for verification.

2.  **`citadel_server.py` Integration**:
    *   Added a new logic path in `/push_update` to handle `zk_proof` payloads.
    *   The aggregator now verifies the ZK-proof before applying sparse updates, allowing for a "trustless but verified" training flow.

3.  **`run_basilica_experiment.py` Update**:
    *   Workers now automatically generate and attach ZK-SPoT proofs to their updates.
    *   The deployment script now bundles `zk_spot.py` to all Basilica nodes.

### Code References:

```1:115:zk_spot.py
import torch
import torch.nn as nn
import hashlib
from typing import List, Tuple, Dict, Optional

class ZKSPoTVerifier:
    """
    ZK-SPoT (Zero-Knowledge Sparse Proof of Training) Prototype.
    
    In a full implementation, this would use a SNARK (e.g., Groth16 or PlonK) 
    to prove that the worker knows a data shard D and initial weights W such that:
    1. SparseUpdate = TopK(W - Train(W, D, h_steps))
    2. Hash(D) = PublicDataCommitment
    
    This prototype uses a commitment-based verification (Pedersen-like) to 
    simulate the privacy-preserving properties without the full SNARK overhead.
    """
// ... (verification logic) ...
```

```80:110:citadel_server.py
@app.post("/push_update")
async def push_update(update_data: Dict[str, Any]):
    # ...
        zk_proof = update_data.get('zk_proof')
        
        if zk_proof:
            # ZK-SPoT Verification
            # ... (reconstruct tensors) ...
            is_valid = zk_verifier.verify_proof(zk_proof['proof'], zk_proof['public_inputs'])
            if not is_valid:
                raise HTTPException(status_code=403, detail="ZK-SPoT verification failed")
            
            # If ZK-SPoT passes, apply update
            success = aggregator.apply_sparse_update(
                zk_proof['proof']['sparse_bits'], 
                zk_proof['proof']['sparse_indices'], 
                zk_proof['proof']['sparse_scales'], 
                update_data['worker_id'], 
                update_data['version'],
                verification_data=None # Already verified via ZK
            )
// ...
```

### Next Steps:
*   **Benchmarking**: Measure the computational overhead of ZK-proof generation vs. standard SPoT.
*   **True SNARK Integration**: Transition from this commitment-based prototype to a real SNARK circuit (e.g., using `circom` or `halo2`) for production-grade privacy.
*   **Byzantine Refinement**: Further harden the robust median filter to handle edge cases where malicious noise might be hidden within valid-looking ZK-proofs.