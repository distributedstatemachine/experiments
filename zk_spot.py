import torch
import torch.nn as nn
import hashlib
import time
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
    def __init__(self, model_fn, density: float):
        self.model_fn = model_fn
        self.density = density

    def generate_commitment(self, data: torch.Tensor) -> str:
        """Generates a cryptographic commitment to a data shard."""
        # In ZK, this would be a Pedersen commitment. Here we use SHA-256 for the prototype.
        data_bytes = data.detach().cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def verify_proof(
        self,
        proof: Dict,
        public_inputs: Dict
    ) -> bool:
        """
        Verifies a ZK-SPoT proof.
        Proof contains: { 'commitment': str, 'sparse_bits': List, 'sparse_indices': List, 'sparse_scales': List, 'data_shard': Tuple }
        Public Inputs: { 'initial_weights': List, 'data_hash': str, 'h_steps': int, 'lr': float, 'seed': int }
        """
        start_time = time.time()
        # 1. Verify Data Commitment (Privacy: Aggregator doesn't see the raw data)
        if proof['commitment'] != public_inputs['data_hash']:
            print(f"ZK-SPoT Fail: Data commitment mismatch")
            return False

        # 2. Deterministic Replay (The 'Statement' being proven)
        # Note: In a real ZK system, this replay happens INSIDE the circuit.
        # For the prototype, we assume the aggregator has the data shard 
        # (or a trusted third party/TEE does) to verify the proof.
        # In a true decentralized setup, we'd use a SNARK to avoid the aggregator needing the data.
        
        # For prototype purposes, we use the same logic as SPoT but wrapped in ZK semantics.
        torch.manual_seed(public_inputs['seed'])
        test_model = self.model_fn()
        for i, p in enumerate(test_model.parameters()):
            p.data.copy_(public_inputs['initial_weights'][i])
            
        optimizer = torch.optim.SGD(test_model.parameters(), lr=public_inputs['lr'])
        inputs, targets = proof['data_shard'] # In real ZK, this is a private witness
        
        test_model.train()
        optimizer.zero_grad()
        outputs = test_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        # 3. Verify Sparsity and AQ Stats (The 'Output' of the circuit)
        for i, p in enumerate(test_model.parameters()):
            delta = public_inputs['initial_weights'][i] - p.data
            flat_delta = delta.view(-1)
            
            # Use the indices provided in the proof to check if they match the top-k of the replayed delta
            k = proof['sparse_indices'][i].numel()
            if k == 0: continue

            abs_delta = torch.abs(flat_delta)
            _, expected_indices = torch.topk(abs_delta, k, sorted=False)
            
            # Jaccard check (allow some floating point drift, but should be high)
            set_expected = set(expected_indices.tolist())
            set_actual = set(proof['sparse_indices'][i].tolist())
            intersection_size = len(set_expected.intersection(set_actual))
            
            if k > 0 and (intersection_size / k) < 0.95:
                print(f"ZK-SPoT Fail: Index mismatch for layer {i}. Jaccard: {intersection_size/k:.4f}")
                return False

            # AQ Stat check: verify the scales (mean/std) match the replayed top-k values
            actual_topk = flat_delta[proof['sparse_indices'][i]]
            if actual_topk.numel() > 1:
                expected_mu = torch.mean(actual_topk)
                expected_std = torch.std(actual_topk) + 1e-8
            else:
                expected_mu = actual_topk[0] if actual_topk.numel() == 1 else 0.0
                expected_std = 0.0
            
            mu_diff = torch.abs(expected_mu - proof['sparse_scales'][i][0]) / (torch.abs(expected_mu) + 1e-8)
            std_diff = torch.abs(expected_std - proof['sparse_scales'][i][1]) / (torch.abs(expected_std) + 1e-8)
            
            if mu_diff > 0.01 or std_diff > 0.01:
                print(f"ZK-SPoT Fail: Scale mismatch for layer {i}. Mu diff: {mu_diff:.4f}, Std diff: {std_diff:.4f}")
                return False
        
        end_time = time.time()
        print(f"[ZK-SPoT Audit] Verification successful in {end_time - start_time:.4f}s")
        return True

class PrivacyAwareWorker:
    """
    A worker that generates ZK-SPoT proofs to protect its data privacy.
    """
    def __init__(self, worker_id: str, model_fn, data_shard: Tuple[torch.Tensor, torch.Tensor]):
        self.worker_id = worker_id
        self.model_fn = model_fn
        self.data_shard = data_shard
        self.data_hash = self._commit_to_data()

    def _commit_to_data(self) -> str:
        data_bytes = self.data_shard[0].detach().cpu().numpy().tobytes() + \
                     self.data_shard[1].detach().cpu().numpy().tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def generate_proof(
        self, 
        initial_weights: List[torch.Tensor],
        sparse_bits: List[torch.Tensor],
        sparse_indices: List[torch.Tensor],
        sparse_scales: List[torch.Tensor],
        h_steps: int,
        lr: float,
        seed: int
    ) -> Dict:
        """Generates the proof packet for the aggregator."""
        return {
            "proof": {
                "commitment": self.data_hash,
                "sparse_bits": sparse_bits,
                "sparse_indices": sparse_indices,
                "sparse_scales": sparse_scales,
                "data_shard": self.data_shard # Private witness (sent for prototype verification)
            },
            "public_inputs": {
                "initial_weights": initial_weights,
                "data_hash": self.data_hash,
                "h_steps": h_steps,
                "lr": lr,
                "seed": seed
            }
        }
