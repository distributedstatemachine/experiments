import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional

class SparseLoCoOptimizer:
    """
    SparseLoCo (Sparse Local Communication) Optimizer for decentralized training.
    Implements Top-k sparsification with Error Feedback (EF).
    """
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        density: float = 0.03, 
        h_steps: int = 500,
        outer_lr: float = 0.7
    ):
        self.params = list(params)
        self.lr = lr
        self.density = density
        self.h_steps = h_steps
        self.outer_lr = outer_lr
        
        # Error feedback buffers (E_i in the paper)
        self.error_buffers = [torch.zeros_like(p.data) for p in self.params]
        
        # Initial model state (W_0 in the paper)
        self.initial_weights = [p.data.clone().detach() for p in self.params]
        
        self.step_count = 0

    @torch.no_grad()
    def get_sparse_update(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes the sparse pseudo-gradient update.
        Returns (values, indices, scales) for each parameter.
        Uses 2-bit quantization for values.
        """
        sparse_values = []
        sparse_indices = []
        sparse_scales = []

        for i, p in enumerate(self.params):
            # Pseudo-gradient: delta = W_initial - W_current
            delta = self.initial_weights[i] - p.data
            
            # Error Feedback: EF_i = delta + E_i
            ef_delta = delta + self.error_buffers[i]
            
            # Top-k Sparsification
            flat_delta = ef_delta.view(-1)
            k = max(1, int(self.density * flat_delta.numel()))
            
            # Get top-k largest absolute values
            abs_delta = torch.abs(flat_delta)
            values, indices = torch.topk(abs_delta, k, sorted=False)
            
            # Actual values at those indices
            topk_values = flat_delta[indices]
            
            # 2-bit Quantization (Sign + 1-bit Magnitude)
            # We quantize to {-scale, -scale/3, scale/3, scale} where scale is max(abs(topk_values))
            if topk_values.numel() > 0:
                scale = torch.max(torch.abs(topk_values))
                # Map to 4 levels: 0, 1, 2, 3
                # We store them as integers representing 2-bit values:
                # 0: -scale, 1: -scale/3, 2: scale/3, 3: scale
                
                signs = torch.sign(topk_values)
                abs_vals = torch.abs(topk_values)
                
                quantized_bits = torch.zeros_like(topk_values, dtype=torch.int8)
                mask_high = abs_vals > (scale / 2)
                
                # Logic for 2-bit mapping:
                # Positive & High -> 3 (scale)
                # Positive & Low  -> 2 (scale/3)
                # Negative & Low  -> 1 (-scale/3)
                # Negative & High -> 0 (-scale)
                
                quantized_bits[(signs > 0) & mask_high] = 3
                quantized_bits[(signs > 0) & (~mask_high)] = 2
                quantized_bits[(signs <= 0) & (~mask_high)] = 1
                quantized_bits[(signs <= 0) & mask_high] = 0
                
                # Dequantize for error feedback calculation
                dequantized = torch.zeros_like(topk_values)
                dequantized[quantized_bits == 3] = scale
                dequantized[quantized_bits == 2] = scale / 3
                dequantized[quantized_bits == 1] = -scale / 3
                dequantized[quantized_bits == 0] = -scale
                
                # Update error buffer with the quantization error
                # E_i = EF_i - Quantized(TopK(EF_i))
                new_error = ef_delta.clone()
                new_error.view(-1)[indices] -= dequantized
                self.error_buffers[i].copy_(new_error)
                
                sparse_values.append(quantized_bits)
                sparse_scales.append(scale)
            else:
                sparse_values.append(torch.tensor([], dtype=torch.int8))
                sparse_scales.append(torch.tensor(0.0))
                
            sparse_indices.append(indices)
            
        return sparse_values, sparse_indices, sparse_scales

    @torch.no_grad()
    def synchronize(self, global_weights: List[torch.Tensor]):
        """
        Synchronize local weights with the global model.
        """
        for i, p in enumerate(self.params):
            p.data.copy_(global_weights[i])
            self.initial_weights[i].copy_(global_weights[i])
            # Reset error buffer on sync to maintain determinism for verification
            # In a real system, we might want to keep it, but for SPoT it must be tracked.
            self.error_buffers[i].zero_()

class BasilicaAggregator:
    """
    Asynchronous SparseLoCo Aggregator for the Basilica "Citadel".
    Handles sparse updates from heterogeneous workers.
    """
    def __init__(self, model: nn.Module, outer_lr: float = 0.7, verifier: Optional[SPoTVerifier] = None):
        self.model = model
        self.outer_lr = outer_lr
        self.params = list(model.parameters())
        self.verifier = verifier
        
        # Version tracking for asynchronous updates
        self.global_version = 0
        self.worker_versions = {}
        
        # Incentive state: track worker reputation/rewards
        self.worker_rewards = {}
        self.worker_slashes = {}

    @torch.no_grad()
    def apply_sparse_update(
        self, 
        sparse_bits: List[torch.Tensor], 
        sparse_indices: List[torch.Tensor], 
        sparse_scales: List[torch.Tensor],
        worker_id: str,
        worker_version: int,
        verification_data: Optional[Dict] = None
    ):
        """
        Applies a sparse update to the global model with staleness compensation and SPoT verification.
        """
        # 1. SPoT Verification (if verifier and data provided)
        if self.verifier and verification_data:
            # Random layer verification to reduce overhead
            num_layers = len(self.params)
            # Verify ~20% of layers randomly, but at least 1
            num_to_verify = max(1, int(0.2 * num_layers))
            layer_indices = torch.randperm(num_layers)[:num_to_verify].tolist()
            
            is_valid = self.verifier.verify_update(
                verification_data['initial_weights'],
                sparse_bits,
                sparse_indices,
                sparse_scales,
                verification_data['data_shard'],
                verification_data['h_steps'],
                verification_data['lr'],
                verification_data['seed'],
                layer_indices=layer_indices
            )
            
            if not is_valid:
                print(f"ALERT: Worker {worker_id} failed SPoT verification! Slashing rewards.")
                self.worker_slashes[worker_id] = self.worker_slashes.get(worker_id, 0) + 1
                return False

        # 2. Staleness compensation: reduce outer_lr if worker is behind
        staleness = self.global_version - worker_version
        # Staleness compensation: η_outer / (1 + λ * staleness)
        # λ = 0.1 as a baseline, but can be tuned based on network latency
        effective_lr = self.outer_lr / (1.0 + 0.1 * max(0, staleness)) 

        for i, p in enumerate(self.params):
            bits = sparse_bits[i]
            idx = sparse_indices[i]
            scale = sparse_scales[i]
            
            if bits.numel() == 0:
                continue

            # Dequantize bits back to values
            val = torch.zeros_like(bits, dtype=torch.float32)
            val[bits == 3] = scale
            val[bits == 2] = scale / 3
            val[bits == 1] = -scale / 3
            val[bits == 0] = -scale
            
            # Apply update: W = W - effective_lr * sparse_delta
            p.data.view(-1)[idx] -= effective_lr * val
            
        self.global_version += 1
        self.worker_versions[worker_id] = self.global_version
        
        # Reward worker for valid update
        self.worker_rewards[worker_id] = self.worker_rewards.get(worker_id, 0) + 1
        return True

    def get_global_weights(self) -> List[torch.Tensor]:
        return [p.data.clone().detach() for p in self.params]

# Verification Mechanism: Sparse Proof of Training (SPoT)
class SPoTVerifier:
    """
    Verifies that a worker actually performed the training.
    """
    def __init__(self, model_fn, density: float):
        self.model_fn = model_fn
        self.density = density

    def verify_update(
        self, 
        initial_weights: List[torch.Tensor], 
        sparse_bits: List[torch.Tensor], 
        sparse_indices: List[torch.Tensor],
        sparse_scales: List[torch.Tensor],
        data_shard: Tuple[torch.Tensor, torch.Tensor],
        h_steps: int,
        lr: float,
        seed: int,
        layer_indices: Optional[List[int]] = None
    ) -> bool:
        """
        Verifies that the sparse update is consistent with a deterministic local training run.
        Supports random layer verification to reduce overhead.
        """
        # 1. Setup deterministic environment
        torch.manual_seed(seed)
        
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
        
        for i in check_indices:
            p = list(test_model.parameters())[i]
            # Note: In a real SPoT, we'd also need to track the error buffer 
            # from the previous synchronization to be 100% accurate.
            # For this simulation, we assume error buffers were reset at sync.
            delta = initial_weights[i] - p.data
            
            flat_delta = delta.view(-1)
            k = sparse_indices[i].numel()
            if k == 0: continue

            _, expected_indices = torch.topk(torch.abs(flat_delta), k, sorted=False)
            
            # Jaccard similarity of indices as a proxy for 'honesty'
            set_expected = set(expected_indices.tolist())
            set_actual = set(sparse_indices[i].tolist())
            intersection = set_expected.intersection(set_actual)
            
            # We also check the scale to prevent magnitude cheating
            expected_topk = flat_delta[expected_indices]
            expected_scale = torch.max(torch.abs(expected_topk))
            
            # Allow for small floating point drift
            scale_diff = torch.abs(expected_scale - sparse_scales[i]) / (expected_scale + 1e-8)

            # Verification threshold: 80% index overlap and <10% scale difference
            if len(intersection) / k < 0.8 or scale_diff > 0.1: 
                return False
                
        return True
