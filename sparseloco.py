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
        outer_lr: float = 0.7,
        beta: float = 0.9, # Momentum for outer update
        use_nag: bool = True # Use Nesterov Accelerated Gradient
    ):
        self.params = list(params)
        self.lr = lr
        self.density = density
        self.h_steps = h_steps
        self.outer_lr = outer_lr
        self.beta = beta
        self.use_nag = use_nag
        
        # Error feedback buffers (E_i in the paper)
        self.error_buffers = [torch.zeros_like(p.data) for p in self.params]
        
        # Outer momentum buffers (M_i)
        self.momentum_buffers = [torch.zeros_like(p.data) for p in self.params]
        
        # Initial model state (W_0 in the paper)
        self.initial_weights = [p.data.clone().detach() for p in self.params]
        
        self.step_count = 0

    @torch.no_grad()
    def get_sparse_update(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Computes the sparse pseudo-gradient update with Adaptive Quantization (AQ).
        Returns (values, indices, scales) for each parameter.
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
            
            # Adaptive 2-bit Quantization (AQ)
            # Instead of fixed levels, we use μ ± σ to define buckets
            if topk_values.numel() > 1:
                mu = torch.mean(topk_values)
                std = torch.std(topk_values) + 1e-8
                
                # Levels: mu - 1.5*std, mu - 0.5*std, mu + 0.5*std, mu + 1.5*std
                levels = torch.tensor([mu - 1.5*std, mu - 0.5*std, mu + 0.5*std, mu + 1.5*std], device=topk_values.device)
                
                # Quantize by finding nearest level
                diffs = torch.abs(topk_values.unsqueeze(-1) - levels)
                quantized_bits = torch.argmin(diffs, dim=-1).to(torch.int8)
                
                # Dequantize for error feedback calculation
                dequantized = levels[quantized_bits.long()]
                
                # Update error buffer
                new_error = ef_delta.clone()
                new_error.view(-1)[indices] -= dequantized
                self.error_buffers[i].copy_(new_error)
                
                sparse_values.append(quantized_bits)
                # Store mu and std as the "scale" for dequantization
                sparse_scales.append(torch.tensor([mu, std], device=topk_values.device))
            elif topk_values.numel() == 1:
                # Fallback for single value
                val = topk_values[0]
                sparse_values.append(torch.tensor([2], dtype=torch.int8)) # Map to mu+0.5*std roughly
                sparse_scales.append(torch.tensor([val, 0.0], device=topk_values.device))
                self.error_buffers[i].view(-1)[indices] = 0
            else:
                sparse_values.append(torch.tensor([], dtype=torch.int8))
                sparse_scales.append(torch.tensor([0.0, 0.0]))
                
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
    def __init__(self, model: nn.Module, outer_lr: float = 0.7, beta: float = 0.9, use_nag: bool = True, verifier: Optional[SPoTVerifier] = None, use_lookahead: bool = True, la_steps: int = 5, la_alpha: float = 0.5):
        self.model = model
        self.outer_lr = outer_lr
        self.beta = beta
        self.use_nag = use_nag
        self.params = list(model.parameters())
        self.verifier = verifier
        
        # Lookahead state
        self.use_lookahead = use_lookahead
        self.la_steps = la_steps
        self.la_alpha = la_alpha
        self.slow_weights = [p.data.clone().detach() for p in self.params]
        self.la_counter = 0
        
        # Version tracking for asynchronous updates
        self.global_version = 0
        self.worker_versions = {}
        
        # Outer momentum buffers (M_i)
        self.momentum_buffers = [torch.zeros_like(p.data) for p in self.params]
        
        # Incentive state: track worker reputation/rewards
        self.worker_rewards = {}
        self.worker_slashes = {}
        
        # Multiplier for long-term retention (loyalty bonus)
        self.worker_loyalty = {} # Tracks consecutive valid updates
        
        # Collusion detection: track recent updates to find identical submissions
        self.recent_updates = {} # worker_id -> hash of update

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
        Applies a sparse update to the global model with staleness compensation, 
        Adaptive Dequantization, and Byzantine-robust Aggregation (Krum/Median).
        """
        # 0. Collusion Detection (Similarity-based)
        # ... existing collusion detection code ...
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
        # Limit history to prevent memory leak
        if len(self.recent_updates) > 100:
            self.recent_updates.pop(next(iter(self.recent_updates)))

        # 1. SPoT Verification (if verifier and data provided)
        # ... existing SPoT verification code ...
        if self.verifier and verification_data:
            # Random layer verification to reduce overhead
            num_layers = len(self.params)
            # Verify ~10% of layers randomly, but at least 1
            num_to_verify = max(1, int(0.1 * num_layers))
            layer_indices = torch.randperm(num_layers)[:num_to_verify].tolist()
            
            # Check if we should perform a full verification (probabilistic audit)
            # 1% chance of full verification to deter sophisticated cheaters
            is_full_audit = torch.rand(1).item() < 0.01
            if is_full_audit:
                print(f"AUDIT: Performing FULL SPoT verification for worker {worker_id}")
                layer_indices = None 

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
                # Progressive Slashing: penalty increases exponentially with consecutive failures
                self.worker_slashes[worker_id] = self.worker_slashes.get(worker_id, 0) + 1
                self.worker_loyalty[worker_id] = 0 # Reset loyalty on failure
                
                # Formula: 10 * 2^(slashes - 1)
                slash_penalty = 10.0 * (2.0 ** (self.worker_slashes[worker_id] - 1))
                self.worker_rewards[worker_id] = max(0, self.worker_rewards.get(worker_id, 0.0) - slash_penalty)
                return False

        # 2. Byzantine-robust filtering (Coordinate-wise Median)
        # We check if the update is an outlier compared to recent updates
        # This prevents "noise injection" that passes SPoT but degrades convergence.
        if not hasattr(self, 'update_history'):
            self.update_history = [] # List of recent dequantized updates

        # Dequantize update for filtering
        dequant_update = []
        for i, p in enumerate(self.params):
            bits = sparse_bits[i]
            scale_data = sparse_scales[i]
            idx = sparse_indices[i]
            
            if bits.numel() == 0:
                dequant_update.append(torch.zeros_like(p.data))
                continue

            mu, std = scale_data[0], scale_data[1]
            if std > 0:
                levels = torch.tensor([mu - 1.5*std, mu - 0.5*std, mu + 0.5*std, mu + 1.5*std], device=bits.device)
                val = levels[bits.long()]
            else:
                val = torch.full_like(bits, mu, dtype=torch.float32)
            
            full_val = torch.zeros_like(p.data).view(-1)
            full_val[idx] = val
            dequant_update.append(full_val.view(p.shape))

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

        # Update history
        self.update_history.append(dequant_update)
        if len(self.update_history) > 10:
            self.update_history.pop(0)

        # 3. Staleness compensation: reduce outer_lr if worker is behind
        staleness = self.global_version - worker_version
        effective_lr = self.outer_lr / (1.0 + 0.1 * max(0, staleness)) 

        for i, p in enumerate(self.params):
            val = dequant_update[i]
            idx = sparse_indices[i]
            
            if idx.numel() == 0:
                continue
            
            # 4. Momentum-based Aggregation (Outer Momentum with NAG)
            m = self.momentum_buffers[i]
            
            # Update momentum: m = beta * m + (1 - beta) * update
            m.view(-1)[idx].mul_(self.beta).add_(val.view(-1)[idx], alpha=1.0 - self.beta)
            
            if self.use_nag:
                # Nesterov update: p = p - lr * (beta * m + (1 - beta) * update)
                nag_update = self.beta * m.view(-1)[idx] + (1.0 - self.beta) * val.view(-1)[idx]
                p.data.view(-1)[idx] -= effective_lr * nag_update
            else:
                # Standard momentum update: p = p - lr * m
                p.data.view(-1)[idx] -= effective_lr * m.view(-1)[idx]
            
        self.global_version += 1
        self.worker_versions[worker_id] = self.global_version
        
        # 4. Lookahead Update: Slow weights update every k steps
        if self.use_lookahead:
            self.la_counter += 1
            if self.la_counter >= self.la_steps:
                self.la_counter = 0
                for i, p in enumerate(self.params):
                    # slow = slow + alpha * (fast - slow)
                    # Lookahead stabilizes the outer trajectory by interpolating 
                    # between the fast, asynchronously updated weights and the slow weights.
                    self.slow_weights[i].add_(p.data - self.slow_weights[i], alpha=self.la_alpha)
                    p.data.copy_(self.slow_weights[i])
        
        # 5. Incentive Design: Loyalty Bonus & Heterogeneity Factor
        # Reward = (Base + LoyaltyBonus) * HeterogeneityFactor
        # HeterogeneityFactor rewards workers using compression (helping bandwidth)
        loyalty = self.worker_loyalty.get(worker_id, 0)
        # Compounding loyalty bonus: 0.1 * log2(1 + loyalty)
        loyalty_bonus = 0.1 * torch.log2(torch.tensor(loyalty + 1.0)).item()
        
        # Check if worker is compressed (from verification_data or metadata)
        # arXiv:2601.02360: Heterogeneity-Aware Rewards
        is_compressed = verification_data.get('is_compressed', False) if verification_data else False
        hetero_factor = 1.2 if is_compressed else 1.0 # 20% bonus for being bandwidth-efficient
        
        reward = (1.0 + loyalty_bonus) * hetero_factor
        
        self.worker_rewards[worker_id] = self.worker_rewards.get(worker_id, 0.0) + reward
        self.worker_loyalty[worker_id] = loyalty + 1
        
        # 4. Dynamic Resource Allocation: Aggregator Migration
        # If the global version is a multiple of 1000, we could propose a new aggregator
        # based on highest reputation (loyalty + rewards).
        if self.global_version % 1000 == 0 and self.worker_rewards:
            top_worker = max(self.worker_rewards, key=self.worker_rewards.get)
            print(f"ELECTION: Worker {top_worker} is eligible for Aggregator (Citadel) promotion.")
            # In a real system, this would trigger a migration handshake.

        return True

    def slash_worker(self, worker_id: str):
        """
        Manually slashes a worker's rewards.
        """
        print(f"MANUAL SLASH: Worker {worker_id} slashed by operator.")
        self.worker_slashes[worker_id] = self.worker_slashes.get(worker_id, 0) + 1
        # Optionally reduce rewards as well
        if worker_id in self.worker_rewards:
            self.worker_rewards[worker_id] = max(0, self.worker_rewards[worker_id] - 10)

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

            # Get top-k largest absolute values
            abs_delta = torch.abs(flat_delta)
            _, expected_indices = torch.topk(abs_delta, k, sorted=False)
            expected_topk = flat_delta[expected_indices]
            
            # Jaccard similarity of indices as a proxy for 'honesty'
            set_expected = set(expected_indices.tolist())
            set_actual = set(sparse_indices[i].tolist())
            intersection = set_expected.intersection(set_actual)
            
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
