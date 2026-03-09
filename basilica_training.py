import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SubspaceCompressor:
    """
    Implements Subspace Projection for activation and gradient compression.
    Based on arXiv:2601.02360 and Subspace Networks.
    """
    def __init__(self, d_model: int, k: int):
        self.d_model = d_model
        self.k = k
        # Random orthonormal basis U
        U = torch.randn(d_model, k)
        self.U, _ = torch.linalg.qr(U)
        
    @torch.no_grad()
    def compress(self, x: torch.Tensor, residual_components: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compresses activations: X_tilde = (X - X_0)U
        """
        if residual_components is not None:
            x = x - residual_components
        return torch.matmul(x, self.U)

    @torch.no_grad()
    def decompress(self, x_tilde: torch.Tensor, residual_components: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decompresses activations: X_hat = X_tilde U^T + X_0
        """
        x_hat = torch.matmul(x_tilde, self.U.t())
        if residual_components is not None:
            x_hat = x_hat + residual_components
        return x_hat

    @torch.no_grad()
    def compress_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Compresses gradients: (grad)U
        """
        return torch.matmul(grad, self.U)

class HeterogeneousSparseLoCo:
    """
    Adapts SparseLoCo for heterogeneous Basilica environments.
    Combines SparseLoCo (arXiv:2508.15706) with Subspace Compression (arXiv:2601.02360).
    """
    def __init__(
        self, 
        model: nn.Module, 
        is_compressed: bool = False,
        k_ratio: float = 1/8,
        density: float = 0.03,
        d_model: int = 512,
        use_lamb: bool = True, # Use Layer-wise Adaptive Moments for Batch training
        use_sam: bool = True, # Use Sharpness-Aware Minimization for local updates
        sam_rho: float = 0.05
    ):
        self.model = model
        self.is_compressed = is_compressed
        self.density = density
        self.params = list(model.parameters())
        self.d_model = d_model
        self.use_lamb = use_lamb
        self.use_sam = use_sam
        self.sam_rho = sam_rho
        
        # SparseLoCo state
        self.error_buffers = [torch.zeros_like(p.data) for p in self.params]
        self.initial_weights = [p.data.clone().detach() for p in self.params]
        
        # LAMB state: first and second moments
        if use_lamb:
            self.m = [torch.zeros_like(p.data) for p in self.params]
            self.v = [torch.zeros_like(p.data) for p in self.params]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-6
            self.t = 0
            self.trust_ratio_clamp = (0.1, 10.0) # Clamp trust ratio for stability
        
        # Subspace state if compressed
        self.compressors = {}
        if is_compressed:
            # Initialize compressors for layers that match d_model (e.g., embeddings, attention outputs)
            k = max(1, int(d_model * k_ratio))
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    if module.weight.shape[-1] == d_model:
                        self.compressors[name] = SubspaceCompressor(d_model, k)
        
        # Embedding drift state (Equation 4: T_perp)
        self.t_perp = {}
        if is_compressed:
            for name, module in model.named_modules():
                if isinstance(module, nn.Embedding) and name in self.compressors:
                    self.t_perp[name] = torch.zeros_like(module.weight.data)

    @torch.no_grad()
    def get_sparse_update(self) -> dict:
        """
        Computes the sparse, quantized pseudo-gradient with Adaptive Quantization (AQ)
        and optional LAMB-style adaptive rate scaling.
        Implements Gradient-Informed Sparsity (GIS) to prioritize high-magnitude updates.
        """
        updates = []
        self.t += 1 if self.use_lamb else 0
        
        for i, p in enumerate(self.params):
            # Pseudo-gradient
            g = self.initial_weights[i] - p.data
            
            if self.use_lamb:
                # Update moments
                self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                
                # Bias correction
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Adaptive update
                u = m_hat / (torch.sqrt(v_hat) + self.eps)
                
                # Layer-wise trust ratio
                r1 = torch.norm(p.data)
                r2 = torch.norm(u)
                
                if r1 > 0 and r2 > 0:
                    trust_ratio = torch.clamp(r1 / r2, *self.trust_ratio_clamp)
                else:
                    trust_ratio = 1.0
                
                delta = trust_ratio * u
            else:
                delta = g

            ef_delta = delta + self.error_buffers[i]
            
            # Gradient-Informed Sparsity (GIS):
            # Instead of simple Top-k on delta, we use the magnitude of the 
            # pseudo-gradient to weight the selection, ensuring we prioritize 
            # updates that represent significant directional changes.
            flat = ef_delta.view(-1)
            
            # We use a power-law weighting for GIS: score = |delta| * (1 + |g|/max(|g|))
            # This biases selection towards elements that also had large raw gradients.
            g_flat = g.view(-1)
            g_max = torch.max(torch.abs(g_flat)) + 1e-8
            gis_score = torch.abs(flat) * (1.0 + torch.abs(g_flat) / g_max)
            
            k = max(1, int(self.density * flat.numel()))
            _, indices = torch.topk(gis_score, k, sorted=False)
            topk_values = flat[indices]
            
            # Adaptive 2-bit Quantization (AQ)
            if topk_values.numel() > 0:
                mu = torch.mean(topk_values)
                std = torch.std(topk_values) + 1e-8
                
                levels = torch.tensor([mu - 1.5*std, mu - 0.5*std, mu + 0.5*std, mu + 1.5*std], device=topk_values.device)
                diffs = torch.abs(topk_values.unsqueeze(-1) - levels)
                bits = torch.argmin(diffs, dim=-1).to(torch.int8)
                
                # Dequantize for error feedback
                dequant = levels[bits.long()]
                
                # Update EF buffer
                new_error = ef_delta.clone()
                new_error.view(-1)[indices] -= dequant
                self.error_buffers[i].copy_(new_error)
                
                updates.append({
                    'bits': bits,
                    'indices': indices,
                    'scale': torch.tensor([mu, std], device=topk_values.device),
                    'shape': p.shape
                })
            else:
                updates.append(None)
                
        return {'updates': updates, 'is_compressed': self.is_compressed, 'density': self.density}

    def adjust_density(self, network_latency: float):
        """
        Dynamically adjusts sparsity based on network conditions.
        If latency is high (> 2s), we decrease density (increase sparsity) to reduce payload size.
        If latency is low (< 0.5s), we increase density to speed up convergence.
        """
        if network_latency > 2.0: 
            self.density = max(0.001, self.density * 0.8)
            print(f"[DynamicDensity] High latency ({network_latency:.2f}s). Density -> {self.density:.4f}")
        elif network_latency < 0.5:
            self.density = min(0.1, self.density * 1.1)
            print(f"[DynamicDensity] Low latency ({network_latency:.2f}s). Density -> {self.density:.4f}")

    def handle_embedding_drift(self):
        """
        Implements Equation 3 & 4 from arXiv:2601.02360 to handle embedding drift
        in heterogeneous settings.
        """
        if not self.is_compressed:
            return

        for name, module in self.model.named_modules():
            if name in self.t_perp and name in self.compressors:
                U = self.compressors[name].U
                ts = module.weight.data
                t_perp = self.t_perp[name]
                
                # Π_S(TS) = TS U U^T
                # ts: [vocab_size, d_model], U: [d_model, k]
                proj_ts = torch.matmul(torch.matmul(ts, U), U.t())
                
                # Update T_perp: T_perp = T_perp + (TS - Π_S(TS))
                t_perp.add_(ts - proj_ts)
                
                # Project TS back to subspace: TS = Π_S(TS)
                ts.copy_(proj_ts)

    @torch.no_grad()
    def synchronize(self, global_weights: List[torch.Tensor], network_latency: Optional[float] = None):
        """
        Synchronize local weights with global weights and handle drift.
        """
        if network_latency is not None:
            self.adjust_density(network_latency)

        for i, p in enumerate(self.params):
            p.data.copy_(global_weights[i])
            self.initial_weights[i].copy_(global_weights[i])
            self.error_buffers[i].zero_()
        
        # Reset LAMB moments on sync to maintain consistency with global state
        if self.use_lamb:
            for i in range(len(self.params)):
                self.m[i].zero_()
                self.v[i].zero_()
            self.t = 0
        
        # After global sync, project embeddings back to subspace
        self.handle_embedding_drift()

    def local_step(self, inputs: torch.tensor, targets: torch.tensor, lr: float, use_fedsam: bool = True):
        """
        Performs a local training step with SAM (Sharpness-Aware Minimization).
        If use_fedsam is True, it implements Federated SAM which scales the 
        perturbation by the local model's distance from the global anchor, 
        improving global convergence in heterogeneous settings.
        """
        if not self.use_sam:
            # Standard SGD/local update
            outputs = self.model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            for p in self.params:
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)
                    p.grad.zero_()
            return loss.item()

        # SAM Step 1: Compute gradient at current point
        outputs = self.model(inputs)
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        
        # Compute epsilon = rho * grad / ||grad||
        grads = [p.grad.clone().detach() for p in self.params if p.grad is not None]
        norm = torch.sqrt(sum([g.norm()**2 for g in grads])) + 1e-12
        
        # FedSAM adaptation: scale rho by the relative drift from initial weights
        # This prevents local sharpness optimization from drifting too far from the global consensus.
        rho = self.sam_rho
        if use_fedsam:
            # We use an anchored version where drift is measured from initial_weights (global consensus)
            drift_norm = torch.sqrt(sum([(p.data - self.initial_weights[i]).norm()**2 for i, p in enumerate(self.params)]))
            # Scale rho: larger drift -> smaller rho to anchor the update
            # This ensures that as we drift from the consensus, we prioritize flatter minima 
            # that are closer to the global anchor.
            rho = rho / (1.0 + drift_norm)
            
        scale = rho / norm
        
        # Move to w + epsilon
        epsilons = []
        for p in self.params:
            if p.grad is not None:
                e = p.grad * scale
                p.data.add_(e)
                epsilons.append(e)
            else:
                epsilons.append(None)
        
        # SAM Step 2: Compute gradient at w + epsilon
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss_adv = F.mse_loss(outputs, targets)
        loss_adv.backward()
        
        # Move back to w and apply gradient from w + epsilon
        for i, p in enumerate(self.params):
            if epsilons[i] is not None:
                p.data.sub_(epsilons[i]) # Restore w
                p.data.add_(p.grad, alpha=-lr) # Update w with gradient from w + epsilon
                p.grad.zero_()
        
        return loss.item()
