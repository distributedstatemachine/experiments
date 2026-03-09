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
        d_model: int = 512
    ):
        self.model = model
        self.is_compressed = is_compressed
        self.density = density
        self.params = list(model.parameters())
        self.d_model = d_model
        
        # SparseLoCo state
        self.error_buffers = [torch.zeros_like(p.data) for p in self.params]
        self.initial_weights = [p.data.clone().detach() for p in self.params]
        
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
        Computes the sparse, quantized pseudo-gradient.
        """
        updates = []
        for i, p in enumerate(self.params):
            delta = self.initial_weights[i] - p.data
            ef_delta = delta + self.error_buffers[i]
            
            # Top-k
            flat = ef_delta.view(-1)
            k = max(1, int(self.density * flat.numel()))
            abs_val, indices = torch.topk(torch.abs(flat), k, sorted=False)
            topk_values = flat[indices]
            
            # 2-bit Quantization
            if topk_values.numel() > 0:
                scale = torch.max(torch.abs(topk_values))
                signs = torch.sign(topk_values)
                abs_vals = torch.abs(topk_values)
                
                bits = torch.zeros_like(topk_values, dtype=torch.int8)
                mask_high = abs_vals > (scale / 2)
                
                bits[(signs > 0) & mask_high] = 3
                bits[(signs > 0) & (~mask_high)] = 2
                bits[(signs <= 0) & (~mask_high)] = 1
                bits[(signs <= 0) & mask_high] = 0
                
                # Dequantize for error feedback
                dequant = torch.zeros_like(topk_values)
                dequant[bits == 3] = scale
                dequant[bits == 2] = scale / 3
                dequant[bits == 1] = -scale / 3
                dequant[bits == 0] = -scale
                
                # Update EF buffer
                new_error = ef_delta.clone()
                new_error.view(-1)[indices] -= dequant
                self.error_buffers[i].copy_(new_error)
                
                updates.append({
                    'bits': bits,
                    'indices': indices,
                    'scale': scale,
                    'shape': p.shape
                })
            else:
                updates.append(None)
                
        return {'updates': updates, 'is_compressed': self.is_compressed}

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
    def synchronize(self, global_weights: List[torch.Tensor]):
        """
        Synchronize local weights with global weights and handle drift.
        """
        for i, p in enumerate(self.params):
            p.data.copy_(global_weights[i])
            self.initial_weights[i].copy_(global_weights[i])
            self.error_buffers[i].zero_()
        
        # After global sync, project embeddings back to subspace
        self.handle_embedding_drift()
