import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import math

"""
for intuation, first try to code deepseekv2 without tensor,layer distribution.
will change it later for more efficient one
"""

# deepseek V2
# Uses absorption mechanism where K,V are "absorbed" into a shared representation
# Employs low-rank decomposition for K,V projections
# Has decoupled rotary positional encoding
class MLA_v2(nn.Module):
    def __init__(self, embd_dim, num_head, latent_dim, rope_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.num_head = num_head
        self.head_dim = embd_dim // num_head
        self.rope_dim = min(rope_dim, self.head_dim)
        assert self.rope_dim % 2 == 0, "rope_dim must be even"
        assert embd_dim % num_head == 0, "Embedding dim must be divisible by number of heads"
        self.rope_half = self.rope_dim // 2
        self.scale = self.head_dim ** -0.5
        self.latent_dim = latent_dim
        
        self.compress = nn.Linear(embd_dim, latent_dim, bias=False)
        self.q_proj = nn.Linear(latent_dim, embd_dim, bias=False)
        self.kv_absorb = nn.Linear(latent_dim, latent_dim, bias=False)
        self.k_up = nn.Linear(latent_dim, embd_dim, bias=False)
        self.v_up = nn.Linear(latent_dim, embd_dim, bias=False)
        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_dim > 0:
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)
        else:
            self.register_buffer('inv_freq', torch.empty(0), persistent=False)

    def forward(self, x):
        B, S, _ = x.shape
        
        device = x.device
        compressed = self.compress(x)
        
        Q = self.q_proj(compressed)
        Q = Q.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        
        
        absorbed = self.kv_absorb(compressed)
        K_shared = self.k_up(absorbed)
        V_shared = self.v_up(absorbed)
        
        K = K_shared.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        V = V_shared.view(B, S, self.num_head, self.head_dim).transpose(1, 2)
        
        if self.rope_dim > 0:
            Q, K = self._apply_rope(Q, K, S, x.device)
        
        attn_out = F.scaled_dot_product_attention(Q, K, V, scale=self.scale)
        out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embd_dim)
        
        return self.out_proj(out)

    def _apply_rope(self, Q, K, seq_len, device):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32) 
        freqs = torch.outer(positions, self.inv_freq)  # type: ignore
        cos_freqs = torch.cos(freqs).view(1, 1, seq_len, self.rope_half)
        sin_freqs = torch.sin(freqs).view(1, 1, seq_len, self.rope_half)
        
        Q = self._rope_transform(Q, cos_freqs, sin_freqs)
        K = self._rope_transform(K, cos_freqs, sin_freqs)
        
        return Q, K

    def _rope_transform(self, x, cos_vals, sin_vals):
        x_rope = x[..., :self.rope_dim]
        x_pass = x[..., self.rope_dim:]
        
        x_even = x_rope[..., 0::2]
        x_odd = x_rope[..., 1::2]
        
        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals
        
        rope_rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
        
        if x_pass.size(-1) > 0:
            return torch.cat([rope_rotated, x_pass], dim=-1)    
        else:
            return rope_rotated

