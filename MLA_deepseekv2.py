import torch 
import torch.nn as nn
import torch.nn.functional as F


class MLAv3(nn.Module):
    def __init__(self,num_head,embd_dim,latent_dim, rope_dim):
        # inherit nn.Module
        super(MLAv3,self).__init__()

        # init 
        self.num_head = num_head
        self.embd_dim = embd_dim
        self.head_dim = embd_dim//num_head
        self.rope_dim = min(rope_dim, self.head_dim)
        self.rope_half = self.rope_dim//2
        assert rope_dim %2 ==0 , 'rope dim must be even'
        assert embd_dim % num_head == 0, 'embd_dim must be divisible by the num head'
        self.latent_dim = latent_dim
        self.scale = (self.head_dim + self.rope_dim) ** -0.5

        #  Joint KV compression
        self.compress_kv = nn.Linear(embd_dim, latent_dim, bias=False)
        self.k_proj = nn.Linear(latent_dim,embd_dim,bias=False)
        self.v_proj = nn.Linear(latent_dim,embd_dim,bias=False)
        
        # query compression
        self.compress_q = nn.Linear(embd_dim,latent_dim,bias=False)
        self.q_proj = nn.Linear(latent_dim, embd_dim, bias=False) 

        # RoPE projections
        self.q_rope = nn.Linear(latent_dim,num_head*rope_dim, bias=False) # W_qr for q_R
        self.k_rope = nn.Linear(embd_dim,rope_dim,bias=False) # W_kr for shared k_R

        """
        # Only need to store one k_r per position in KV cache
        # q_r from c_q is fine since queries are computed fresh each time
        # Total KV cache = (d_c + d_R_h) * l  # Not (d_c + num_head * d_R_h) * l
        # TL:dr,Shared key strategy - one RoPE key per position (from x) shared across all heads,
        # but separate RoPE queries per head (from c_q). Saves memory while preserving positional info

        # self.q_rope = nn.Linear(latent_dim,num_head*rope_dim, bias=False) 
        # self.q_rope = nn.Linear(embd_dim, num_head*rope_dim, bias=False)  
        # Current:    latent_dim * (num_head * rope_dim)
        # Alternative: embd_dim * (num_head * rope_dim)
        # Increase:   (embd_dim - latent_dim) * (num_head * rope_dim)
        """

        # out projection
        self.out_proj = nn.Linear(embd_dim,embd_dim,bias=False)

        self._init_rope()

    # dim (rope_dim,1)
    def _init_rope(self):
        inv_freq= 1.0 / (10000.0 ** (torch.arange(0,self.rope_dim,2,dtype=torch.float32) / self.rope_dim) )
        self.register_buffer('inv_freq', inv_freq,persistent=False)
        
    def forward(self,x):
        B,S,_ = x.shape

        # kv supress,upward
        c_kv = self.compress_kv(x)  # [B,S,latent_dim]
        k_c = self.k_proj(c_kv)       # [B,S,embd_dim]
        v = self.v_proj(c_kv)   
        # query compress and upward
        c_q = self.compress_q(x) # [B,S,latent_dim]
        q_c = self.q_proj(c_q)  # [B,S, embd_dim]

        #  rope components
        q_r = self.q_rope(c_q) # [batch, seq_len, num_head * rope_dim] - RoPE queries
        k_r = self.k_rope(x)  # [batch, seq_len, rope_embd]


        # reshape
        q_c = q_c.view(B,S,self.num_head,self.head_dim).transpose(1,2) #[B,S, embd_dim]-> [B,S, nh, dh]-> [B,nh,S,dh]
        k_c = k_c.view(B,S,self.num_head,self.head_dim).transpose(1,2) #[B,S, embd_dim]-> [B,S, nh, dh]-> [B,nh,S,dh]
        v = v.view(B,S,self.num_head,self.head_dim).transpose(1,2) # bdt

        q_r = q_r.view(B,S,self.num_head,self.rope_dim).transpose(1,2)
        # k_r-> [batch, seq_len, rope_dim] -> [B,1,S,rope_dim] -> we access it nh times efficiently at dim 1
        k_r = k_r.unsqueeze(1).expand(-1,self.num_head,-1,-1) # shared across heads

        # apply role to qr and kr
        q_r = self._apply_rope(q_r,S)
        k_r = self._apply_rope(k_r,S)

        # concat, q_i -> [q_ci , q_ri], k_i -> [k_ci,k_ri]
        q = torch.cat([q_c, q_r], dim=-1)  # [b, nh, s, dh + dr]
        k = torch.cat([k_c, k_r],dim = -1) # [b,nh,s, dh + dr]

        # attn compute
        
        scores = torch.matmul(q,k.transpose(-2,-1))*self.scale # b,nh,s,s
        attn_mask = torch.tril(torch.ones(S,S)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores,dim = -1)
        attended_values = torch.matmul(attn_weights,v) # b,nh,s,s -> b,nh,s,embd_dim -> b,nh,s,embd_dim

        out = attended_values.transpose(1,2).contiguous().view(B,S,self.embd_dim)
        return self.out_proj(out)
        


    def _apply_rope(self,x,S):
        # f(x,m) = xcos(m.theta) + rotate(x).sin(m.theta)

        # RoPE rotates pairs of dimentions by position-dependent angles
        # for position t and dim pair(2i , 2i+1):
        # [x_2i,x_2i+1] gets rotated by angle θ_i *t

        # rotation formula
        # θ_i = 1/ (10k^ (2i/rope_dim)) for i = 0,... rope_dim /2
        # cost_t = cos(θ_i * t)
        # sin_t = sin(θ_i * t)

        # rotation matrix applied to each pair
        # [x'_2i] =   [cos_t  -sin_t] [x_2i ]
        # [x'_2i+1] = [sin_t    cos_t] [x_2i+1]
        pos = torch.arange(S,dtype=torch.float32)
        # bilinear operation (n,) * (m,) -> (n,m) 
        freqs = torch.outer(pos, self.inv_freq) # type: ignore
        #(S,rope_half(same as dh/2) )
        cos_vals = torch.cos(freqs).view(1,1,S, self.rope_half)
        sin_vals = torch.sin(freqs).view(1,1,S,self.rope_half)

        x_rope = x[..., :self.rope_dim]    # First rope_dim dimensions to rotate
        x_pass = x[..., self.rope_dim:]    # Remaining dimensions (unchanged)
        

        x_even = x_rope[..., 0::2]         # Dimensions 0, 2, 4, 6, ... 
        x_odd = x_rope[..., 1::2]          # Dimensions 1, 3, 5, 7, ...

        rotated_even = x_even * cos_vals - x_odd * sin_vals
        rotated_odd = x_even * sin_vals + x_odd * cos_vals

        rope_rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)
        # Stack: [..., rope_half, 2] -> Flatten: [..., rope_dim]
        # This interleaves: [even0, odd0, even1, odd1, even2, odd2, ...]
        if x_pass.size(-1) > 0:
            return torch.cat([rope_rotated, x_pass], dim=-1)
        else:
            return rope_rotated





# RoPE applies position-dependent rotations to query and key vectors
# It needs to operate on the final Q and K representations that will be used in attention
# The rotation is position-specific and must be applied right before the dot-product attention
        
# MLA's Challenge with rope:
# in standard, MLA compress KV jointly
# then up project k,v
# but if you apply the RoPE after up-projection, you loose the compressin benefit for k since the RoPE changes k

# Standard MLA: h → compress → C_kv → up_project → K, V
# With RoPE:    h → compress → C_kv → up_project → K → RoPE(K) ❌
# Problem: RoPE(K) breaks the low-rank structure!

# This is why DeepSeek-V2 had to develop a modified approach - they needed to find a way to:

# Keep the compression benefits of MLA
# Still apply position embeddings effectively
# Maintain the mathematical properties RoPE requires
# The solution involves careful architectural changes to make RoPE work
# with the compressed representations while preserving both efficiency gains and positional awareness.


# As a solution, we propose the decoupled RoPE strategy that uses additional multi-head
#  queries q𝑅
#  𝑡,𝑖 
# ∈ R𝑑𝑅
#  ℎ and a shared key k𝑅
#  𝑡 ∈ R𝑑𝑅
#  ℎ to carry RoPE, where 𝑑𝑅
#  ℎ 
# denotes the per-head
#  dimension of the decoupled queries and key. Equipped with the decoupled RoPE strategy, MLA
#  performs the following computation:


"""

# Compressed path (maintains low-rank benefits):
h → compress → C_kv → up_project → K_c, V
h → compress → C_q → up_project → Q_c


# RoPE path (separate position-aware components):
C_q → W_qr → Q_r → RoPE(Q_r)  # Multi-head RoPE queries
C_kv → W_kr → K_r → RoPE(K_r) # Shared RoPE key


# Concatenation strategy:
Q_final = [Q_c, Q_r]  # Concat compressed + RoPE queries
K_final = [K_c, K_r]  # Concat compressed + RoPE keys
V_final = V           # Only values, no RoPE needed


Empirical Justification
The paper shows this works because:
Most attention patterns are dominated by content similarity (handled by compressed path)
Positional relationships only need a smaller dimensional space (rope_dim << head_dim)
The combination captures both efficiently
TL;DR: It works because it separates "what" (content) from "where" (position),
compresses the "what" heavily, and keeps "where" in a smaller dedicated space! 
"""







# so the core of MLA is the low rank joint compression for keys and values to reduce kv cache:

# C_kv = W_dkv*h -> compressed kv, w_dkv -> down projection             # 𝑑𝑐 (≪𝑑ℎ.𝑛ℎ)
# K_c = W_uk * c_kv -> key up_projection from compressed
# V_c = W_uv * c_kv -> values up_projection form the compressed


# and over that, in order to reduce the activation memory during training, also perform 
# low rank compression for the queries, even if it cannot reduce the kv cache.

# C_q = W_dq*h ->  compressed query, W_dq -> down projection weight
# Q_c = W_uq* C_q -> up projection of query, W_uq -> up_projection from compressed one

#  c_q ∈ R𝑑′-> the compressed latent vector for queries
# 𝑊𝐷𝑄 ∈ R𝑑′
# let's code mha first then convert it to mla
# q_r -> rope(W_qr * C_q)
# k_t -> rope(W_kr * h)
# q_i -> [q_Ci, q_Ri]
# k_i ->[k_Ci, k_Ri]
# o_i -> softmax(...) @ v
# u -> W_o[o_1,....._o_n]

#  During inference, the decoupled key should also be cached. Therefore,
#  DeepSeek-V2 requires a total KV cache containing (𝑑𝑐 +𝑑𝑅ℎ)𝑙 elements.



