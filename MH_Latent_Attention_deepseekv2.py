import torch 
import torch.nn as nn
import torch.nn.functional as F


class MLAv3(nn.Module):
    def __init__(self,num_head,embd_dim,latent_dim, rope_dim):
        super(MLAv3,self).__init__()
        
        self.num_head = num_head
        self.embd_dim = embd_dim
        self.head_dim = embd_dim//num_head
        assert rope_dim %2 ==0 , 'rope dim must be even'
        self.rope_dim = min(rope_dim, self.head_dim)
        self.rope_half = self.rope_dim//2
        assert embd_dim % num_head == 0, 'embd_dim must be divisible by the num head'
        self.latent_dim = latent_dim
        self.scale = (self.head_dim + self.rope_dim) ** -0.5

        #  Joint KV compression
        self.compress_kv = nn.Linear(embd_dim, latent_dim, bias=False)
        self.kv_proj = nn.Linear(latent_dim, 2 * embd_dim, bias=False)
        
        # query compression
        self.compress_q = nn.Linear(embd_dim,latent_dim,bias=False)
        self.q_proj = nn.Linear(latent_dim, embd_dim, bias=False) 

        # RoPE projections
        self.q_rope = nn.Linear(latent_dim,num_head*self.rope_dim, bias=False) # W_qr for q_R
        self.k_rope = nn.Linear(latent_dim,self.rope_dim,bias=False) # W_kr for shared k_R

        # out projection
        self.out_proj = nn.Linear(embd_dim,embd_dim,bias=False)

        self._init_rope()

    # dim (rope_dim,1)
    def _init_rope(self):
        inv_freq= 1.0 / (10000.0 ** (torch.arange(0,self.rope_dim,2,dtype=torch.float32) / self.rope_dim) )
        self.register_buffer('inv_freq', inv_freq,persistent=False)

        # adding caching
        self._cos_cache = {}
        self._sin_cache = {}
        self.max_cached_len = 0
        
    def forward(self,x):
        B,S,_ = x.shape

        # kv supress,upward
        c_kv = self.compress_kv(x)  # [B,S,latent_dim]
        kv = self.kv_proj(c_kv)   # [B,S,2*embd_dim]
        # query compress and upward
        c_q = self.compress_q(x) # [B,S,latent_dim]
        q_c = self.q_proj(c_q)  # [B,S, embd_dim]

        #  rope components
        q_r = self.q_rope(c_q) # [batch, S, num_head * rope_dim] - RoPE queries
        k_r = self.k_rope(c_kv)  # [batch, S, rope_embd]

        k_c, v = kv.chunk(2, dim=-1)  # [B,S,embd_dim], [B,S,embd_dim]
        # reshape
        q_c = q_c.view(B,S,self.num_head,self.head_dim).transpose(1,2) #[B,S, embd_dim]-> [B,S, nh, dh]-> [B,nh,S,dh]
        k_c = k_c.view(B,S,self.num_head,self.head_dim).transpose(1,2) #[B,S, embd_dim]-> [B,S, nh, dh]-> [B,nh,S,dh]
        v = v.view(B,S,self.num_head,self.head_dim).transpose(1,2) 

        q_r = q_r.view(B,S,self.num_head,self.rope_dim).transpose(1,2)
        # k_r-> [batch, S, rope_dim] -> [B,1,S,rope_dim] -> we access it nh times efficiently at dim 1
        k_r = k_r.unsqueeze(1).expand(-1,self.num_head,-1,-1) # shared across heads

        # apply rope to qr and kr
        q_r = self._apply_rope(q_r,S)
        k_r = self._apply_rope(k_r,S)

        # concat, q_i -> [q_ci , q_ri], k_i -> [k_ci,k_ri]
        q = torch.cat([q_c, q_r], dim=-1)  # [b, nh, s, dh + dr]
        k = torch.cat([k_c, k_r],dim = -1) # [b,nh,s, dh + dr]

        # attn compute
        attended_values = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True,  
            scale=self.scale
        )

        out = attended_values.transpose(1, 2).contiguous().view(B, S, self.embd_dim)
        return self.out_proj(out)
        

    def _get_rope_cache(self,S,device):
        if S <= self.max_cached_len and S in self._cos_cache:
            return self._cos_cache[S], self._sin_cache[S]
        pos = torch.arange(S, dtype=torch.float32, device=device)
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore  # dumbass still giving tensor thing
        freqs = torch.outer(pos, inv_freq)    
        cos_vals = torch.cos(freqs).view(1,1,S,self.rope_half)
        sin_vals = torch.sin(freqs).view(1,1,S,self.rope_half)

        self._cos_cache[S] = cos_vals
        self._sin_cache[S] = sin_vals
        self.max_cached_len = max(self.max_cached_len, S)

        return cos_vals,sin_vals
    

    def _apply_rope(self,x,S):

        cos_vals,sin_vals = self._get_rope_cache(S,x.device)
        
        x_even = x[..., 0::2]         # Dimensions 0, 2, 4, 6, ... 
        x_odd = x[..., 1::2]          # Dimensions 1, 3, 5, 7, ...

        # Fused rotation
        x_rotated = torch.empty_like(x[..., :self.rope_dim]) # B, num_head, S, rope_dim
        x_rotated[..., 0::2] = x_even * cos_vals - x_odd * sin_vals
        x_rotated[..., 1::2] = x_even * sin_vals + x_odd * cos_vals
        return x_rotated
    
