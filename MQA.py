# in MQA -> use multi query but same K,V for all heads.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiqueryAttention(nn.Module):
    def __init__(self,embd_dim, num_heads):
        super(MultiqueryAttention,self).__init__()
        self.embd_dim = embd_dim
        self.num_heads = num_heads

        self.head_dim = embd_dim // num_heads 
        assert embd_dim == num_heads * self.head_dim ,"embedding dim must be divisible by number of heads"

        self.q_linear = nn.Linear(embd_dim,embd_dim)
        self.k_linear = nn.Linear(embd_dim,self.head_dim)
        self.v_linear = nn.Linear(embd_dim,self.head_dim)

        # output final
        self.output = nn.Linear(embd_dim,embd_dim)

    def forward(self,x):
         
        B,S, _ = x.shape

        #  so q in multi head attn but k,v single fine
        Q = self.q_linear(x).view(B,S,self.num_heads,self.head_dim)
        Q = Q.permute(0,2,1,3) # B,n,s,h
        #  shared kv
        K = self.k_linear(x) # B,S,head_dim
        V = self.v_linear(x) # B,S,head_dim

        # now broadcast K,V to the same shape as Q -> in 4 B,1,S,head_dim
        # in short broadcast across heads
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)  

        # now attention scores then weihths
        attn_scores= torch.matmul(Q,K.transpose(-2,-1))/ self.head_dim**0.5   # b,n,s,h * b,1,h,s -> bnss
        attn_weights = F.softmax(attn_scores,-1)  # 
        attn_values = attn_weights @  V   

        # output   bsH
        out = attn_values.permute(0,2,1,3).contiguous()
        out = out.view(B,S,self.embd_dim)

        return self.output(out)

