import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embd_dim, num_heads):
        super(MultiheadAttention,self).__init__()
        self.embd_dim = embd_dim
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads

        assert self.head_dim * num_heads == self.embd_dim, " embedding dim must be divisible by number of heads"

        # Linear layers q,k,v
        self.q_linear = nn.Linear(embd_dim,embd_dim)
        self.k_linear = nn.Linear(embd_dim,embd_dim)
        self.v_linear = nn.Linear(embd_dim,embd_dim)

        # output linear layer (for that attn weights with @V)
        self.out_linear = nn.Linear(embd_dim,embd_dim)

    def forward(self,x):
        B,S = x.shape[0], x.shape[1]

        # now calculate Q,K,V
        Q = self.q_linear(x).view(B,S,self.num_heads,self.head_dim)
        K = self.k_linear(x).view(B,S,self.num_heads,self.head_dim)
        V = self.v_linear(x).view(B,S,self.num_heads,self.head_dim)

        # now let's reshape it to b,n,s,h
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # B,n,S,h * B,n,h,S -> B,n,S,S 
        # compute -> 2bns2 
        attn_scores = torch.matmul(Q,K.transpose(-2,-1))/ self.head_dim ** 0.5 # 
        # B,n,S,S -> batchsize, numheads, query seqlen, keyseqlen -> sq sk

        # casual# S,S -> (1, 1 ,s, s )
        mask = torch.tril(torch.ones(S,S)).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)
        attn_weights = F.softmax(attn_scores, dim = -1) # at final sk
        # b,n,s,s * b,n,s,h -> b,n,s, h
        #attended values
        attended_values = torch.matmul(attn_weights, V) # bnsh
        
        # concat attn heads 
        attended_values = attended_values.transpose(1,2).contiguous().view(B,S,self.embd_dim)
        
        # output projection
        output = self.out_linear(attended_values)
        return output