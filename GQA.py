

class GroupQueryAttention(nn.Module):
    def __init__(self, embd_dim, num_heads,num_groups):
        super(GroupQueryAttention,self).__init__()
        assert num_heads % num_groups == 0 , "num_heads must be divisible by num_groups"
        assert embd_dim % num_groups == 0 , "embd_dim must be divisible by num_heads"

        self.embd_dim = embd_dim  
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = embd_dim//num_heads
        self.group_heads= num_heads//num_groups
# num heads, kv in groups, so need num of gps, then we get how many heads per gp 
#  24 and 4 then heads per gp -> 6

        # initialize qkv ,W (use proj instead of linear)
        self.q_proj = nn.Linear(embd_dim,embd_dim)
        self.k_proj = nn.Linear(embd_dim,self.head_dim* num_groups)
        self.v_proj = nn.Linear(embd_dim,self.head_dim * num_groups)
        # final projection
        self.out_proj = nn.Linear(embd_dim,embd_dim)

    
    def forward(self,x):
        B,S,_ = x.shape

        #shape will be B,S,embd_dim -> # (B, num_heads, S, head_dim)
        Q = self.q_proj(x).view(B,S,self.num_heads,self.head_dim).transpose(1,2)
        #  B,S,embd_dim -> # (B, kv_groups, S, head_dim)
        K = self.k_proj(x).view(B,S, self.num_groups,self.head_dim).transpose(1,2)
        V = self.v_proj(x).view(B,S,self.num_groups,self.head_dim).transpose(1,2)

        #  so i've Q -> B,nh, S,h_d
        #  and KV -> B,ng, S, h_d
        # repeat or map KV so each head get KV (Q- 12 heads, KV - 4 heads , if gps is 3 heads)
        # maping -> Map each head index to its group index ([0,0,0,1,1,1,2,2,2,3,3,3])
        device = x.device
        head_to_group = torch.arange(self.num_heads,device=device) // self.group_heads
        # shape  = (num_heads, )
        # index K and V per head's group 
        # Uses shared memory (no copies): shape → (B, nh, S, h_d)
        K = K[:,head_to_group] # (B, nh, S, h_d)
        V = V[:,head_to_group] # (B, nh, S, h_d)
        # attn socre, weights, value weights, output 
        attn_score = torch.matmul(Q,K.transpose(-2,-1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(attn_score,-1)

        # value weihgts  V -> B,nh, S, h_d and attn_weights ->  B,nh, S,S 
        # (B,nh, S,S) *  (B,nh, S, h_d) -> B,nh,s,h_d
       
        value_weights = torch.matmul(attn_weights,V).transpose(1,2).contiguous().view(B,S,self.embd_dim)

        return self.out_proj(value_weights)

