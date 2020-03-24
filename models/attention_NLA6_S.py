import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import time

import math


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, rpe=False, scale=False, return_kv=False, layer=0):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.rpe = rpe
        self.scale = scale
        self.return_kv = return_kv
        self.layer = layer

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        if self.rpe:
            self.rel_k = nn.Parameter(torch.randn(out_channels, 1, kernel_size), requires_grad=True) # 32 x 1 x 7

        self.out_channels = out_channels

        self.nl_ratio = 0.25
        self.nl_channels = int(self.nl_ratio*self.out_channels)
        self.l_channels = self.out_channels - self.nl_channels

        self.query_conv = nn.Conv2d(in_channels, self.l_channels, kernel_size=1, bias=bias)
        self.key_conv = nn.Conv2d(in_channels, self.l_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, self.l_channels, kernel_size=1, bias=bias)
        self.last_conv = nn.Conv2d(self.l_channels // self.groups, self.l_channels // self.groups, kernel_size=1, bias=bias)

        self.nl_query_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl_key_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl_value_conv = nn.Conv2d(in_channels//2, self.nl_channels, kernel_size=1, bias=bias)
        self.nl_last_conv = nn.Conv2d(self.nl_channels // self.groups, self.nl_channels // self.groups, kernel_size=1, bias=bias)

        self.act = nn.Tanh()


    def forward(self, x, abs_x, idx):
        batch, channels, npoints, neighbors = x.size() # B, C, N, K
        # B : batch
        # G : groups
        # C : local operation channels
        # C' : non-local operation channels
        # K : neighbors, K = 20 for local / non-local operation both.
        # C + C' = C_out 

        ''' 1. Local operation '''
        ''' 1.1. get point features '''
        x_l_qkv = x # B, C, N, K

        ''' 1.2. transform by Wq, Wk, Wv '''
        q_out = self.query_conv(x_l_qkv) # B, C, N, K
        k_out = self.key_conv(x_l_qkv) # B, C, N, K
        v_out = self.value_conv(x_l_qkv) # B, C, N, K

        ''' 1.3. Multi-heads for local operations. '''
        q_out = q_out.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        k_out = k_out.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K
        v_out = v_out.view(batch, self.groups, self.l_channels // self.groups, npoints, -1) # B, G, C//G, N, K

      
        #TODO RPE for non-local operation?
        ''' 1.4. relative positional encoding ''' 
        if self.rpe:
            k_out = k_out + self.rel_k 
  
        # k_out : B, C, N, K / self.rel_k : C, 1, K

        ''' 1.5. Addressing '''
        if self.scale:
            scaler = torch.tensor([self.l_channels / self.groups]).cuda()
            out = torch.rsqrt(scaler) * (q_out * k_out).sum(2) # B, G, N, K
        else:
            out = (q_out * k_out).sum(2) # B, G, N, K

        out = F.softmax(out, dim=-1) # B, G, N, K

        ''' 1.6. Attention score for the node selection (Different per groups) '''
        idx_zeros = torch.zeros(batch, self.groups, npoints, npoints, device='cuda')
        idx_score = idx.repeat(1, self.groups, 1, 1) # B, G, N, K

        idx_zeros.scatter_(dim=3, index = idx_score, src=out) # gradient path provided by out(i.e. statistics of local operation.) -> no linear projection layer like 'self-attention graph pooling' is needed.
        score = idx_zeros.sum(dim=2, keepdim=True) # B, G, 1, N

        val_score, idx_score = score.topk(k=neighbors, dim=3) # B, G, 1, N -> B, G, 1, K'

        ''' 1.7. Scaling V '''
        out = out.unsqueeze(2).expand_as(v_out) # B, G, C//G, N, K
        out = torch.einsum('bgcnk,bgcnk -> bgcn', out, v_out) # b, G, C//G, N, K -> B, G, C//G, N

        if self.scale:
            out = out.permute(0, 2, 1, 3) # B, C//G, G, N 
            out = self.last_conv(out).permute(0, 2, 1, 3) # B, G, C//G, N


        ''' 1.8. Concat heads  ''' 
        out = out.contiguous().view(batch, -1, npoints, 1) # B, G, C//G, N -> B, C, N, 1


        ''' 2. Non-local MHA over selected nodes '''
        # q_nl : B, G, C'//G, N
        # k_nl, v_nl : B, G, C'//G, K'
        # omit nl.

        # (original)
        # qTk : (B, G, N, C'//G) x (B, G, C'//G, N) = B, G, N, N
        # qTkV : (B, G, N, N) x (B, G, N, C'//G) = B, G, N, C//G 

        # (select) -> memory / flops friendly, while maintaining non-local diffusion.
        # qTk : (B, G, N, C'//G) x (B, G, C'//G, K') = B, G, N, K'
        # qTkV : (B, G, N, K') x (B, G, K', C'//G) = B, G, N, C'//

        ''' 2.1. get point features '''
        x_nl_qkv = abs_x

        ''' 2.2. transform by Wq, Wk, Wv '''
        q_nl_out = self.nl_query_conv(x_nl_qkv) # B, C', N
        k_nl_out = self.nl_key_conv(x_nl_qkv) # B, C', N
        v_nl_out = self.nl_value_conv(x_nl_qkv) # B, C', N

        ''' 2.3. Multi-heads for non-local operations. '''
        q_nl_out = q_nl_out.view(batch, self.groups, self.nl_channels // self.groups, npoints) # B, G, C'//G, N
        k_nl_out = k_nl_out.view(batch, self.groups, self.nl_channels // self.groups, npoints) # B, G, C'//G, N
        v_nl_out = v_nl_out.view(batch, self.groups, self.nl_channels // self.groups, npoints) # B, G, C'//G, N

        ''' 2.4. select q, k, v by top-k idx '''
        idx_score = idx_score.repeat(1,1,self.nl_channels // self.groups, 1) # B, G, 1, K' -> B, G, C'//G, K'
        val_score = val_score.repeat(1,1,self.nl_channels // self.groups, 1) # B, G, 1, K' -> B, G, C'//G, K'
        k_nl_out = torch.gather(k_nl_out, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'
        v_nl_out = torch.gather(v_nl_out, 3, idx_score) # B, G, C'//G, N -> B, G, C'//G, K'

        #TODO : activation is needed or not? it's related to soften & linear procedure.
        v_nl_out = v_nl_out * self.act(val_score) # activation

        ''' 2.5. multiply attention score for providing gradient path to self.scoring '''
        out_all = torch.matmul(torch.transpose(q_nl_out, 2,3), k_nl_out) # B, G, N, K'
        out_all = F.softmax(out_all, dim=-1) # B, G, N, K'

        if self.scale:
            scaler = torch.tensor([self.nl_channels / self.groups]).cuda()
            out_all = torch.rsqrt(scaler) * out_all # B, G, N, K'

        out_all = torch.matmul(out_all, torch.transpose(v_nl_out, 2, 3)) # B, G, N, C'//G

        if self.scale:
            out_all = self.nl_last_conv(out_all.permute(0, 3, 1, 2)).permute(0, 2, 1, 3) # B, G, N, C'//G -> B, C'//G, G, N -> B, G, C'//G, N
            out_all = out_all.contiguous().view(batch, -1, npoints, 1) # B, C', N, 1

        else:
            out_all = out_all.permute(0, 1, 3, 2).contiguous().view(batch, -1, npoints, 1) # B, G, N, C'//G -> B, G, C'//G, N -> B, C', N, 1

        ''' 3. Concat '''
        if self.layer > 0:
            out = torch.cat([out, out_all], dim=1)
        return out
