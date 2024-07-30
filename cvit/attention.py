import math

import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat

from cvit.rotary import AxialRotaryEmbedding
from cvit.layers import RMSNorm

def create_attention_mask(image):
    b, hw, d = image.shape
    w = int(math.sqrt(hw))
    image = rearrange(image, 'b (h w) c -> b c h w', w = w)

    mask = torch.ones((w*w, w*w), dtype=torch.bool)

    for i in range(w):
        for j in range(w):
            idx_center = i*w + j
            
            
            start_h = max(0, i - 1)
            end_h = min(w, i + 2)
            start_w = max(0, j - 1)
            end_w = min(w, j + 2)
            
            for ii in range(start_h, end_h):
                for jj in range(start_w, end_w):
                    idx_neighbour = ii*w + jj
                    mask[idx_center, idx_neighbour] = False

    return mask


class LocalAttention(nn.Module):
    def __init__(self, d_model, n_head, img_size=768):
        super().__init__()
        head_dim = d_model // n_head
        self.norm = RMSNorm(d_model)
        self.n_head = n_head
        self.attn_mask = None
 
        self.qkv = nn.Linear(d_model ,3 * d_model , bias = False)
        self.out = nn.Linear(d_model, d_model,  bias=False)
        self.rotary = AxialRotaryEmbedding(head_dim, img_size)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        B, HW, D = x.size()
        device = x.device
        h = self.n_head
        sin, cos = self.rotary(x)

        x = self.norm(x)
        
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b n (k h d) -> b n k h d', h=h, k=3)

        q, k, v = qkv.chunk(3, dim=2)
        
        q = rearrange(q.squeeze(2), 'b n h d -> b h n d')
        k = rearrange(k.squeeze(2), 'b n h d -> b h n d')
        v = rearrange(v.squeeze(2), 'b n h d -> b h n d')

        q, k = self.rotary.apply_rotary(q, k, sin, cos)

        if self.attn_mask is None:
            attn_mask = create_attention_mask(x)
            self.attn_mask = attn_mask
            
        if self.training:
            p = 0.1
        else:
            p = 0.0

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask.to(device), dropout_p=p)
        out = rearrange(out, 'b n l d -> b l (n d)')
  
        out = self.out(out)
        
        return self.drop(out)
