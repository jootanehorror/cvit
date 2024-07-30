from torch import nn
from torch.nn import functional as F
import torch
from einops import rearrange, repeat
from math import sqrt, pi, log

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output
    


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult, act=F.mish):
        super().__init__()
        self.norm_in = RMSNorm(dim)
        self.norm_out = RMSNorm(dim)
        hidden_dim = int(dim * ff_mult)

        self.w1 = nn.Linear(dim, 2 * hidden_dim, bias=False)

        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        self.act = act


        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.norm_in(x)

        x = self.w1(x)

        x, gate = x.chunk(2, dim=-1)

        x = self.act(x) * gate
        
        x = self.drop1(x)

        x = self.w2(x)

        x = self.norm_out(x)

        x = self.drop2(x)

        return x
    


class ConvModule(nn.Module):
    def __init__(self, dim, ff_mult, act=F.mish, groups=False):
        super().__init__()
        dim = dim//4
        hidden_dim = int(dim * ff_mult)

        self.pw1 = nn.Conv2d(dim, 2 * hidden_dim, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        
        if groups:
            self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,  padding=1, bias=False)

        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)

        self.act = act

        self.drop1 = nn.Dropout(0.05)
        self.drop2 = nn.Dropout(0.05)
        self.drop3 = nn.Dropout(0.05)

    def forward(self, x):
        x = rearrange(x,'b (h w) (p1 p2 c) -> b c (p1 h) (p2 w)',h=int(sqrt(x.size(1))), p1=2, p2=2)

        x, gate = self.pw1(x).chunk(2, dim=1)
        x = self.act(self.norm1(x)) * gate
        x = self.drop1(x)

        x = self.act(self.norm2(self.conv(x)))
        x = self.drop2(x)

        x = self.pw2(x)
        x = self.drop3(x)
        
        x = rearrange(x, 'b c (p1 h) (p2 w) -> b (h w ) (p1 p2 c)', p1=2, p2=2)
        return x
    

class Pool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv= nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size = 2, groups = dim, stride = 2, bias = False),
            nn.Conv2d(dim * 2, dim, kernel_size = 1, bias = False)
        )


    def forward(self, x):
  
        x = rearrange(x, 'b (h w) c -> b c h w', h = int(sqrt(x.shape[1])))
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x
    

class PatchEmbed(nn.Module):

    def __init__(self, patch_size, in_chans, dim):
        super().__init__()
        self.patch_size = patch_size

        patch_dim = in_chans * patch_size ** 2

        self.norm = RMSNorm(patch_dim)

        self.proj = nn.Linear(patch_dim, dim)
        
        self.norm_out = RMSNorm(dim)

    def forward(self, x):
 
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        x = self.norm(x)

        x = self.proj(x)

        x = self.norm_out(x)
        
        return x
