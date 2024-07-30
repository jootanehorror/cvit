import torch
from einops import rearrange, repeat
from torch.cuda.amp import autocast
from torch import nn
from math import sqrt, pi, log


@autocast(enabled = False)
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    @autocast(enabled = False)
    def forward(self, x):
        device, dtype, n = x.device, x.dtype, int(sqrt(x.shape[-2]))

        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq = seq * scales * pi

        x_sinu = repeat(seq, 'i d -> i j d', j = n)
        y_sinu = repeat(seq, 'j d -> i j d', i = n)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))

        return sin, cos 
    
    def apply_rotary(self, q, k, sin, cos):
        B, H, N ,D = q.size()
        q = rearrange(q, 'b h n d ->  (b h) n d')
        k = rearrange(k, 'b h n d ->  (b h) n d')

        q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))

        q = rearrange(q, '(b h) n d -> b h n d', b=B)
        k = rearrange(k, '(b h) n d -> b h n d', b=B)


        return q, k
