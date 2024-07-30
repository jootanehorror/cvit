from torch import nn
from torch.nn import functional as F
import torch

from cvit.layers import *
from cvit.attention import LocalAttention



class Block(nn.Module):
    def __init__(self, dim, ff_mult, act, n_head, pooling, img_size=768):
        super().__init__()
        
        self.attn = LocalAttention(dim, n_head, img_size)
        self.ff = FeedForward(dim, ff_mult, act)
        self.convmod = ConvModule(dim, ff_mult=2, act=act, groups=False)
        
        if pooling:
            self.pool = Pool(dim)
            
        else:
            self.pool = nn.Identity()


    def forward(self, x):

        x = x + self.attn(x)

        x = x + self.convmod(x)

        x = x + self.ff(x)

        x = self.pool(x)

        return x
    


class CViT(nn.Module):
    def __init__(self, dim, n_head, ff_mult, act, patch_size=16, img_size=256, in_channels=3, n_layers=8, pool=[3, 5]):
        super().__init__()

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * (patch_size**2)

        self.patch = PatchEmbed(patch_size=patch_size, in_chans=in_channels, dim=self.patch_dim)
        
        self.blocks = nn.ModuleList([Block(dim,
                                            ff_mult,
                                            act,
                                            n_head,
                                            pooling=True if i in pool else False,
                                            img_size=img_size)
                                    for i in range(n_layers)]
                                    )

        self.norm_out = RMSNorm(dim)
      
    def forward(self, image):

        x = self.patch(image)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        
        return x
