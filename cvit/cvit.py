from torch import nn
from torch.nn import functional as F
import torch

from cvit.layers import *
from cvit.attention import *



class Block(nn.Module):
    def __init__(self, dim, ff_mult, act, n_head, pooling, img_size=768):
        super().__init__()
        
        self.attn = LocalAttention(dim, n_head, img_size)
        self.ff = FeedForward(dim, ff_mult, act)
        self.convmod = ConvModule(dim, ff_mult=2, act=act, groups=False)
        
        if pooling:
            self.pool = Pool(dim, n_head)
            
        else:
            self.pool = nn.Identity()


    def forward(self, x):

        x = x + self.attn(x)

        x = x + self.convmod(x)

        x = x + self.ff(x)

        x = self.pool(x)

        return x



class LatentBlock(nn.Module):
    def __init__(self, dim, ff_mult, act, n_head, patch_size, n=512):
        super().__init__()
        
        self.attn = LatentAttention(dim, n_head, n)

        self.ff = FeedForward(dim, ff_mult, act)
        
        self.pool = AttentionPool2d(dim, n_head, patch_size)
    


    def forward(self, x):

        x = x + self.attn(x)

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

      
    def forward(self, image):

        x = self.patch(image)

        for block in self.blocks:
            x = block(x)
   
        return x




class CViTClassification(nn.Module):
    def __init__(self, dim, n_head, ff_mult, act, patch_size=16, img_size=256, in_channels=3, n_layers=8, pool=[3, 5, 6], n=128, num_classes =1000):
        super().__init__()
        
        self.body = CViT(dim, n_head, ff_mult, act, patch_size, img_size, in_channels, n_layers, pool)
        
        latent_patch_size = (img_size // (patch_size * (2 ** len(pool))))
        
        self.latent_block = LatentBlock(dim, ff_mult, act, n_head, latent_patch_size, n)

        self.head = nn.Linear(dim, num_classes, bias=False)
      
    def forward(self, image):

        x = self.body(image)

        x = self.latent_block(x)

        x = self.head(x)

        return x
