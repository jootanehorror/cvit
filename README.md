## CViT - Pytorch

CViT - PyTorch: A novel Vision Transformer architecture that integrates convolutional blocks between patches, merging the advantages of convolutional neural networks and transformers for improved visual data processing.

## Parameters

- `dim`: int
Dimension of the model

- `heads`: int.  
Number of heads in Attention layer

- `ff_mult`: int.
Multiplier for the feedforward layer 

- `act`: torch.functional.  
Activation function in layers.

- `patch_size`: torch.functional.  
Size of patches. image_size must be divisible by patch_size.


- `image_size`: int.  
Size of images.

- `in_channels`: int.  
Number of image's channels.


- `n_layer`: int.  
Number of blocks in model.

- `pool`: list.  
Index for pooling layers.


## Usage

```python
import torch
import torch.functional as F

from cvit import CViT


model = CViT(dim=768, n_head=12, ff_mult=4, act=F.gelu, patch_size=16, img_size=256, in_channels=3, n_layers=8, pool=[3, 5]))
image = torch.randn(1, 3, 256, 256)
out = model(image)

```
