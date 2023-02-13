# [WIP] Unnamed AI Library

A PyTorch-based machine learning library designed to make it easy for independent researchers/enthusiasts to experiment with ML concepts.

# Model

The `ai.model` module contains various functions/classes for creating PyTorch models.

Here's an example of an MNIST fully-connected model (and the pure PyTorch equivalent on the right):

<table>
<tr>
<td> ai.model </td> <td> torch.nn </td>
</tr>
<tr>
<td>

```python
import ai.model as m
from math import prod

class FullyConnected(m.Model):
    def __init__(self, shape_in=[1, 28, 28], n_out=10, dim=128, n_layers=4):
        super().__init__(m.seq(
            m.flatten(),
            m.fc(prod(shape_in), dim, actv='relu'),
            m.repeat(n_layers, m.fc(dim, dim, actv='relu')),
            m.fc(dim, n_out),
        ))

model = FullyConnected().init() # randomly init params
# or
model = FullyConnected().init(some_path) # load params from disk
```

</td>
<td>

```python
import torch
from torch import nn
from math import prod

class FullyConnected(nn.Module):
    def __init__(self, shape_in=[1, 28, 28], n_out=10, dim=128, n_layers=4):
        super().__init__()
        layers = [nn.Linear(prod(shape_in), dim), nn.ReLU()]
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dim, n_out))
        self._net = nn.Sequential(*layers)

    def forward(self, x):
        return self._net(torch.flatten(x, 1))

model = FullyConnected()

model.apply(some_param_init_fn) # randomly init params
# or
model.load_state_dict(torch.load(some_path)) # load params from disk
```

</td>
</tr>
</table>

There are 3 parts to `ai.model`:

1) `ai.model.Model` (and subclasses like `DiffusionModel`). This class is what interacts with the rest of the `ai` ecosystem. It can either be a wrapper around your top-level module (by passing the module to `Model.__init__`) or it can be the top-level module itself (by implementing the `forward` method). 

2) `ai.model.f.*` which contains functional operations that act directly on tensors

3) Everything else is a function that returns a torch module. This avoids the need to remember which operations are classes and which are built via functions. And as a bonus, it has a clean, lowercase aesthetic.

Here are three functionally identical ways to use the `Model` class:

```python
import ai.model as m

# 1
class MyModel(m.Model):
    def __init__(self):
        super().__init__(m.seq(
            m.flatten(),
            m.fc(8, 8),
        ))
model = MyModel()

# 2
class MyModel(m.Model):
    def __init__(self):
        super().__init__()
        self.net = m.fc(8, 8)
    def forward(self, x):
        x = m.f.flatten(x)
        return self.net(x)
model = MyModel()

# 3
model = m.Model(m.seq(m.flatten(), m.fc(8, 8)))
```

### Model Examples

Image autoencoder using ResNet blocks:

```python
import ai.model as m

class ImgAutoencoder(m.Model):
    def __init__(self,
        imsize,
        bottleneck,
        nc_img=3,
        nc_min=32,
        nc_max=512,
        enc_block=lambda size, nc1, nc2: m.resblk(nc1, nc2, stride=2),
        dec_block=lambda size, nc1, nc2: m.resblk(nc1, nc2, stride=0.5),
    ):
        super().__init__()

        self.encode = m.seq(
            m.conv(nc_img, nc_min, actv='mish'),
            m.pyramid(imsize, bottleneck, nc_min, nc_max, enc_block),
        )

        self.decode = m.seq(
            m.pyramid(bottleneck, imsize, nc_min, nc_max, dec_block),
            m.conv(nc_min, nc_img, actv='tanh'),
        )

    def forward(self, x):
        return self.decode(self.encode(x))
```

GAN example: [StyleGAN2](examples/stylegan2/model.py)

Transformer example: [Vision Transformer](examples/vit/model.py)

RL examples: [AlphaZero](examples/alphazero/model.py) and [MuZero](examples/muzero/model.py)

Diffusion example: [Diffusion MLP](examples/diffusion/model.py)

### Model API

Currently, the two core building blocks are `fc` for fully-connected layers and `conv` for convolution layers. They are essentially `torch.nn.Linear` and `torch.nn.Conv2d` respectively, with optional additonal operations in sequence (e.g. activation function). Certain args switch it to a custom implementation of `Linear` and `Conv2d` (e.g. learning rate scaling).

Full API reference coming soon.

# Infer

The `ai.infer` module can be used to setup inference workers and clients.

```python
inferencer = ai.infer.Inferencer(model) # launch inference worker
y = inferencer(x) # call worker
del inferencer # stop worker
```

A more detailed example:

```python
import ai

# using an MNIST model as an example
model = ai.examples.mnist.Model().init()

# spawn a worker process
inferencer = ai.infer.Inferencer(
    model,
    'cuda', # the worker will move the model to this device
    64, # the maximum inference batch size (will be less if there arent
        # sufficient requests available at the moment)
)

# the inferencer can be used as if it is the model
x = torch.randn(1, 1, 28, 28)
y1 = model(x)
y2 = inferencer(x)
assert (y1 == y2).all()

# update the parameters of the worker's model
inferencer.update_params(model.state_dict())

# you can also create an InferencerClient which can make inference requests but
# doesn't hold a reference to the worker (useful when passing it to other
# processes e.g. when data workers need to make inference requests)
client = inferencer.create_client()
y = client(x)

# requests can be made asynchronously
request_id = client.infer_async(x)
y = client.wait_for_resp(request_id)

# you can stop the worker directly via
del inferencer
# or you can just let `inferencer` go out of scope
```

For more information, see [Inferencer](infer/inferencer.py) and [InferenceClient](infer/client.py).

