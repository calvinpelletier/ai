# [WIP] Unnamed AI Library

A PyTorch-based machine learning library designed for quick and easy experimentation with a variety of ML concepts.

Full API reference coming soon.

MNIST classification example:

```python
import ai
import ai.model as m

outpath = '/tmp/mnist'
device = 'cpu'
batch_size = 64

# simplest mnist model
model = ai.Model(m.seq(
    m.flatten(), # flatten img
    m.fc(28 * 28, 10), # fully-connected layer
))

# dataset
train_ds, val_ds = ai.data.mnist().split()

# logging, validation, etc.
trial = ai.Trial(outpath, val_data=val_ds.iterator(batch_size, device))

ai.Trainer(
    ai.train.Classify(), # training environment
    train_ds.iterator(batch_size, device, train=True), # training data
).train(
    model.init().train().to(device), # model
    ai.opt.adam(model, lr=1e-3), # optimizer
    trial.hook(), # training hook
    timelimit=10,
)
```

GAN example: [Stylegan2](examples/stylegan2/main.py)

Reinforcement learning example: [AlphaZero](examples/alphazero/main.py)

More examples: [ai/examples](examples)

# Train

This is the simplest way to train a model:
```python
import ai

ai.Trainer(env, data).train(model, opt)
```
Where `env` is a callable that calculates the loss, `data` is an iterable that produces batches of training data, and `opt` is the optimizer.

There are five main parts to the `ai.train` module: trainers, environments, hooks, optimizers, and loss functions.

### Trainers

Trainers loop over a data iterator, call the environment, backprop the loss, and step the optimizer. They have two methods: `.train()` and `.validate()`. There are currently 2 trainers:

1) `Trainer`
2) `MultiTrainer` for training multiple models simultaneously as they interact with each other.

### Training Environments

A training environement is a callable that takes 3 arguments (the model, a batch of data, and the current step number) and returns a loss value. For multi-training, it takes 4 arguments (the current phase, a dict of models, a batch of data, and the step number). See [ai/train/env/diffusion.py](train/env/diffusion.py) for an example of an environment, and [ai/train/env/gan.py](train/env/gan.py) for an example of a multi-training environment.

### Training Hooks

Training hooks handle everything not directly responsible for model training (logging, validation, saving snapshots, saving samples, running evaluation tasks, checking for early stopping, etc.). The trainer calls the hook at the beginning of every training step. The simplest way to use them is to create one from a `Trial` object (discussed in the "Lab" section), or you can implement your own by extending `ai.train.HookInterface`. See [ai/train/hook.py](train/hook.py) for more info.

### Optimizers

Optimizers created via `ai.train.opt` (or simply `ai.opt`) are essentially just pytorch optimizers plus optional gradient clipping. There's also some QoL stuff like `ai.opt.build` which creates an optimizer from a `Config` object.

### Loss functions

`ai.train.loss` (or `ai.loss`) is still in the early phase of development but it has a few useful loss functions like `ai.loss.PerceptualLoss` for LPIPS or traditional perceptual loss, and `ai.loss.ComboLoss` for doing weighted sums of multiple losses.

# Model

The `ai.model` module contains various functions/classes for creating PyTorch models.

```python
import ai.model as m

m.fc(8, 8) # fully-connected layer (linear transformation)
m.fc(8, 8, actv='relu') # linear transformation followed by non-linearity
m.fc(8, 8, scale_w=True) # scale the weights' learning rate by 1/sqrt(input_dim)

m.conv(8, 8, k=5, stride=2, actv='relu', norm='batch') # convolution
m.conv(8, 8, stride=0.5) # equivalent to transposed conv with stride=2
m.modconv(8, 8, 4) # a convolution which will be modulated by a vector of size 4

# sequence
m.seq(
    m.fc(8, 8, actv='relu'),
    m.fc(8, 8, actv='relu'),
)
# or simply:
m.repeat(2, m.fc(8, 8, actv='relu'))

m.res(m.fc(8, 8)) # residual around a fully-connected layer

# resnet block
m.res(
    # main
    m.seq(
        m.conv(4, 8, stride=2, norm='batch', actv='mish'),
        m.conv(8, 8, norm='batch'),
        m.se(8), # squeeze-excite (self-modulate using global information)
    ),
    # shortcut
    m.conv(4, 8, k=1, stride=2), # (stride is done via avg pool because k==1)
)
# or simply:
m.resblk(4, 8, stride=2)

# image pyramid (shrink from 32x32 to 4x4 and deepen from 8 to 64 channels)
m.pyramid(32, 4, 8, 64, lambda: _, a, b: m.resblk(a, b, stride=2))

# transformer encoder block
m.seq(
    m.res(m.sa(4, 2)), # self-attention
    m.res(m.seq(m.fc(4, 8, actv='gelu'), m.fc(8, 4))), # feed-forward block
)
# or simply:
m.tx_enc_blk(4, 2, 8)
```

There are 3 parts to `ai.model`:

1) `ai.model.Model` (and subclasses like `DiffusionModel`). This class is what interacts with the rest of the `ai` ecosystem. 

2) `ai.model.f.*` which contains functional operations that act directly on tensors

3) Everything else is a function that returns a torch module. This avoids the need to remember which modules are classes and which are built via functions. The core building blocks are `fc` for fully-connected layers, `conv` for convolutions, and `attn`/`sa` for attention/self-attention. See [ai/model/linear.py](model/linear.py), [ai/model/conv2d.py](model/conv2d.py), and [ai/model/attention.py](model/attention.py) for more details.

Here are two functionally identical ways to use the `Model` class:

```python
import ai.model as m

# 1: as a wrapper around your top-level module 
# (by passing it to the constructor)
class MyModel(m.Model):
    def __init__(self):
        super().__init__(m.seq(
            m.flatten(),
            m.fc(8, 8),
        ))
model = MyModel()
# or simply:
model = m.Model(m.seq(m.flatten(), m.fc(8, 8)))

# 2: as the top-level module itself
# (by implementing 'forward')
class MyModel(m.Model):
    def __init__(self):
        super().__init__()
        self.net = m.fc(8, 8)
    def forward(self, x):
        x = m.f.flatten(x) # note the 'f' (functional)
        return self.net(x)
model = MyModel()
```

After building the model, you'll need to initialize the parameters:

```python
model.init() # randomly
# or
model.init('/path/to/model/weights.pt') # from disk
```

### Model Examples

MNIST MLP (and the pure PyTorch equivalent on the right):

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

RL example: [MuZero MLP](examples/muzero/model.py)

Diffusion example: [Diffusion MLP](examples/diffusion/model.py)

# Data

### Dataset

`ai.data.Dataset` (and its subclasses) is a representation of a dataset that can be held in memory all at once. Calling the `iterator` method launches one or more data workers (which begin loading/generating and preprocessing data) and returns a `DataIterator`. Iterating over the `DataLoader` fetches batches of data from the worker(s), transfers them to the appropriate device, a runs postprocessing.

MNIST example:

```python
import ai

batch_size = 64
device = 'cuda'

# load (download first if needed) the MNIST dataset from $AI_DATASETS_PATH/mnist
ds = ai.data.mnist()
# or alternatively, provide a path
ds = ai.data.mnist('/tmp/mnist')

# split into a train set and a validation set
train_ds, val_ds = ds.split() # standard split
train_ds, val_ds = ds.split(.9, .1) # custom split

# check length
print(len(train_ds)) # 63000
print(train_ds.length(batch_size)) # 62976 (the length accounting for dropping
                                   # the last incomplete batch)

# load and examine 100 samples
samples = val_ds.sample(100, device)
ai.util.print_info(samples['x']) # shape=[100,1,28,28] bounds=[-1.00,1.00]
                                 # dtype=float32 device=cuda:0
ai.util.print_info(samples['y']) # shape=[100] bounds=[0,9] dtype=uint8
                                 # device=cuda:0

# train iterator (shuffles and loops infinitely)
train_iter = train_ds.iterator(batch_size, device, train=True)
# val iterator (doesnt shuffle and loops for one epoch)
val_iter = val_ds.iterator(batch_size, device, train=False)

# iterate
for batch in val_iter:
    pass
```

Dataset classes:
- `ai.data.Dataset`
- `ai.data.ImgDataset`

Built-in datasets:
- `ai.data.mnist`
- `ai.data.cifar10`
- `ai.data.toy.moons`

`ai.data.Dataset` takes 4 arguments:

1) data: the data held in memory (e.g. a list of image files).
2) (optional) preprocess: fn called by the data workers (e.g. load image from disk).
3) (optional) postprocess: fn called on batches after transfering to the device (e.g. convert image to float32 and normalize).
4) (optional) default_split: the default way to split the dataset when `.split()` is called without arguments.

In more complex cases, you can bypass this and get a `DataIterator` object by directly using `ai.data.util.create_data_iterator`, which takes a `torch.utils.data.Dataset` object as its main argument.

### Reinforcement learning data

When the data is being generated by the model being trained (i.e reinforcement learning), extend the `ai.data.RLDataIterator` class which is an iterable with a `model_update_interval` attribute and a `model_update` method for receiving regular updates to the model's parameters.

For example, "self play" where a model plays games against itself to generate "replays" which are then used to train the model. Upon creation of an `ai.data.SelfPlay` object, an inferencer is launched for the model and data workers are spawned which play games by calling the inferencer. The resulting replays are stored in a replay buffer from which the trainer pulls batches of data. The trainer then periodically sends fresh parameters to the inferencer. See [AlphaZero](examples/alphazero/main.py).

```python
import ai
from ai.examples.alphazero import AlphaZeroMLP

game = ai.game.TicTacToe()

data = ai.data.SelfPlay(
    # game and player
    game,
    ai.game.MctsPlayer(game, AlphaZeroMLP(game).init()),

    # external configuration
    32, # batch size
    'cuda:0', # device
    100, # ask trainer for a model update every N steps

    # internal configuration
    8, # number of data workers
    256, # maximum inference batch size
    'cuda:1', # inference device
    128, # size of the intermediate buffer
    4, # number of replay times (how many times a state can be
       # fetched, once all the states in a replay hit this threshold, it is 
       # ejected from the buffer and replace by a new one)
)
```

NOTE: I recommend using `None` for the number of workers for now. Torch data loaders max out at 12 workers which isn't enough to justify the added latency of using a remote inferencer. I implemented a custom lightweight version of it but still maxed out around 20 workers. I'm working on a solution where each worker process is also multi-threaded (GIL shouldn't be an issue since the workers are mainly i/o bound with the inferencer calls).

# Game

Games
- `ai.game.Chess`
- `ai.game.TicTacToe`
- `ai.game.CartPole`
- `ai.game.ToyGame`
- `ai.game.Connect2`
- `ai.game.Chess1d`

Players
- `ai.game.RandomPlayer`
- `ai.game.MctsPlayer`

Algorithms
- `ai.game.MonteCarloTreeSearch`

### MCTS

`ai.game.MonteCarloTreeSearch` supports both modeled (e.g. MuZero) and model-free (e.g. AlphaZero) reinforcement learning. It takes 3 arguments: an `ai.game.MctsConfig` config object, a callable player, and optional value bounds for normalization (if not given, it will figure it out during play). To get a policy, pass a game object to the `run` method.

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

