# [WIP] Unnamed AI Library

A PyTorch-based machine learning library designed for quick and easy experimentation with a variety of ML concepts.

Full API reference coming soon.

## Examples

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
    model.init().to(device), # model
    ai.opt.Adam(model, lr=1e-3), # optimizer
    trial.hook(), # training hook
    timelimit=10,
)
```

GAN example: [StyleGAN2](examples/stylegan2/main.py)

Reinforcement learning example: [AlphaZero](examples/alphazero/main.py)

More examples: [ai/examples](examples)

For a WIP example of using `ai` for a large project, see [ai/examples/chess](examples/chess) (I'm currently investigating the ideal architecture for predicting the moves/outcomes of human chess matches).

## Install

`pip` package coming soon. In the meantime:

Ubuntu:
```bash
git clone https://github.com/calvinpelletier/ai.git

sudo apt update
sudo apt install python3.9-venv
python3.9 -m venv ./ai/.venv
source ./ai/.venv/bin/activate
pip install -r ai/requirements.txt

export PYTHONPATH="$(pwd):$PYTHONPATH"
export AI_DATASETS_PATH="/path/where/datasets/will/be/stored"
python ai/examples/mnist/main.py /tmp/mnist --device=cpu
```

## Table of Contents

- [Model](#model)
- [Train](#train)
- [Data](#data)
- [Lab](#lab)
- [Infer](#infer)
- [Game](#game)
- [Task](#task)
- [Util](#util)

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

Optimizers created via `ai.train.opt` (or simply `ai.opt`) are essentially just torch optimizers plus optional gradient clipping. There's also some QoL stuff like `ai.opt.build` which creates an optimizer from a `Config` object.

### Loss functions

`ai.train.loss` (or `ai.loss`) is still in the early phase of development but it has a few useful loss functions like `ai.loss.PerceptualLoss` for LPIPS or traditional perceptual loss, and `ai.loss.ComboLoss` for doing weighted sums of multiple losses.

# Data

### Dataset

`ai.data.Dataset` (and its subclasses) is a representation of a dataset that can be held in memory all at once. Calling the `iterator` method launches one or more data workers (which begin loading/generating and preprocessing data) and returns a `DataIterator`. Iterating over the `DataIterator` fetches batches of data from the worker(s), transfers them to the appropriate device, and runs postprocessing.

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

NOTE: I recommend using `None` for the number of workers for now. Torch data loaders max out at 12 workers which isn't enough to justify the added latency of using a remote inferencer. I implemented a custom lightweight version of it but still maxed out around 20 workers on my laptop. I'm working on a solution where each worker process is also multi-threaded (GIL shouldn't be an issue since the workers are mainly i/o bound with the inferencer calls).

# Lab

`ai.lab` is for running ML experiments and storing/examining all the resulting information.

There are currently 3 lab objects:

1) `ai.lab.Trial` (or `ai.Trial`): Trials are the smallest lab unit. A Trial is essentially a single training run. It can log metrics during training like training/validation losses and task evaluations. It can also save/load snapshots (model and opt state dicts), sample outputs from the model, etc.

2) `ai.lab.Experiment` (or `ai.Experiment`): Experiments are collections of trials. For example, you could create an experiment with the goal of maximizing the classification accuracy on some validation dataset. Then manually run trials or run an automatic hyperparameter search, and examine how various hyperparameters affect the results. 

3) `ai.lab.Study` (or `ai.Study`): Studies are basically directories. They are open-ended collections of trials, experiments, and anything else you might like to save to the disk.

The first argument of any lab object's constructor is a path to where information should be stored on disk. It can either be an exact path or a path relative to the $AI_LAB_PATH environment variable (inferred from whether there's a leading '/'). All lab objects also have a boolean keyword argument `clean` (default: false), which will delete and recreate the path if true. 

Note: `ai.lab` is still pretty barebones at the moment. I haven't decided yet whether it should act as an interface with other existing solutions like neptune/mlflow or if it should compete with them.

### Example

`ai.lab` is best explained through an example. Consider the hypothetical: you're interested in the difference between various methods for measuring image similarity. Let's create a study for this.

```python
import ai

study = ai.Study('imsim')
print(study.path) # $AI_LAB_PATH/imsim
```

As an initial investigation, you might come up with the idea to train an image autoencoder to recreate images of faces to see what happens when you optimize using different loss functions: pixel distance, distance in the intermediate features of a general-purpose image model, and distance in the output embedding of a specialized model. Let's setup the dataset, model, loss functions, and trainer.

```python
device = 'cuda'
imsize = 64

val_ds, train_ds = ai.data.ImgDataset('ffhq', imsize).split(.01, .99)
val_iter = val_ds.iterator(128, device, train=False)

from ai.model.ae import ImgAutoencoder
model = ImgAutoencoder(imsize, 4, 16, 256).to(device)

losses = [
    ('pixel', ai.loss.L2Loss()),
    ('percep', ai.loss.PerceptualLoss()),
    ('face_id', ai.loss.FaceIdentityLoss()),
]

def run(loss_fn, hook, batch_size, lr, grad_clip, steplimit=5000):
    trainer = ai.Trainer(
        ai.train.Reconstruct(loss_fn), # training environment
        train_ds.iterator(batch_size, device, train=True), # training data
    )
    trainer.train(
        model.init(),
        ai.opt.AdamW(model, lr=lr, grad_clip=grad_clip),
        hook,
        steplimit=steplimit,
    )
    return trainer.validate(model, val_iter)
```

The first step would be to run a hyperparameter search for each loss function.

```python
for loss_name, loss_fn in losses:
    # inside our study, create an experiment for a hyperparameter search (hps)
    exp = study.experiment(f'hps/{loss_name}', val_data=val_iter)
    print(exp.path) # $AI_LAB_PATH/imsim/hps/<loss_name>

    # run 16 trials using the run function we defined earlier
    exp.run(16, lambda trial: run(
        loss_fn,

        # trial.hook handles validation and early stopping to prune unpromising
        # trials
        trial.hook(),

        # trial.hp both specifies the searchable hyperparameter space for the
        # whole experiment and selects the exact hyperparameters for this
        # specific trial.
        trial.hp.lin('batch_size', 8, 64, step=8), # linear
        trial.hp.log('learning_rate', 1e-4, 1e-2), # logarithmic
        trial.hp.lst('grad_clip', [False, True]), # list
    ))
    print(loss_name, exp.best_hparams)
```

Next, we would use the best hyperparameters to run a full training session for each loss function.

```python
# load from samples from the val set for inspecting the model during training
samples = val_ds.sample(8, device)

for loss_name, loss_fn in losses:
    # inside our study, create a trial for the main training run of this loss
    trial = study.trial(
        loss_name,
        clean=True, # delete this trial if it already exists
        save_snapshots=True, # regularly save the model and optimizer
        val_data=val_iter, # regularly run validation

        # save side-by-side comparisons of sample inputs and their resulting
        # outputs at regular intervals during training
        sampler=lambda path, step, model: ai.util.save_img_grid(
            path / f'{step}.png',
            [samples, model(samples)],
        ),
    )
    print(trial.path) # $AI_LAB_PATH/imsim/<loss_name>

    # get the best hyperparameters from the search
    hp = study.experiment(f'hps/{loss_name}').best_hparams

    # run training
    run(
        loss_fn,
        trial.hook(),
        hp['batch_size'],
        hp['learning_rate'],
        hp['grad_clip'],
        steplimit=10_000,
    )
```

Finally, we could compare the results by creating an image grid of the models' outputs.

```python
model.eval()
comparison = [samples]
for loss_name, _ in losses:
    model.init(study.trial(loss_name).model_path()) # load params from disk
    comparison.append(model(samples))
ai.util.save_img_grid(study.path / 'comparison.png', comparison)
```

See [ai/examples/imsim](examples/imsim/main.py).

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

# Task

TODO

# Util

TODO
