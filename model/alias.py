import torch
from einops.layers.torch import Rearrange


embed = torch.nn.Embedding
lstm = torch.nn.LSTM
modules = torch.nn.ModuleList
rearrange = Rearrange
