import torch
from einops.layers.torch import Rearrange


embed = torch.nn.Embedding
lstm = torch.nn.LSTM
modules = torch.nn.ModuleList
module_dict = torch.nn.ModuleDict
null = torch.nn.Identity
rearrange = Rearrange
