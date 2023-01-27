import torch
from fire import Fire


def fire(x):
    Fire(x)


def no_grad():
    return torch.no_grad()
