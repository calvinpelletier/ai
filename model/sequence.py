from torch import nn
from copy import deepcopy


def seq(*a):
    return nn.Sequential(*a)


def repeat(n, obj):
    assert n >= 0
    if n == 0:
        return nn.Identity()
    if n == 1:
        return obj

    objs = [obj]
    while len(objs) < n:
        objs.append(deepcopy(obj))

    return nn.Sequential(*objs)
