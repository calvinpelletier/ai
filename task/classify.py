import torch
from typing import Callable, Iterable

from ai.task.base import Task
from ai.util import no_op
from ai.model import Model


class Classify(Task):
    def __init__(s, data: Iterable):
        s._data = data

    def __call__(s, model: Model, log: Callable = no_op):
        correct = 0
        total = 0
        for batch in s._data:
            x, y = batch['x'], batch['y']
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += x.shape[0]
        assert total > 0, 'empty data in classify task'
        acc = 100. * correct / total
        log('accuracy', acc)
        return acc
