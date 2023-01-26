import torch
from torch.utils.tensorboard import SummaryWriter

from ai.train.schedule import Exponential


class Hook:
    def pre_step(s, step):
        pass

    def log(s, k, v):
        pass

    def post_step(s):
        return False

    def done(s):
        pass


class Tensorboard(Hook):
    def __init__(s, path, log_schedule=Exponential(1, 1024)):
        s._writer = SummaryWriter(path)
        s._log_schedule = log_schedule
        s._step = None

    def pre_step(s, step):
        s._step = step
        s._is_log_step = s._log_schedule(step)
        if s._is_log_step:
            print('~'*16 + str(step) + '~'*16)

    def log(s, k, v):
        if not s._is_log_step:
            return

        if torch.is_tensor(v):
            v = v.item()

        print(f'{k}: {v}')
        s._writer.add_scalar(k, v, s._step)

    def post_step(s):
        return False

    def done(s):
        pass
