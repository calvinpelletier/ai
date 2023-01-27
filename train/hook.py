import torch

from ai.train.schedule import Exponential


class Hook:
    def __init__(s,
        log=None,
        log_schedule=Exponential(1, 1024),
        print_=True,
    ):
        s._log = log
        s._log_schedule = log_schedule
        s._print = print_
        s._step = None
        s._hooks = []

    def add(s, hook, schedule=Exponential(2**10, 2**16)):
        s._hooks.append((hook, schedule))

    def pre_step(s, step, model, opt):
        s._step = step

        s._is_log_step = s._log_schedule(step)
        if s._is_log_step and s._print:
            print('~'*16 + str(step) + '~'*16)

        for hook, sch in s._hooks:
            if sch(step):
                hook(step, model, opt)

    def log(s, k, v):
        if not s._log or not s._is_log_step:
            return

        if torch.is_tensor(v):
            v = v.item()

        if s._print:
            print(f'{k}: {v}')

        if s._log:
            s._log(s._step, k, v)

    def post_step(s):
        return False

    def done(s):
        pass


class NullHook:
    def pre_step(s, step, model, opt):
        pass

    def log(s, k, v):
        pass

    def post_step(s):
        return False

    def done(s):
        pass
