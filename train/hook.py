import torch
from time import time

from ai.train.schedule import Logarithmic, Schedule


class Hook:
    def __init__(s,
        log=None,
        log_schedule=Logarithmic(1, 1024),
        print_=True,
    ):
        s._log = log
        s._log_schedule = Schedule(log_schedule)
        s._print = print_
        s._step = None
        s._hooks = []

        # for calculating steps per sec
        s._last_log_step = None
        s._last_log_time = None

    def add(s, hook, schedule=Logarithmic()):
        s._hooks.append((hook, Schedule(schedule)))

    def pre_step(s, step, model, opt):
        s._step = step

        s._is_log_step = s._log_schedule(step)
        if s._is_log_step:
            sps = s._steps_per_sec()
            if s._print:
                print('~'*32 + f'step: {step} ({sps:.2f}/sec)' + '~'*32)

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

    def _steps_per_sec(s):
        ts = time()
        if s._last_log_step is None:
            steps_per_sec = 0.
        else:
            step_delta = s._step - s._last_log_step
            time_delta = ts - s._last_log_time
            steps_per_sec = step_delta / time_delta
            if s._log:
                s._log(s._step, 'steps_per_sec', steps_per_sec)
        s._last_log_step = s._step
        s._last_log_time = ts
        return steps_per_sec


class NullHook:
    def pre_step(s, step, model, opt):
        pass

    def log(s, k, v):
        pass

    def post_step(s):
        return False

    def done(s):
        pass
