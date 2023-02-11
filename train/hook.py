import torch
from time import time

from ai.train.schedule import Schedule
from ai.util import print_header


class Hook:
    def __init__(s, log=None, save=None, val=None, sample=None, print_=True):
        s._log = log
        s._save = save
        s._val = val
        s._sample = sample
        s._print = print_

        for subhook in [s._log, s._save, s._val, s._sample]:
            if subhook is not None:
                subhook['interval'] = Schedule(subhook['interval'])

        s._step = None
        s._is_log_step = False

        # set by trainer via s.setup()
        s._model = None
        s._opt = None
        if s._val is not None:
            s._val['fn'] = None

        # for calculating steps per sec
        s._last_log_step = None
        s._last_log_time = None

    def setup(s, model, opt, validate):
        s._model = model
        s._opt = opt
        if s._val is not None:
            s._val['fn'] = validate

    def step(s, step):
        s._step = step
        s._check_log_step()
        stop = s._run_subhooks()
        s._model.train()

    def done(s):
        if s._print:
            print_header(f'DONE (step={s._step})')
        val_loss = s._run_subhooks(True)
        if s._print:
            print_header('')
        return val_loss

    def train_log(s, k, v):
        if s._is_log_step:
            s.log('train.'+k, v)

    def val_log(s, k, v):
        s.log('val.'+k, v)

    def log(s, k, v):
        if torch.is_tensor(v):
            v = v.item()

        if s._print:
            if isinstance(v, float):
                print(f'{k}: {v:.4f}')
            else:
                print(f'{k}: {v}')

        if s._log is not None:
            s._log['fn'](s._step, k, v)

    def _run_subhooks(s, is_done=False):
        s._model.eval()
        stop = False
        val_loss = None
        with torch.no_grad():
            if s._save is not None and s._save['interval'](s._step):
                s._save['fn'](s._step, s._model, s._opt)

            if s._sample is not None and s._sample['interval'](s._step):
                s._sample['fn'](s._step, s._model)

            if s._val is not None:
                if is_done or s._val['interval'](s._step):
                    val_loss = s._val['fn'](s._model, s._val['data'], s.val_log)
                    if not is_done and s._val['stopper'] is not None:
                        if s._val['stopper'](s._step, val_loss):
                            stop = True

        if is_done:
            return val_loss
        return stop

    def _check_log_step(s):
        if s._log is None:
            return

        s._is_log_step = s._log['interval'](s._step)
        if s._is_log_step:
            sps = s._steps_per_sec()
            if s._print:
                print_header(f'step: {s._step} ({sps:.2f}/sec)')
            if s._log is not None:
                s._log['fn'](s._step, 'train.steps_per_sec', sps)

    def _steps_per_sec(s):
        ts = time()
        if s._last_log_step is None:
            steps_per_sec = 0.
        else:
            step_delta = s._step - s._last_log_step
            time_delta = ts - s._last_log_time
            steps_per_sec = step_delta / time_delta
        s._last_log_step = s._step
        s._last_log_time = ts
        return steps_per_sec


class NullHook:
    def setup(s):
        pass

    def pre_step(s, step, model, opt):
        pass

    def log(s, k, v):
        pass

    def post_step(s):
        return False

    def done(s):
        pass
