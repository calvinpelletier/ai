import torch
from time import time
from typing import Optional, Callable

from ai.train.schedule import Schedule
from ai.util import print_header
from ai.train.opt import OptLike
from ai.model import Model


class HookInterface:
    '''Training hook inferface.'''

    def setup(s,
        model: Model | dict[str, Model],
        opt: OptLike | dict[str, OptLike],
        validate: Callable,
    ):
        '''Called by the trainer at the start of training.

        ARGS
            model
                the model (or dict of models in the case of multi training)
            opt
                the optimizer (or dict of opts in the case of multi training)
            validate
                a function that can be used for validation (Trainer.validate)
        '''

        raise NotImplementedError()

    def step(s, step: int) -> bool:
        '''Called by the trainer before every train step.

        ARGS
            step
                the current step count
        RETURNS
            whether to early stop or not
        '''

        raise NotImplementedError()

    def done(s) -> Optional[torch.Tensor | float]:
        '''Called by the trainer after training stops.

        RETURNS
            optionally returns the final validation loss
        '''

        raise NotImplementedError()

    def train_log(s, k: str, v: int | float | torch.Tensor):
        '''Log some intermediate key-value pair from training.

        NOTE: this function is given to the singleton at ai/train/log.py, thus
        allowing anybody to call it by simply importing it (e.g. the trainer,
        the training environment, the loss function, etc.)

        ARGS
            k
                key
            v
                value
        '''

        raise NotImplementedError()


# TODO: rework
class Hook(HookInterface):
    '''A simple implementation of the training hook interface.

    Consists of 4 subhooks: log, save, val, and sample.

    Each subhook is enabled by passing a dict to the constructor. For example:
    ```
    Hook(log={
        'fn': ai.util.logger.Tensorboard(log_path),
        'interval': 1000,
    })
    ```

    ARGS
        log
            Used for training and validation logging.
                Training:
                    key prefix: 'train.' (e.g. 'train.loss')
                    only every n steps (where n is the log interval)
                Validation:
                    key prefix: 'val.'
            Keys:
                'fn': fn(key, value)
                'interval': int
        save
            Save model/optimizer snapshots.
            Keys:
                'fn': fn(step, model, opt)
                'interval': int
        val
            Model validation.
            Keys:
                'data': validation data
                'interval': int
                (optional) 'stopper': stopper(val_loss) -> should_stop
            NOTE: the validation function is received from the trainer when
            setup is called.
        sample
            Save samples from the model for inspection.
            Keys:
                'fn': fn(step, model)
                'interval': int
        print_
            whether to print info or not
    '''

    def __init__(s,
        log: Optional[dict] = None,
        save: Optional[dict] = None,
        val: Optional[dict] = None,
        sample: Optional[dict] = None,
        print_: bool = True,
    ):
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

    def setup(s,
        model: Model | dict[str, Model],
        opt: OptLike | dict[str, OptLike],
        validate: Callable,
    ):
        s._model = model
        s._opt = opt
        if s._val is not None:
            s._val['fn'] = validate

    def step(s, step: int):
        s._step = step
        s._check_log_step()
        stop = s._run_subhooks()
        s._model.train()
        return stop

    def done(s) -> Optional[torch.Tensor | float]:
        if s._print:
            print_header(f'DONE (step={s._step})')
        val_loss = s._run_subhooks(True)
        if s._print:
            print_header('')
        return val_loss

    def train_log(s, k: str, v: int | float | torch.Tensor):
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


class NullHook(HookInterface):
    '''Hook that does nothing.'''

    def setup(s, model, opt, validate):
        pass

    def step(s, step):
        pass

    def done(s):
        pass

    def train_log(s, k, v):
        pass
