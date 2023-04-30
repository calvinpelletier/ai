import torch
from functools import partial
from typing import Any, Optional, Iterable, Callable

from ai.util.logger import Tensorboard
from ai.train.hook import Hook
from ai.lab.base import LabEntity
from ai.util.schedule import Logarithmic, ScheduleConfig


class Trial(LabEntity):
    def __init__(s,
        path,
        clean: bool = False,

        logger: Any = Tensorboard,
        log_interval: ScheduleConfig = Logarithmic(1, 1024),

        save_snapshots: bool = False,
        save_interval: ScheduleConfig = Logarithmic(1024, 65536),

        val_data: Optional[Iterable] = None,
        val_interval: ScheduleConfig = Logarithmic(256, 8192),
        val_stopper: Optional[Callable] = None,

        sampler: Optional[Callable] = None,
        sample_interval: ScheduleConfig = Logarithmic(256, 65536),

        task: Optional[Callable] = None,
        task_interval: ScheduleConfig = Logarithmic(1024, 65536),
        task_stopper: Optional[Callable] = None,
    ):
        super().__init__(path, clean)

        s._hook_kwargs = {}
        if logger is not None:
            s._hook_kwargs['log'] = {
                'fn': logger(s.path / 'log'),
                'interval': log_interval,
            }
        if save_snapshots:
            s._hook_kwargs['save'] = {
                'fn': s.save_snapshot,
                'interval': save_interval,
            }
        if val_data is not None:
            s._hook_kwargs['val'] = {
                'data': val_data,
                'interval': val_interval,
                'stopper': val_stopper,
            }
        if sampler is not None:
            samples_path = s.path / 'samples'
            samples_path.mkdir(exist_ok=True)
            s._hook_kwargs['sample'] = {
                'fn': partial(sampler, samples_path),
                'interval': sample_interval,
            }
        if task is not None:
            s._hook_kwargs['task'] = {
                'fn': task,
                'interval': task_interval,
                'stopper': task_stopper,
            }

        # if part of an experiment
        s.hp = None

    def hook(s):
        return Hook(**s._hook_kwargs)

    def save_snapshot(s, step, model, opt):
        if step == 0:
            return

        path = s.path / 'snapshots/latest'
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'metadata.csv', 'w') as f:
            f.write(str(step) + '\n')

        if isinstance(model, dict):
            assert isinstance(opt, dict)
            models, opts = model, opt
            for k, model in models.items():
                torch.save(model.state_dict(), path / f'model_{k}.pt')
            for k, opt in opts.items():
                torch.save(opt.state_dict(), path / f'opt_{k}.pt')
        else:
            torch.save(model.state_dict(), path / 'model.pt')
            torch.save(opt.state_dict(), path / 'opt.pt')

        print('[INFO] saved snapshot')

    def load_snapshot(s, model, opt=None, snapshot='latest'):
        path = s.path / 'snapshots' / snapshot
        if isinstance(model, dict):
            models, opts = model, opt
            for k, model in models.items():
                model.init(path / f'model_{k}.pt')
            if opts is not None:
                for k, opt in opts.items():
                    opt.load_state_dict(torch.load(path / f'opt_{k}.pt'))
        else:
            model.init(path / 'model.pt')
            if opt is not None:
                opt.load_state_dict(torch.load(path / 'opt.pt'))

    def model_path(s, key=None, snapshot='latest'):
        fname = 'model' if key is None else f'model_{key}'
        return s.path / f'snapshots/{snapshot}/{fname}.pt'
