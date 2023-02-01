import torch
from shutil import rmtree

from ai.train.log import Tensorboard
from ai.path import lab as lab_path
from ai.train.hook import Hook


class Trial:
    def __init__(s, path, clean=False, log=Tensorboard):
        s.path = lab_path(path)
        if clean and s.path.exists():
            rmtree(s.path)
        s.path.mkdir(parents=True, exist_ok=True)

        s.log = log(s.path / 'log')

    def hook(s, snapshot=True):
        hook = Hook(s.log)
        if snapshot:
            hook.add(s.save_snapshot)
        return hook

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
