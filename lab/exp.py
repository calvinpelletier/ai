import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from functools import partial
import matplotlib.pyplot as plt

from ai.lab.base import LabEntity
from ai.lab.trial import Trial
from ai.util import print_header


class Experiment(LabEntity):
    def __init__(s, path, clean=False, direction='min', prune=False, **trial_kw):
        super().__init__(path, clean)
        assert direction in ['min', 'max']
        if prune:
            assert 'val_data' in trial_kw or 'task' in trial_kw, (
                'Need val_data or task in trial kwargs if prune==True')
            assert 'task' not in trial_kw, 'TODO: task-based early stopping'
        s._trial_kw = trial_kw

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        s._exp = optuna.create_study(
            study_name=str(s.path),
            storage=f'sqlite:///{s.path}/experiment.db',
            load_if_exists=True,
            direction='minimize' if direction == 'min' else 'maximize',
            sampler=TPESampler(),
            pruner=SuccessiveHalvingPruner() if prune else None,
        )

    @property
    def best_hparams(s):
        return s._exp.best_params

    def run(s, n, fn):
        print(f'RUNNING EXPERIMENT {s.path} (n={n})\n')
        s._exp.optimize(partial(s._run, fn), n_trials=n)

    def show_plot(s, hparam, show_pruned=False, only_best=None):
        trials = []
        for t in s._exp.trials:
            if show_pruned or t.state == TrialState.COMPLETE:
                trials.append(t)
        if only_best is not None:
            idx = int(len(trials) * only_best)
            trials = sorted(trials, key=lambda t: t.value)[:idx]

        x, y = [], []
        for t in trials:
            x.append(t.params[hparam])
            y.append(t.value)
        plt.scatter(x, y)
        plt.show()

    def _run(s, fn, optuna_trial):
        id = str(optuna_trial.number)
        print_header(f'TRIAL {id}')
        trial = _ExpTrial(s.path / f'trials/{id}', optuna_trial, **s._trial_kw)
        result = fn(trial)
        print(f'RESULT: {result:.4f}\n')
        return result


class _ExpTrial(Trial):
    def __init__(s, path, optuna_trial, **kw):
        super().__init__(path, val_stopper=s.pruner, **kw)
        s._optuna_trial = optuna_trial
        s.hp = _HyperParams(optuna_trial)

    def pruner(s, step, val_loss):
        s._optuna_trial.report(val_loss, step)
        if s._optuna_trial.should_prune():
            print_header('')
            print('PRUNED\n')
            raise optuna.TrialPruned()
        return False


class _HyperParams:
    def __init__(s, optuna_trial, prefix=None):
        s._optuna = optuna_trial
        s._prefix = prefix

    def lin(s, name, min_, max_, step=1):
        name = s._prefix_name(name)

        has_float = False
        for x in [min_, max_, step]:
            if isinstance(x, float):
                has_float = True
                break

        if has_float:
            val = s._optuna.suggest_float(name, min_, max_, step=step)
        else:
            for x in [min_, max_, step]:
                assert isinstance(x, int)
            val = s._optuna.suggest_int(name, min_, max_, step=step)

        print(f'{name}: {val}')
        return val

    def log(s, name, min_, max_):
        name = s._prefix_name(name)

        if isinstance(min_, float) or isinstance(max_, float):
            val = s._optuna.suggest_float(name, min_, max_, log=True)
        else:
            assert isinstance(min_, int) and isinstance(max_, int)
            val = s._optuna.suggest_int(name, min_, max_, log=True)

        print(f'{name}: {val}')
        return val

    def lst(s, name, items):
        name = s._prefix_name(name)
        val = s._optuna.suggest_categorical(name, items)
        print(f'{name}: {val}')
        return val

    def _prefix_name(s, name):
        if s._prefix is None:
            return name
        return f'{s._prefix}.{name}'
