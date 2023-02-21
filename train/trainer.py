import torch
import math
from typing import Optional, Callable, Iterable, Any, Union, Dict

from ai.train.hook import HookInterface, NullHook
from ai.util import Timer, no_op
from ai.train.log import log as train_log
from ai.model import Model
from ai.train.opt import OptLike
from ai.train.util import on_interval


class _Base:
    '''Base class shared by Trainer and MultiTrainer.'''

    def __init__(s, env: Callable, data: Iterable):
        '''
        ARGS
            env
                The training environment (see ai/train/env.py).
            data
                The training data iterable.
        '''

        s._env = env
        s._train_data = data

    def train(s,
        model: Union[Model, Dict[str, Model]],
        opt: Union[OptLike, Dict[str, OptLike]],
        hook: Optional[HookInterface] = None,
        step: int = 0,
        timelimit: Optional[int] = None,
        steplimit: Optional[int] = None,
    ):
        '''Run training.

        ARGS
            model
                The model (or dict of models in the case of multi training).
            opt
                The optimizer (or dict of opts in the case of multi training).
            hook
                An optional training hook for logging, saving, etc.
                (see ai/train/hook.py)
            step
                The current step (in case training is being resumed).
            timelimit
                An optional limit of the training time in seconds.
            steplimit
                An optional limit of the number of training steps.
        RETURNS
            whatever hook.done() returns
        '''

        # use a null hook if a hook isnt provided
        if hook is None:
            hook = NullHook()

        # set everything up
        hook.setup(model, opt, s.validate)
        train_log.setup(hook.train_log)
        timer = Timer(timelimit)
        steplimit = math.inf if steplimit is None else step + steplimit

        # run training
        return s._train(model, opt, hook, step, timer, steplimit)

    def validate(s,
        model: Union[Model, Dict[str, Model]],
        data: Iterable,
        log: Callable[[str, Any], None] = no_op,
    ) -> torch.Tensor:
        '''Run validation.

        ARGS
            model
                The model (or dict of models in the case of multi training).
            data
                The validation data.
            log
                An optional logging function.
        RETURNS
            the average validation loss
        '''

        # switch model to eval mode
        if isinstance(dict, model):
            for x in model.values():
                x.eval()
        else:
            model.eval()

        # run validation
        with torch.no_grad():
            ret = s._validate(model, data, log)
        return ret

    def _train(s, model, opt, hook, step, timer, steplimit):
        raise NotImplementedError()

    def _validate(s, model, data, log):
        raise NotImplementedError()


class Trainer(_Base):
    '''The standard trainer class.'''

    def __init__(s, env: Callable, data: Iterable):
        '''
        ARGS
            env
                The training environment (see ai/train/env.py).
            data
                The training data iterable.
        '''

        super().__init__(env, data)

        # for reinforcement learning
        s._data_wants_model_updates = hasattr(
            s._train_data, 'model_update_interval')

    def _train(s, model, opt, hook, step, timer, steplimit):
        model.train()
        for batch in s._train_data:
            # call hook and possibly stop training
            early_stop = hook.step(step)
            if early_stop or timer() or (step >= steplimit):
                break

            # execute single training step
            opt.zero_grad()
            loss = s._env(model, batch, step)
            loss.backward()
            opt.step()

            train_log('loss', loss)
            s._post_step(step, model)
            step += 1

        return hook.done()

    def _validate(s, model, data, log):
        loss = 0.
        n = 0
        for batch in data:
            loss += s._env(model, batch).item()
            n += 1
        loss /= n
        log('loss', loss)
        return loss

    def _post_step(s, step, model):
        if (step > 0 and
            s._data_wants_model_updates and
            on_interval(step, s._train_data.model_update_interval)
        ):
            s._train_data.model_update(model.state_dict())


class MultiTrainer(_Base):
    '''Multi-model trainer.

    For training multiple models simultaneously as they interact with each
    other (e.g. GANs).
    '''

    def _train(s, models, opts, hook, step, timer, steplimit):
        for model in models.values():
            model.train()
            model.set_req_grad(False)

        for batch in s._train_data:
            # call hook and possibly stop training
            early_stop = hook.step(step)
            if early_stop or timer() or (step >= steplimit):
                break

            # step each model ~
            for key in models.keys():
                model = models[key]
                opt = opts[key]

                model.set_req_grad(True)
                opt.zero_grad()
                loss = s._env(key, models, batch, step)
                loss.backward()
                opt.step()
                model.set_req_grad(False)

                train_log(f'{key}.loss', loss)
            # ~

            step += 1

        return hook.done()
