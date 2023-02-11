import torch
import math

from ai.train.hook import NullHook
from ai.util import Timer, no_op
from ai.opt import Opt


class _Base:
    def train(s,
        model,
        opt,
        hook=None,
        step=0,
        timelimit=None,
        steplimit=None,
    ):
        if hook is None:
            hook = NullHook()
        hook.setup(model, opt, s.validate)
        log = hook.train_log

        timer = Timer(timelimit)

        steplimit = math.inf if steplimit is None else step + steplimit

        return s._train(model, opt, hook, log, step, timer, steplimit)

    def _train(s, model, opt, hook, log, step, timer, steplimit):
        raise NotImplementedError()

    def validate(s, model, data, log=no_op):
        with torch.no_grad():
            ret = s._validate(model, data, log)
        return ret

    def _validate(s, model, data, log):
        raise NotImplementedError()


class Trainer(_Base):
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def _train(s, model, opt, hook, log, step, timer, steplimit):
        for batch in s._train_data:
            early_stop = hook.step(step)
            if early_stop or timer() or (step >= steplimit):
                break

            opt.zero_grad()
            loss = s._env(model, batch, step, log)
            loss.backward()
            opt.step()

            log('loss', loss)
            step += 1

        return hook.done()

    def _validate(s, model, data, log):
        model.eval()
        loss = 0.
        n = 0
        for batch in data:
            loss += s._env(model, batch).item()
            n += 1
        loss /= n
        log('loss', loss)
        return loss


class MultiTrainer(_Base):
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def train(s,
        models,
        opts,
        hook=None,
        step=0,
        timelimit=None,
        steplimit=None,
    ):
        hook, timer, steplimit = s._setup(hook, step, timelimit, steplimit)

        keys = models.keys()
        for batch in s._train_data:
            # pre step
            hook.pre_step(step, models, opts)
            for model in models.values():
                model.train()
                model.set_req_grad(False)

            # step each model ~
            for key in keys:
                model = models[key]
                opt = opts[key]
                model.set_req_grad(True)

                opt.zero_grad()
                loss = s._env(key, models, batch, step, log)
                loss.backward()
                opt.step()

                model.set_req_grad(False)
                hook.log(f'loss.{key}', loss)
            # ~

            # post step
            stop = s._post_step(step, model, opt, hook)
            step += 1
            if stop or timer() or (step >= steplimit):
                break

        hook.done()
        return step
