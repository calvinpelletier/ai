import math

from ai.train.hook import NullHook
from ai.util import Timer


class _Base:
    def _setup(s, hook, step, timelimit, steplimit):
        if hook is None:
            hook = NullHook()
        s._env.log = hook.log
        timer = Timer(timelimit)
        steplimit = math.inf if steplimit is None else step + steplimit
        return hook, timer, steplimit

    def _post_step(s, step, model, opt, hook):
        return hook.post_step()


class Trainer(_Base):
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def train(s,
        model,
        opt,
        hook=None,
        step=0,
        timelimit=None,
        steplimit=None,
    ):
        hook, timer, steplimit = s._setup(hook, step, timelimit, steplimit)

        for batch in s._train_data:
            # pre step
            hook.pre_step(step, model, opt)
            model.train() # inside loop b/c hook might switch model to eval mode

            # step
            opt.zero_grad()
            loss = s._env(model, batch, step)
            loss.backward()
            opt.step()

            # post step
            hook.log('loss', loss)
            stop = s._post_step(step, model, opt, hook)
            step += 1
            if stop or timer() or (step >= steplimit):
                break

        hook.done()
        return step

    def validate(s):
        pass


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
                loss = getattr(s._env, key)(models, batch, step)
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
