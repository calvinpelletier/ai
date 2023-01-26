from ai.train.hook import Hook
from ai.util import Timer


class Trainer:
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def train(s,
        model,
        opt,
        hook=Hook(),
        step=0,
        timelimit=None,
        steplimit=None,
    ):
        timer = Timer(timelimit)
        steplimit = math.inf if steplimit is None else step + steplimit

        for batch in s._train_data:
            # prep model
            # (inside loop b/c hook might switch model to eval mode)
            model.train()

            # execute step
            opt.zero_grad()
            loss = s._env(model, batch, step)
            loss.backward()
            opt.step()

            # call hook, possibly stop training
            stop = hook.step(step, loss)
            step += 1
            if stop or timer() or (step >= steplimit):
                break

        return step


class MultiTrainer:
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def train(s,
        models,
        opts,
        hook=Hook(),
        step=0,
        timelimit=None,
        steplimit=None,
    ):
        timer = Timer(timelimit)
        steplimit = math.inf if steplimit is None else step + steplimit

        keys = models.keys()
        for batch in s._train_data:
            # prep models
            # (inside loop b/c hook might switch model to eval mode)
            for model in models.values():
                model.train()
                model.set_req_grad(False)

            # execute one step per model
            losses = {}
            for key in keys:
                model = models[key]
                opt = opts[key]

                model.set_req_grad(True)
                opt.zero_grad()
                loss = getattr(s._env, key)(models, batch, step)
                loss.backward()
                opt.step()
                model.set_req_grad(False)

                losses[key] = loss

            # call hook, possibly stop training
            stop = hook.step(step, loss)
            step += 1
            if stop or timer() or (step >= steplimit):
                break

        return step
