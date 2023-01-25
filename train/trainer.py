from ai.train.hook import Hook
from ai.util import Timer


class Trainer:
    def __init__(s, env, train_data, val_data=None):
        s._env = env
        s._train_data = train_data
        s._val_data = val_data

    def train(s, model, opt, hook=Hook(), i=0, timelimit=None, steplimit=None):
        timer = Timer(timelimit)
        steplimit = math.inf if steplimit is None else steplimit + i

        for batch in s._train_data:
            # prep
            model.train() # inside loop b/c hook might switch model to eval mode
            opt.zero_grad()

            # execute step
            loss = s._env(model, batch)
            loss.backward()
            opt.step()

            # call hook, possibly stop training
            stop = hook.step(i, loss)
            i += 1
            if stop or timer() or (i >= steplimit):
                break

        return i
