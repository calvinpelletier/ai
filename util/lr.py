from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(s, rates, losses):
        s.rates = rates
        s.losses = losses

    def suggest(s, skip_start=10, skip_end=5):
        return s._get_suggestion(*s._trim(skip_start, skip_end))

    def plot(s, skip_start=10, skip_end=5, show=True, save=None):
        assert skip_start >= 0 and skip_end >= 0
        assert vertical is None or isinstance(vertical, float)

        fig, ax = plt.subplots()
        ax.plot(rates, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        ax.axvline(x=s._get_suggestion(), color='red')

        if show:
            plt.show()
        if save:
            plt.savefig(save)

    def _trim(s, n_start=10, n_end=5):
        if n_end == 0:
            return s.rates[n_start:], s.losses[n_start:]
        return s.rates[n_start:-n_end], s.losses[n_start:-n_end]

    def _get_suggestion(s, rates, losses):
        idx = np.gradient(np.array(losses)).argmin()
        return rates[idx]


def analyze_lr(
    model,
    opt,
    env,
    train_data,
    val_data=None,
    max_lr=10,
    n_steps=100,
    smooth_f=0.05,
    diverge_th=5,
):
    assert not _has_scheduler(opt)
    assert 0 <= smooth_f < 1

    lr_schedule = ExponentialLR(opt, max_lr, n_steps)

    rates = []
    losses = []
    best_loss = None

    train_data = iter(train_data)
    for step in tqdm(range(n_steps)):
        loss = _train_step(model, opt, env, next(train_data))
        if val_data:
            loss = _validate(model, env, val_data)

        rates.append(lr_schedule.get_lr()[0])
        lr_schedule.step()

        if best_loss is None:
            best_loss = loss
        else:
            if smooth_f > 0:
                loss = smooth_f * loss + (1 - smooth_f) * losses[-1]
            best_loss = min(best_loss, loss)
        losses.append(loss)

        if loss > diverge_th * best_loss:
            break

    return Analysis(rates, losses)


def _train_step(model, opt, env, batch):
    model.train()
    opt.zero_grad()
    loss = env(model, batch)
    loss.backward()
    opt.step()
    return loss.item()


def _validate(model, env, val_data):
    model.eval()
    loss = 0.
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_data):
            loss += env(model, batch).item()
            n += 1
    return loss / n


def _has_scheduler(opt):
    for param_group in opt.param_groups:
        if 'initial_lr' in param_group:
            return True
    return False
