from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import matplotlib.pyplot as plt


class Analysis:
    def __init__(s, rates, losses):
        s.rates = rates
        s.losses = losses

    def suggest(s, skip_start=0, skip_end=0):
        return s._get_suggestion(*s._trim(skip_start, skip_end))

    def plot(s, skip_start=0, skip_end=0, show=True, save=None):
        assert skip_start >= 0 and skip_end >= 0
        rates, losses = s._trim(skip_start, skip_end)

        fig, ax = plt.subplots()
        ax.plot(rates, losses)
        ax.set_xscale('log')
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Loss')
        ax.axvline(x=s._get_suggestion(rates, losses), color='red')

        if show:
            plt.show()
        if save:
            plt.savefig(save)

    def _trim(s, n_start, n_end):
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
    n_val=10,
    max_lr=10,
    n_steps=100,
    smooth_f=0.05,
    diverge_th=1.25,
):
    assert not _has_scheduler(opt)
    assert 0 <= smooth_f < 1

    lr_schedule = _ExponentialLR(opt, max_lr, n_steps)

    rates = []
    losses = []
    best_loss = None

    train_data = iter(train_data)
    for step in tqdm(range(n_steps)):
        loss = _train_step(model, opt, env, next(train_data))
        if val_data:
            loss = _validate(model, env, val_data, n_val)

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


class _ExponentialLR(_LRScheduler):
    def __init__(s, opt, end_lr, num_iter):
        assert num_iter > 1
        s.end_lr = end_lr
        s.num_iter = num_iter
        super().__init__(opt, -1)

    def get_lr(s):
        r = s.last_epoch / (s.num_iter - 1)
        return [base_lr * (s.end_lr / base_lr) ** r for base_lr in s.base_lrs]


def _train_step(model, opt, env, batch):
    model.train()
    opt.zero_grad()
    loss = env(model, batch)
    loss.backward()
    opt.step()
    return loss.item()


def _validate(model, env, val_data, n_val=None):
    model.eval()
    loss = 0.
    n = 0
    with torch.no_grad():
        for batch in val_data:
            loss += env(model, batch).item()
            n += 1
            if n_val is not None and n >= n_val:
                break
    return loss / n


def _has_scheduler(opt):
    for param_group in opt.param_groups:
        if 'initial_lr' in param_group:
            return True
    return False
