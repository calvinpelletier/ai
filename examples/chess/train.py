import torch

import ai
from ai.train.logger import log

from .model import build_model


def train(cfg, val_iter, train_ds, device, save_snapshots=False):
    model = build_model(cfg)

    name, wnb = name_and_wnb(cfg)
    trial = ai.Trial(
        name,
        clean=True,
        config=cfg,
        wnb=wnb,
        save_snapshots=save_snapshots,
        val_data=val_iter,
        val_stopper=ai.train.EarlyStopper(
            cfg.train.stop.early.improvement,
            cfg.train.stop.early.patience,
        ),
    )

    ai.Trainer(
        build_env(cfg),
        train_ds.iterator(cfg.train.bs, device, train=True),
    ).train(
        model.init().to(device),
        ai.opt.build(cfg.train.opt, model),
        trial.hook(),
        steplimit=cfg.train.stop.steplimit,
        timelimit=cfg.train.stop.timelimit,
    )


def name_and_wnb(cfg):
    project = 'ac'
    ds = cfg.data.replace('/', '-')
    task = cfg.task
    id = ai.util.gen_id()
    group = cfg.group if hasattr(cfg, 'group') else None

    if group is not None:
        return '/'.join([project, ds, task, group, id]), {
            'project': '.'.join([project, ds, task]),
            'name': '-'.join([group, id]),
            'id': '-'.join([group, id]),
            'group': group,
        }

    return '/'.join([project, ds, task, id]), {
        'project': '.'.join([project, ds, task]),
        'name': id,
        'id': id,
    }


class Env:
    def __init__(s, inputs, loss_cls, cfg):
        s._inputs = inputs
        s._loss = loss_cls(cfg)

    def __call__(s, model, data, step=0):
        return s._loss(data, model(*[data[i] for i in s._inputs]))


class ValueLoss:
    def __init__(s, cfg):
        s._loss_fn = torch.nn.MSELoss()

    def __call__(s, data, output):
        return s._loss_fn(output, data['value'].to(torch.float32))


class ActionLoss:
    def __init__(s, cfg):
        s._loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s, data, output):
        return s._loss_fn(output, data['action'].to(torch.long))


class LegalLoss:
    def __init__(s, cfg):
        s._loss_fn = torch.nn.MSELoss()

    def __call__(s, data, output):
        return s._loss_fn(output, data['legal'].to(torch.float32))


class MetaLoss:
    def __init__(s, cfg):
        s._loss_fn = torch.nn.MSELoss()

    def __call__(s, data, output):
        return s._loss_fn(output, data['meta'].to(torch.float32))


class BoardLoss:
    def __init__(s, cfg):
        s._loss_fn = torch.nn.L1Loss()

    def __call__(s, data, output):
        return s._loss_fn(output, data['board'].to(torch.float32))


class ValueActionLoss:
    def __init__(s, cfg):
        s._v_weight = cfg.loss.v.weight
        s._v_loss_fn = torch.nn.MSELoss()
        s._a_loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(s, data, output):
        value, policy = output
        v_loss = s._v_loss_fn(value, data['value'].to(torch.float32))
        a_loss = s._a_loss_fn(policy, data['action'].to(torch.long))
        log('loss.value', v_loss)
        log('loss.action', a_loss)
        return v_loss * s._v_weight + a_loss


MAP = {
    'b2l': (['board'], LegalLoss),
    'b2a': (['board'], ActionLoss),
    'h2b': (['history', 'history_len'], BoardLoss),
}
def build_env(cfg):
    inputs, loss_cls = MAP[cfg.task]
    return Env(inputs, loss_cls, cfg.train)
