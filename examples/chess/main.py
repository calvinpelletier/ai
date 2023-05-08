import ai

from ai.examples.chess.data import build_dataset
from ai.examples.chess.train import train


class CLI:
    def __init__(s, ds='test/10k', device='cuda', val_batch_size=128):
        s._ds_name = ds
        s._device = device
        s._val_bs = val_batch_size

    def b2l_local(s):
        task = 'b2l'
        group = 'local'
        def configure():
            c = Config(s._ds_name, task, group)
            c('model.dim', [4, 8, 16, 32])
            c('model.pre.type', ['emb', 'fc'])
            c('model.pre.posify', [True, False])
            if c.model.pre.type == 'fc':
                c('model.pre.n_layers', [1, 2, 4])
            c('model.main.type', 'none')
            return c
        s._train_many(task, configure)

    def b2l_global(s):
        task = 'b2l'
        group = 'global'
        def configure():
            c = Config(s._ds_name, task, group)
            c('model.dim', [16, 32, 64])
            c('model.pre.type', ['emb', 'fc'])
            c('model.pre.posify', False)
            if c.model.pre.type == 'fc':
                c('model.pre.n_layers', [2, 4])
            configure_processor(c('model.main'), [1, 2, 4])
            return c
        s._train_many(task, configure)

    def h2b(s):
        task = 'h2b'
        def configure():
            c = Config(s._ds_name, task)
            c('model.dim', [64, 128, 256, 512])
            c('model.encode.type', 'lstm')
            c('model.encode.n_layers', [1, 2, 4])
            c('model.decode.n_layers', [1, 2, 4])
            return c
        s._train_many(task, configure)

    def b2a(s):
        task = 'b2a'
        def configure():
            c = Config(s._ds_name, task)
            c('model.dim', [128, 256, 512])
            c('model.pre.type', ['emb', 'fc'])
            c('model.pre.posify', False)
            if c.model.pre.type == 'fc':
                c('model.pre.n_layers', [2, 4])
            configure_processor(c('model.main'), [1, 2, 4])
            return c
        s._train_many(task, configure)

    def _train_many(s, task, configure):
        val_iter, train_ds = s._setup_data(s._ds_name, task)
        for cfg in gen_configs(configure):
            train(cfg, val_iter, train_ds, s._device)

    def _setup_data(s, ds, task):
        val_ds, train_ds = build_dataset(ds, task).split(.01, .99)
        val_iter = val_ds.iterator(s._val_bs, s._device, train=False)
        return val_iter, train_ds


class Config(ai.ChooseConfig):
    def __init__(s, ds, task, group=None):
        super().__init__({'data': ds, 'task': task})
        if group is not None:
            s('group', group)

        s('train.bs', 32)

        s('train.opt.type', 'adamw')
        s('train.opt.lr', 0.001)
        s('train.opt.grad_clip', True)

        s('train.stop.timelimit', None)
        s('train.stop.steplimit', None)
        s('train.stop.early.improvement', 0.03)
        s('train.stop.early.patience', 3)


def gen_configs(generator, n=None, wait=32):
    max_n = n
    cfgs = set()
    i = 0
    last_new = 0
    n = 0
    while (max_n is None or n < max_n) and i - last_new < wait:
        i += 1
        cfgs.add(generator())
        if len(cfgs) > n:
            last_new = i
            n = len(cfgs)
    return list(cfgs)


def configure_processor(c, n, optional=False, has_pos_info=False):
    types = ['conv', 'tx', 'hybrid']
    if optional:
        types.append('none')
    c('type', types)

    if c.type == 'conv':
        c('n_blocks', n)
    elif c.type == 'tx':
        c('n_blocks', n)
        c('n_heads', n)
    elif c.type == 'hybrid':
        c('first', ['conv', 'tx'])
        c('conv.n_blocks', n)
        c('tx.n_blocks', n)
        c('tx.n_heads', n)

    posify_required = (
        not has_pos_info and
        (c.type == 'tx' or (c.type == 'hybrid' and c.first == 'tx'))
    )
    posify = [True] if posify_required else [True, False]
    if c.type == 'hybrid':
        if c.first == 'conv':
            c('conv.posify', posify)
        else:
            assert c.first == 'tx'
            c('tx.posify', posify)
    else:
        c('posify', posify)


if __name__ == '__main__':
    ai.run(CLI)
