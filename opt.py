from torch import optim


OPTS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

def build_opt(cfg, model):
    if cfg.type not in OPTS:
        raise ValueError(f'unknown opt: {cfg.type}')
    return OPTS[cfg.type](model.parameters(), lr=cfg.lr)
