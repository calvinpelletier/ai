from torch import optim


OPTS = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
}

def build(cfg, model):
    if cfg.type not in OPTS:
        raise ValueError(f'unknown opt: {cfg.type}')
    return OPTS[cfg.type](model.parameters(), lr=cfg.lr)


def sgd(model, lr=1e-4):
    return optim.SGD(model.parameters(), lr)


def adam(model, lr=1e-4):
    return optim.Adam(model.parameters(), lr)
