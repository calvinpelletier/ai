from copy import deepcopy
import yaml


class _Config:
    def __init__(s, cfg):
        for k, v in cfg.items():
             s._set_(k, v)

    def _set_(s, k, v):
        if isinstance(v, dict):
            setattr(s, k, _Config(v))
        else:
            setattr(s, k, v)

    def _as_string_(s, lines, depth):
        prefix = ' ' * (2 * depth)
        for k, v in vars(s).items():
            if isinstance(v, _Config):
                lines.append(f'{prefix}{k}:')
                v._as_string_(lines, depth + 1)
            else:
                lines.append(f'{prefix}{k}: {str(v)}')


class Config(_Config):
    def __init__(s, cfg, override=None):
        cfg = _load_if_pathlike(cfg)

        if override is not None:
            override = _load_if_pathlike(override)
            cfg = _override(cfg, override)

        super().__init__(cfg)

    def __str__(s):
        lines = []
        s._as_string_(lines, 0)
        return '\n'.join(lines) + '\n'


def _load_if_pathlike(cfg):
    if isinstance(cfg, dict):
        return cfg
    return _load_yaml(cfg)


def _load_yaml(path):
    with open(path, 'r') as f:
        x = yaml.safe_load(f)
    return x


def _override(default, override):
    ret = deepcopy(default)
    for key, value in override.items():
        assert key
        parts = key.split('.')
        if len(parts) > 1:
            x = ret
            for k in parts[:-1]:
                x = ret[k]
            x[parts[-1]] = value
        else:
            ret[key] = value
    return ret


def _merge(default, override):
    ret = deepcopy(default)
    ret.update(override)
    return ret
