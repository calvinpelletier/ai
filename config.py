from copy import deepcopy
import yaml


class _BaseConfig:
    def _set_(s, k, v):
        if len(k) > 1:
            if hasattr(s, k[0]):
                getattr(s, k[0])._set_(k[1:], v)
            else:
                setattr(s, k[0], _SubConfig(k[1:], v))
        else:
            assert not hasattr(s, k[0])
            setattr(s, k[0], v)

    def _str_(s, lines, depth):
        prefix = ' ' * (4 * depth)
        for k, v in vars(s).items():
            if isinstance(v, _SubConfig):
                lines.append(f'{prefix}{k}:')
                v._str_(lines, depth + 1)
            else:
                lines.append(f'{prefix}{k}: {str(v)}')


class Config(_BaseConfig):
    def __init__(s, cfg_dict, override=None):
        if override is not None:
            if isinstance(override, str):
                override = _load_yaml(override)
            else:
                assert isinstance(override, dict)
            cfg_dict = _merge(cfg_dict, override)

        for k, v in cfg_dict.items():
            s._set_(k.split('.'), v)

    def __str__(s):
        lines = []
        s._str_(lines, 0)
        return '\n'.join(lines)

def _load_yaml(path):
    with open(path, 'r') as f:
        x = yaml.safe_load(f)
    return x

def _merge(default, override):
    ret = deepcopy(default)
    ret.update(override)
    return ret


class _SubConfig(_BaseConfig):
     def __init__(s, k, v):
         s._set_(k, v)
