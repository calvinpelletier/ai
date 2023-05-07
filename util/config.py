import yaml
from copy import deepcopy
from random import choice


class _NullValue:
    pass


class Config:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # CONSTRUCTORS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(s, data={}):
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(s, k, s._create_subconfig_(v))
            else:
                setattr(s, k, s._parse_value_(v))

    @classmethod
    def load(cls, path, override=None):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if override is not None:
            data = _override(data, override)

        return cls(data)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PUBLIC METHODS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __call__(s, key, value=_NullValue):
        assert key
        return s._set_(key.split('.'), value)

    def __str__(s):
        lines = []
        s._as_string_(lines, 0)
        return '\n'.join(lines) + '\n'

    def _as_flat_dict_(s):
        ret = {}
        for k, v in vars(s).items():
            if isinstance(v, Config):
                v._as_flat_dict_helper_(ret, f'{k}.')
            else:
                ret[k] = v
        return ret

    def _as_dict_(s):
        ret = {}
        for k, v in vars(s).items():
            ret[k] = v._as_dict_() if isinstance(v, Config) else v
        return ret

    def _write_yaml_(s, path):
        with open(path, 'w') as f:
            yaml.dump(s._as_dict_(), f)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PRIVATE/PROTECTED METHODS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _set_(s, key, value):
        k = key[0]
        if len(key) > 1:
            if not hasattr(s, k):
                setattr(s, k, s._create_subconfig_())
            return getattr(s, k)._set_(key[1:], value)
        else:
            if value == _NullValue:
                if hasattr(s, k):
                    subcfg = getattr(s, k)
                    assert isinstance(subcfg, Config)
                else:
                    subcfg = s._create_subconfig_()
                    setattr(s, k, subcfg)
                return subcfg
            else:
                assert not hasattr(s, k)
                v = s._parse_value_(value)
                setattr(s, k, v)
                return v

    def _as_flat_dict_helper_(s, ret, prefix):
        for k, v in vars(s).items():
            if isinstance(v, Config):
                v._as_flat_dict_helper_(ret, prefix + f'{k}.')
            else:
                ret[prefix + k] = v

    def _as_string_(s, lines, depth):
        prefix = ' ' * (2 * depth)
        for k, v in vars(s).items():
            if isinstance(v, Config):
                lines.append(f'{prefix}{k}:')
                v._as_string_(lines, depth + 1)
            else:
                lines.append(f'{prefix}{k}: {str(v)}')

    def _create_subconfig_(s, data={}):
        return Config(data)

    def _parse_value_(s, value):
        return value
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ChooseConfig(Config):
    def _create_subconfig_(s, data={}):
        return ChooseConfig(data)

    def _parse_value_(s, value):
        if isinstance(value, list):
            return choice(value)
        return value

    def __hash__(s):
        return hash(frozenset(s._as_flat_dict_().items()))

    def __eq__(s, other):
        if not isinstance(other, Config):
            return False
        return s._as_flat_dict_() == other._as_flat_dict_()


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
