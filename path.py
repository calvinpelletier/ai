from os import environ
from pathlib import Path


def _get_env_path(name):
    path = environ.get(name)
    if path is None:
        raise Exception(f'Missing {name} environment variable.')
    return Path(path)

DATASETS_PATH = _get_env_path('AI_DATASETS_PATH')


def dataset(path):
    if path.startswith('/'):
        return Path(path)
    return DATASETS_PATH / path
