from os import environ
from pathlib import Path


def dataset(path):
    if path.startswith('/'):
        return Path(path)
    return _get_env_path('AI_DATASETS_PATH') / path

def lab(path):
    if path.startswith('/'):
        return Path(path)
    return _get_env_path('AI_LAB_PATH') / path


def _get_env_path(name):
    path = environ.get(name)
    if path is None:
        raise Exception(f'Missing {name} environment variable.')
    return Path(path)
