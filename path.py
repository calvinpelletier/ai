from os import environ
from pathlib import Path


def dataset_path(path):
    return _path('AI_DATASETS_PATH', path)

def model_path(path):
    return _path('AI_MODELS_PATH', path)

def lab_path(path):
    return _path('AI_LAB_PATH', path)


def _path(env, path):
    if path.startswith('/'):
        return Path(path)
    return _get_env_path(env) / path

def _get_env_path(name):
    path = environ.get(name)
    if path is None:
        raise Exception(f'Missing {name} environment variable.')
    return Path(path)
