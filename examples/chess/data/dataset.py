import numpy as np

from ai.data import Dataset
from ai.util.path import dataset_path


MAP = {
    'a': [('action', np.int16)],
    'b': [('board', np.int8)],
    'h': [('history', np.int16), ('history_len', np.uint8)],
    'l': [('legal', np.uint8)],
    'm': [('meta', np.int32)],
}
def build_dataset(data, task):
    path = dataset_path(f'chess/{data}')

    data = {}
    for x in task:
        if x == '2':
            continue
        for name, type_ in MAP[x]:
            data[name] = np.load(path / f'{name}.npy').astype(type_)

    return Dataset(data)
