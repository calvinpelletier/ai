import torch


def transfer_data(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [x.to(device) for x in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError(f'unexpected data type: {typeof(data)}')


def data_iter(iterator, device, postprocessor=None):
    for batch in iterator:
        batch = transfer_data(batch, device)
        batch = process_data(batch, postprocessor)
        yield batch


def inf_data_iter(iterator, device, postprocessor=None):
    while 1:
        for batch in iterator:
            batch = transfer_data(batch, device)
            batch = process_data(batch, postprocessor)
            yield batch


def process_data(data, fn):
    if isinstance(fn, dict):
        assert isinstance(data, dict)
        for k in fn.keys():
            data[k] = fn[k](data[k])
    elif fn is not None:
        data = fn(data)
    return data
