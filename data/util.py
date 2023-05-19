import torch


def transfer_data(data, device, nonblock=True):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=nonblock)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [x.to(device, non_blocking=nonblock) for x in data]
    elif isinstance(data, dict):
        return {k: v.to(device, non_blocking=nonblock) for k, v in data.items()}
    else:
        raise ValueError(f'unexpected data type: {type(data)}')


def data_iter(iterator, device, postprocessor=None, nonblock=True):
    for batch in iterator:
        batch = transfer_data(batch, device, nonblock)
        batch = process_data(batch, postprocessor)
        yield batch


def inf_data_iter(iterator, device, postprocessor=None, nonblock=True):
    while 1:
        for batch in iterator:
            batch = transfer_data(batch, device, nonblock)
            batch = process_data(batch, postprocessor)
            yield batch


def process_data(data, fn):
    if isinstance(fn, dict):
        assert isinstance(data, dict)
        for k in fn.keys():
            data[k] = fn[k](data[k])
        return data

    if fn is not None:
        if isinstance(data, dict):
            return {k: fn(v) for k, v in data.items()}
        return fn(data)

    return data
