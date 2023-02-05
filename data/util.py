import torch


def create_data_loader(
    data,
    batch_size,
    device='cuda',
    shuffle=False,
    infinite=False,
    drop_last=True,
    n_workers=1,
    postprocess=None,
):
    '''
    data : torch.utils.data.(Dataset or IterableDataset)
    batch_size : int
    device : str
    shuffle : bool
    infinite : bool
        wrap loader in an infinite loop
    drop_last : bool
        drop last batch if incomplete
    n_workers : int
    postprocessor : callable or null
        a function called after the data has been transfered to the device
        args
            tensor or list/dict of tensors
        returns
            tensor or list/dict of tensors
    '''

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'drop_last': drop_last if batch_size is not None else False,
        'num_workers': n_workers,
    }
    if device == 'cuda' or device.startswith('cuda:'):
        loader_kwargs['pin_memory'] = True
    elif device == 'cpu':
        pass
    else:
        raise ValueError(f'unexpected device: {device}')

    loader = torch.utils.data.DataLoader(data, **loader_kwargs)
    return data_iter(loader, device, infinite, postprocess)


def data_iter(loader, device, infinite=False, postprocess=None):
    while 1:
        for batch in loader:
            batch = transfer_data(batch, device)
            batch = process(batch, postprocess)
            yield batch

        if not infinite:
            break


def process(data, fn):
    if isinstance(fn, dict):
        assert isinstance(data, dict)
        for k in fn.keys():
            data[k] = fn[k](data[k])
    elif fn is not None:
        data = fn(data)
    return data


def transfer_data(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [x.to(device) for x in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError(f'unexpected data type: {typeof(data)}')
