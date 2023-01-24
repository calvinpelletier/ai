import torch


def create_data_loader(
    data,
    batch_size,
    device='cuda',
    shuffle=False,
    infinite=False,
    n_workers=1,
    postprocess=None,
):
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': n_workers,
    }
    if device == 'cuda' or device.startswith('cuda:'):
        loader_kwargs['pin_memory'] = True
    elif device == 'cpu':
        pass
    else:
        raise ValueError(f'unexpected device: {device}')

    loader = torch.utils.data.DataLoader(data, **loader_kwargs)
    it = data_iter(loader, device, postprocess)
    if infinite:
        return inf_loop(it)
    return it


def data_iter(loader, device, postprocess=None):
    for batch in loader:
        batch = transfer_data(batch, device)
        if postprocess is not None:
            batch = postprocess(batch)
        yield batch


def inf_loop(it):
    while 1:
        for x in it:
            yield x


def transfer_data(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, tuple) or isinstance(data, list):
        return [x.to(device) for x in data]
    elif isinstance(data, dict):
        return {k: v.to(device) for k, v in data.items()}
    else:
        raise ValueError(f'unexpected data type: {typeof(data)}')
