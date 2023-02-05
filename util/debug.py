import torch


def print_info(tensor, name=''):
    shape = '[{}]'.format(','.join([str(x) for x in tensor.shape]))
    min_, max_ = torch.min(tensor).item(), torch.max(tensor).item()
    if isinstance(min_, float):
        min_, max_ = f'{min_:.2f}', f'{max_:.2f}'
    dtype = str(tensor.dtype).split('.')[1]
    print(f'{name}: ' + ' '.join([
        f'shape={shape}',
        f'bounds=[{min_},{max_}]',
        f'dtype={dtype}',
        f'device={tensor.device}',
    ]))
