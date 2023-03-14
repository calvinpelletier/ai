import numpy as np
from typing import Callable, Iterable, Union, Dict

from ai.util.fid import calc_fid, calc_fid_stats_for_dataset
from ai.util import no_op
from ai.model import Model
from ai.data import ImgDataset
from ai.task.base import Task


class ImgGenTask(Task):
    def __init__(s,
        ds: ImgDataset,
        device: str = 'cuda',
        bs: int = 64,
        n_workers: int = 8,
        n_imgs: int = 10_000,
        generator_key: str = 'G',
    ):
        s._ds = ds
        s._device = device
        s._bs = bs
        s._n_workers = n_workers
        s._n_imgs = n_imgs
        s._generator_key = generator_key
        s._ds_fid_stats = s._load_or_calc_fid_stats(ds)

    def __call__(s,
        model: Union[Model, Dict[str, Model]],
        log: Callable = no_op,
    ):
        if isinstance(model, dict):
            model = model[s._generator_key]

        fid = calc_fid(model, s._ds_fid_stats, s._n_imgs, s._device, s._bs,
            s._n_workers)
        log('fid', fid)
        return fid

    def _load_or_calc_fid_stats(s, ds):
        path = ds.metadata_path / 'fid.npz'
        if path.exists():
            return np.load(path)

        stats = calc_fid_stats_for_dataset(ds, s._device, s._bs, s._n_workers)
        np.savez(path, **stats)
        return stats
