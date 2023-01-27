import numpy as np

from ai.util.fid import calc_fid, calc_fid_stats_for_dataset
from ai.util import null_op


class ImgGenTask:
    def __init__(s,
        ds,
        log=null_op,
        device='cuda',
        bs=64,
        n_workers=8,
        n_imgs=10_000,
    ):
        s._ds = ds
        s._log = log
        s._device = device
        s._bs = bs
        s._n_workers = n_workers
        s._n_imgs = n_imgs
        s._ds_fid_stats = s._load_or_calc_fid_stats(ds)

    def __call__(s, generator, step=None):
        fid = calc_fid(generator, s._ds_fid_stats, s._n_imgs, s._device, s._bs,
            s._n_workers)
        s._log(step, 'task.fid', fid)
        return fid

    def _load_or_calc_fid_stats(s, ds):
        path = ds.metadata_path / 'fid.npz'
        if path.exists():
            return np.load(path)

        stats = calc_fid_stats_for_dataset(ds, s._device, s._bs, s._n_workers)
        np.savez(path, **stats)
        return stats
