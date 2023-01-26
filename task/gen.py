import numpy as np

from ai.util.fid import calc_fid, calc_fid_stats_for_dataset


class ImgGenTask:
    def __init__(s, ds, device='cuda', n_samples=10_000, bs=64, n_workers=8):
        s._device = device
        s._n_samples = n_samples
        s._bs = bs
        s._n_workers = n_workers

        s._ds_fid_stats = s._load_or_calc_fid_stats(ds)

    def __call__(s, generator):
        return calc_fid(generator, s._ds_fid_stats, s._n_samples, s._device,
            s._bs, s._n_workers)

    def _load_or_calc_fid_stats(s, ds):
        path = ds.metadata_path / 'fid.npz'
        if path.exists():
            return np.load(path)

        stats = calc_fid_stats_for_dataset(ds, s._device, s._bs, s._n_workers)
        np.savez(path, **stats)
        return stats
