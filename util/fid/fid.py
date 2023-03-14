import torch
import numpy as np
from scipy import linalg

from ai.util.fid.inception import InceptionV3
from ai.util import assert_shape


DIMS = 2048


def calc_fid(gen, ds_stats, n=10_000, device='cuda', bs=64, n_workers=8):
    return _calc_fid(
        calc_fid_stats_for_generator(gen, n, device, bs, n_workers),
        ds_stats,
    )


def calc_fid_stats_for_dataset(ds, device='cuda', bs=64, n_workers=8):
    model = _inception(device)
    data = ds.iterator(bs, device, n_workers)

    feats = np.empty([ds.length(bs), DIMS])
    i = 0
    with torch.no_grad():
        for batch in data:
            feat = model(batch)[0]
            assert_shape(feat, [bs, DIMS, 1, 1])
            feat = feat.reshape(bs, DIMS).cpu().numpy()
            feats[i:i+bs] = feat
            i += bs

    return _calc_fid_stats(feats)


def calc_fid_stats_for_generator(gen, n, device='cuda', bs=64, n_workers=8):
    model = _inception(device)
    gen.to(device).eval()
    n_batches = n // bs

    feats = np.empty([n_batches * bs, DIMS])
    with torch.no_grad():
        for i in range(n_batches):
            img = gen(torch.randn(bs, gen.z_dim, device=device))
            feat = model(img)[0]
            assert_shape(feat, [bs, DIMS, 1, 1])
            feat = feat.reshape(bs, DIMS).cpu().numpy()
            feats[i*bs:(i+1)*bs] = feat

    return _calc_fid_stats(feats)


def _calc_fid_stats(feats):
    return {
        'mu': np.mean(feats, axis=0),
        'sigma': np.cov(feats, rowvar=False),
    }


def _calc_fid(stats1, stats2, eps=1e-6):
    mu1 = np.atleast_1d(stats1['mu'])
    mu2 = np.atleast_1d(stats2['mu'])
    sigma1 = np.atleast_2d(stats1['sigma'])
    sigma2 = np.atleast_2d(stats2['sigma'])
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f'imaginary component {m}')
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
        + np.trace(sigma2) - 2 * tr_covmean)


def _inception(device):
    return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]]).to(device).eval()
