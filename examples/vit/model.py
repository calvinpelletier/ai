import torch
from torch import nn
from einops import repeat
from typing import Optional

import ai.model as m
from ai.util import assert_equal


class VisionTransformer(m.Model):
    def __init__(s,
        imsize: int,
        patch_size: int,
        n_out: int,
        dim: int,
        n_blocks: int,
        n_heads: int,
        mlp_dim: int,
        head_dim: Optional[int] = None,
        nc_in: int = 3,
        pool: str = 'cls',
        dropout: float = 0.,
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}
        s._pool = pool

        img_h, img_w = parse_size(imsize)
        patch_h, patch_w = parse_size(patch_size)
        assert img_h % patch_h == 0 and img_w % patch_w == 0
        n_patches = (img_h // patch_h) * (img_w // patch_w)
        patch_dim = nc_in * patch_h * patch_w

        s._to_patch_embedding = m.seq(
            m.rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_h, p2=patch_w),
            m.layer_norm(patch_dim),
            m.fc(patch_dim, dim),
            m.layer_norm(dim),
        )

        s._pos_embedding = nn.Parameter(torch.randn(1, n_patches + 1, dim))
        s._cls_token = nn.Parameter(torch.randn(1, 1, dim))

        s._transformer = m.tx_enc(n_blocks, dim, n_heads, mlp_dim,
            head_dim, dropout)

        s._mlp_head = m.seq(
            m.layer_norm(dim),
            m.fc(dim, n_out),
        )

    def init_params(s):
        nn.init.normal_(s._pos_embedding)
        nn.init.normal_(s._cls_token)

    def forward(s, img):
        x = s._to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(s._cls_token, '1 1 d -> b 1 d', b=b)
        assert_equal(cls_tokens, s._cls_token.repeat(b, 1, 1))
        x = torch.cat((cls_tokens, x), dim=1)
        x += s._pos_embedding[:, :(n + 1)]

        x = s._transformer(x)

        x = x.mean(dim=1) if s._pool == 'mean' else x[:, 0]

        return s._mlp_head(x)

def parse_size(size):
    if isinstance(size, tuple) or isinstance(size, list):
        assert len(size) == 2
        return size
    assert isinstance(size, int)
    return size, size
