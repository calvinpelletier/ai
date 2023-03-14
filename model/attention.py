import torch
from torch import nn
from einops import rearrange
from typing import Optional

from ai.model.sequence import seq, repeat
from ai.model.linear import fc
from ai.model.etc import res


def tx_enc(
    n_blocks: int,
    dim: int,
    n_heads: int,
    mlp_dim: int,
    head_dim: Optional[int] = None,
    dropout: float = 0.,
    norm: bool = True,
):
    '''Transformer encoder.

    INPUT
        tensor[b, n, <dim>]
    OUTPUT
        tensor[b, n, <dim>]

    ARGS
        n_blocks : int
            number of transformer blocks
        dim : int
            input/output size
        n_heads : int
            number of attention heads
        mlp_dim : int
            hidden size of feed-forward block
        head_dim : int or null
            size of attention heads (if null, head_dim = dim // n_heads)
        dropout : float
            dropout probability
        norm : bool
            do layer normalization before the attention and feed-forward blocks
    '''

    blk = tx_enc_blk(dim, n_heads, mlp_dim, head_dim, dropout, norm)
    return repeat(n_blocks, blk)


def tx_enc_blk(
    dim: int,
    n_heads: int,
    mlp_dim: int,
    head_dim: Optional[int] = None,
    dropout: float = 0.,
    norm: bool = True,
):
    '''Transformer encoder block.

    INPUT
        tensor[b, n, <dim>]
    OUTPUT
        tensor[b, n, <dim>]

    ARGS
        dim : int
            input/output size
        n_heads : int
            number of attention heads
        mlp_dim : int
            hidden size of feed-forward block
        head_dim : int or null
            size of attention heads (if null, head_dim = dim // n_heads)
        dropout : float
            dropout probability
        norm : bool
            do layer normalization before the attention and feed-forward blocks
    '''

    if head_dim is None or head_dim == dim // n_heads:
        return nn.TransformerEncoderLayer(
            dim,
            n_heads,
            mlp_dim,
            dropout,
            nn.GELU(),
            batch_first=True,
            norm_first=norm,
        )
    return seq(
        res(sa(dim, n_heads, head_dim, dropout, norm)),
        res(ff(dim, mlp_dim, dropout, norm)),
    )


def sa(
    dim: int,
    n_heads: int,
    head_dim: Optional[int] = None,
    dropout: float = 0.,
    norm: bool = True,
):
    '''Multi-head self-attention.

    INPUT
        tensor[b, n, <dim>]
    OUTPUT
        tensor[b, n, <dim>]

    ARGS
        dim : int
            input/output size
        n_heads : int
            number of attention heads
        head_dim : int or null
            size of attention heads (if null, head_dim = dim // n_heads)
        dropout : float
            dropout probability
        norm : bool
            pre-normalize with layer normalization
    '''

    if head_dim is None or head_dim == dim // n_heads:
        return _TorchSelfAttention(dim, n_heads, dropout, norm)
    return _SelfAttention(dim, n_heads, head_dim, dropout, norm)


def ff(
    dim: int,
    hidden_dim: int,
    dropout: float = 0.,
    norm: bool = True,
    actv: str = 'gelu',
):
    '''Feed-forward block.

    INPUT
        tensor[b, n, <dim>]
    OUTPUT
        tensor[b, n, <dim>]

    ARGS
        dim : int
            input/output size
        hidden_dim : int
            hidden size
        dropout : float
            dropout probability
        norm : bool
            do layer normalization before the fully-connected layers
        actv : str
            activation function (see model/actv.py for possible values)
    '''
    return seq(
        nn.LayerNorm(dim) if norm else nn.Identity(),
        fc(dim, hidden_dim, actv=actv, dropout=dropout),
        fc(hidden_dim, dim, dropout=dropout),
    )


class Attention(nn.Module):
    '''Multi-head attention.

    INPUT
        q : tensor[b, n, <dim>]
            The source of the query.
        kv: tensor[b, n, <dim>]
            The source of the key and value.
    OUTPUT
        tensor[b, n, <dim>]

    ARGS
        dim : int
            input/output size
        n_heads : int
            number of attention heads
        head_dim : int or null
            size of attention heads (if null, head_dim = dim // n_heads)
        dropout : float
            dropout probability
        norm : bool
            pre-normalize with layer normalization
    '''

    def __init__(s,
        dim: int,
        n_heads: int,
        dropout: float = 0.,
        norm: bool = True,
    ):
        super().__init__()
        s._norm = nn.LayerNorm(dim) if norm else nn.Identity()
        s._attn = nn.MultiheadAttention(
            dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(s, q, kv, need_weights=False, **attn_kw):
        q = s._norm(q)
        kv = s._norm(kv)
        out, weights = s._attn(q, k, v, need_weights=need_weights, **attn_kw)
        if need_weights:
            return out, weights
        return out

def attn(*a, **kw):
    return Attention(*a, **kw)


class _TorchSelfAttention(nn.Module):
    def __init__(s,
        dim: int,
        n_heads: int,
        dropout: float = 0.,
        norm: bool = True,
    ):
        super().__init__()
        s._norm = nn.LayerNorm(dim) if norm else nn.Identity()
        s._attn = nn.MultiheadAttention(
            dim,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(s, x, need_weights=False, **attn_kw):
        x = s._norm(x)
        out, weights = s._attn(x, x, x, need_weights=need_weights, **attn_kw)
        if need_weights:
            return out, weights
        return out


class _SelfAttention(nn.Module):
    def __init__(s,
        dim: int,
        n_heads: int,
        head_dim: int,
        dropout: float = 0.,
        norm: bool = True,
    ):
        super().__init__()
        inner_dim = head_dim * n_heads
        s._n_heads = n_heads
        s._scale = head_dim ** -0.5

        s._norm = nn.LayerNorm(dim) if norm else nn.Identity()

        s._attend = seq(nn.Softmax(dim=-1), nn.Dropout(dropout))

        s._to_qkv = fc(dim, inner_dim * 3, bias=False)

        s._to_out = seq(
            fc(inner_dim, dim),
            nn.Dropout(dropout),
        ) if n_heads != 1 or head_dim != dim else nn.Identity()

    def forward(s, x):
        x = s._norm(x)

        qkv = s._to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=s._n_heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * s._scale

        attn = s._attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return s._to_out(out)
