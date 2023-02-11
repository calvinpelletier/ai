import torch
import torch.nn.functional as F
import numpy as np


DEVICE = 'cuda'


def assert_equal(a, b):
    a = a.detach().cpu().numpy()
    b = b.detach().cpu().numpy()
    assert a.shape == b.shape
    assert np.isclose(a, b).all()


def assert_shape(tensor, shape):
    if -1 in shape:
        assert len(tensor.shape) == len(shape)
        for a, b in zip(tensor.shape, shape):
            assert b == -1 or a == b
    else:
        assert list(tensor.shape) == list(shape)


def assert_bounds(tensor, bounds):
    assert torch.min(tensor) >= bounds[0]
    assert torch.max(tensor) <= bounds[1]


def assert_autoencode(model, shape, device='cuda', lr=1e-3):
    model.init().to(device).train()
    x = torch.randn(*shape).to(device)

    with torch.no_grad():
        y = model(x)
        assert_shape(y, shape)
        loss1 = F.mse_loss(y, x)

    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(8):
        opt.zero_grad()
        loss = F.mse_loss(model(x), x.detach())
        loss.backward()
        opt.step()

    with torch.no_grad():
        loss2 = F.mse_loss(model(x), x)

    assert loss2 < loss1
