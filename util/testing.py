import torch
import torch.nn.functional as F


def assert_shape(tensor, shape):
    if -1 in shape:
        assert len(tensor.shape) == len(shape)
        for a, b in zip(tensor.shape, shape):
            assert b == -1 or a == b
    else:
        assert list(tensor.shape) == list(shape)


def assert_autoencode(model, shape, device='cuda', lr=1e-3):
    model.init().to(device)
    x = torch.randn(*shape).to(device)

    model.eval()
    with torch.no_grad():
        y = model(x)
        assert_shape(y, shape)
        loss1 = F.mse_loss(y, x)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(8):
        opt.zero_grad()
        loss = F.mse_loss(model(x), x.detach())
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        loss2 = F.mse_loss(model(x), x)

    assert loss2 < loss1
