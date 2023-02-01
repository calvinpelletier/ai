import torch
import numpy as np
from time import sleep

from ai.model import Model, fc
from ai.infer import Inferencer


def test_inferencer():
    model = Model(fc(8, 8)).init().eval()
    inferencer = Inferencer(model, batch_size=1)
    assert_parity(model, inferencer)


def test_weight_update():
    model = Model(fc(8, 8)).init().eval()
    inferencer = Inferencer(model, batch_size=1)

    # reinitialize model params and send to inferencer
    model.init()
    inferencer.update_weights(model.state_dict())
    sleep(2)

    assert_parity(model, inferencer)


def test_batching():
    model = Model(fc(8, 8)).init().eval()
    inferencer = Inferencer(model, batch_size=8, debug=True)

    inputs = [torch.randn(1, 8) for _ in range(16)]
    with torch.no_grad():
        gt = [model(x) for x in inputs]
    outputs = inferencer.multi_infer([([inputs[i]], {}) for i in range(16)])
    for y1, y2 in zip(outputs, gt):
        assert close(y1, y2)

    info = inferencer.debug()
    print(info)
    assert info['avg_batch_size'] > 1


def assert_parity(model, inferencer):
    x = torch.randn(1, 8)
    with torch.no_grad():
        y1 = model(x)
    y2 = inferencer(x)
    assert close(y1, y2)


def close(a, b):
    return np.allclose(a.numpy(), b.numpy(), rtol=1e-4, atol=1e-6)
