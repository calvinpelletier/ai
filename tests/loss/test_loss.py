import torch
import pytest

from ai.util.testing import *
from ai.train.loss import PerceptualLoss, FaceIdentityLoss, L2Loss
from ai.model.ae import ImgAutoencoder


def test_l2_loss():
    _test(L2Loss())

@pytest.mark.filterwarnings('ignore:.*deprecated')
def test_perceptual_loss():
    _test(PerceptualLoss('lpips-alex'))
    _test(PerceptualLoss('lpips-vgg'))
    _test(PerceptualLoss('trad-vgg'))

def test_face_id_loss():
    _test(FaceIdentityLoss())


def _model():
    return ImgAutoencoder(64, 16).init().to(DEVICE)

def _data():
    return torch.randn(8, 3, 64, 64, device=DEVICE)

def _test(loss_fn):
    model = _model()
    loss = loss_fn(model(_data()), _data())
    loss.backward()
