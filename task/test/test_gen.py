import pytest

from ai.data.img import ImgDataset
from ai.task import ImgGenTask
from ai.examples.stylegan2.model import Generator


DEVICE = 'cuda'


@pytest.mark.filterwarnings('ignore:invalid value encountered')
def test_img_gen_task():
    ds = ImgDataset('ffhq', 64)
    task = ImgGenTask(ds, device=DEVICE, n_imgs=256)
    gen = Generator(64).init()
    fid = task(gen)
    print(fid)
    assert 100 < fid < 1000
