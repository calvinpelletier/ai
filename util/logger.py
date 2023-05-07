from torch.utils.tensorboard import SummaryWriter
import wandb


class Tensorboard:
    def __init__(s, path, wnb=False):
        s._writer = SummaryWriter(path)

        s._wnb_data = {} if wnb else None
        s._wnb_step = None

    def __call__(s, step, key, value):
        s._writer.add_scalar(key, value, step)

        if s._wnb_data is not None:
            if step != s._wnb_step:
                s._send_wnb()
                s._wnb_step = step
            s._wnb_data[key] = value

    def _send_wnb(s):
        if s._wnb_data:
            assert s._wnb_step is not None
            wandb.log(s._wnb_data, step=s._wnb_step)
        s._wnb_data = {}
        s._wnb_step = None
