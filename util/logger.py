from torch.utils.tensorboard import SummaryWriter
import wandb


class _BaseLogger:
    def __init__(s, wnb=False):
        s._wnb_data = {} if wnb else None
        s._wnb_step = None

    def __call__(s, step, key, value):
        if step > 0 and s._wnb_data is not None:
            if step != s._wnb_step:
                s._send_wnb()
                s._wnb_step = step
            s._wnb_data[key] = value

    def done(s):
        if s._wnb_data is not None:
            s._send_wnb()
            wandb.finish()

    def _send_wnb(s):
        if s._wnb_data:
            assert s._wnb_step is not None
            wandb.log(s._wnb_data, step=s._wnb_step)
        s._wnb_data = {}
        s._wnb_step = None


class Tensorboard(_BaseLogger):
    def __init__(s, path, wnb=False):
        super().__init__(wnb)
        s._writer = SummaryWriter(path)

    def __call__(s, step, key, value):
        super().__call__(step, key, value)
        s._writer.add_scalar(key, value, step)
