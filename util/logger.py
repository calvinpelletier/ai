from torch.utils.tensorboard import SummaryWriter


class Tensorboard:
    def __init__(s, path):
        s._writer = SummaryWriter(path)

    def __call__(s, step, key, value):
        s._writer.add_scalar(key, value, step)
