from collections import namedtuple
import torch
from torch import nn

from ai.path import model_path


class ArcFace(nn.Module):
    def __init__(s, path='arcface.pt'):
        super().__init__()

        s.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        s.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.6),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=True),
        )

        chunks = [
            get_chunk(in_channel=64, depth=64, num_units=3),
            get_chunk(in_channel=64, depth=128, num_units=4),
            get_chunk(in_channel=128, depth=256, num_units=14),
            get_chunk(in_channel=256, depth=512, num_units=3),
        ]
        blocks = []
        for chunk in chunks:
            for block in chunk:
                blocks.append(Block(
                    block.in_channel,
                    block.depth,
                    block.stride,
                ))
        s.body = nn.Sequential(*blocks)

        if path is not None:
            s.load_state_dict(torch.load(model_path(path)))

    def forward(s, x):
        x = s.input_layer(x)
        x = s.body(x)
        x = s.output_layer(x)
        return l2_norm(x)


BlockCfg = namedtuple('Block', ['in_channel', 'depth', 'stride'])

class Block(nn.Module):
    def __init__(s, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            s.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            s.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        s.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(s, x):
        shortcut = s.shortcut_layer(x)
        res = s.res_layer(x)
        return res + shortcut

def get_chunk(in_channel, depth, num_units, stride=2):
    return [BlockCfg(in_channel, depth, stride)] + \
        [BlockCfg(depth, depth, 1) for i in range(num_units - 1)]


class SEModule(nn.Module):
    def __init__(s, nc, reduc):
        super().__init__()
        s.avg_pool = nn.AdaptiveAvgPool2d(1)
        s.fc1 = nn.Conv2d(nc, nc // reduc, kernel_size=1, padding=0, bias=False)
        s.relu = nn.ReLU(inplace=True)
        s.fc2 = nn.Conv2d(nc // reduc, nc, kernel_size=1, padding=0, bias=False)
        s.sigmoid = nn.Sigmoid()

    def forward(s, x):
        module_input = x
        x = s.avg_pool(x)
        x = s.fc1(x)
        x = s.relu(x)
        x = s.fc2(x)
        x = s.sigmoid(x)
        return module_input * x


class Flatten(nn.Module):
    def forward(s, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
