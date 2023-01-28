'''parameter initialization'''

import torch.nn as nn
import torch.nn.init as init


def param_init(obj):
    if hasattr(obj, 'init_params') and callable(obj.init_params):
            obj.init_params()

    # linear
    elif isinstance(obj, nn.Linear):
        init.xavier_normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)

    # conv
    elif isinstance(obj, nn.Conv1d):
        init.normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)
    elif isinstance(obj, nn.Conv2d):
        init.xavier_normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)
    elif isinstance(obj, nn.Conv3d):
        init.xavier_normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)

    # transpose conv
    elif isinstance(obj, nn.ConvTranspose1d):
        init.normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)
    elif isinstance(obj, nn.ConvTranspose2d):
        init.xavier_normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)
    elif isinstance(obj, nn.ConvTranspose3d):
        init.xavier_normal_(obj.weight.data)
        if obj.bias is not None:
            init.normal_(obj.bias.data)

    # batch norm
    elif isinstance(obj, nn.BatchNorm1d):
        init.normal_(obj.weight.data, mean=1, std=0.02)
        init.constant_(obj.bias.data, 0)
    elif isinstance(obj, nn.BatchNorm2d):
        if obj.weight is not None:
            init.normal_(obj.weight.data, mean=1, std=0.02)
        if obj.bias is not None:
            init.constant_(obj.bias.data, 0)
    elif isinstance(obj, nn.BatchNorm3d):
        init.normal_(obj.weight.data, mean=1, std=0.02)
        init.constant_(obj.bias.data, 0)

    # lstm
    elif isinstance(obj, nn.LSTM):
        for param in obj.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(obj, nn.LSTMCell):
        for param in obj.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

    # gru
    elif isinstance(obj, nn.GRU):
        for param in obj.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(obj, nn.GRUCell):
        for param in obj.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
