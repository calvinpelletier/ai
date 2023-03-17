# type: ignore
import torch
import numpy as np
from copy import deepcopy
from typing import Optional

from ai.data.queue import DataQueue
from ai.data.rl import RLDataIterator
from ai.infer import Inferencer
from ai.data.buffer import ReplayBuffer
from ai.util.config import Config
from ai.game import Game, Player


USE_TORCH_LOADER = False


class SelfPlay(RLDataIterator):
    @staticmethod
    def from_cfg(cfg: Config, game: Game, player: Player):
        return SelfPlay(
            game,
            player,
            batch_size=cfg.batch_size,
            device=cfg.device,
            model_update_interval=cfg.model_update_interval,
            n_workers=cfg.data.n_workers,
            infer_bs=cfg.infer.batch_size,
            infer_device=cfg.infer.device,
            buf_size=cfg.data.buf_size,
            n_replay_times=cfg.data.n_replay_times,
        )

    def __init__(s,
        game: Game,
        player: Player,

        # external configuration
        batch_size: int = 32,
        device: str = 'cuda',
        model_update_interval: int = 100,

        # internal configuration
        n_workers: Optional[int] = None,
        infer_bs: int = 64,
        infer_device: Optional[str] = None,
        buf_size: Optional[int] = None,
        n_replay_times: int = 4,
    ):
        game = deepcopy(game)
        player = deepcopy(player)

        # check args ~
        if infer_device is None:
            infer_device = device

        if buf_size is None:
            buf_size = batch_size
        else:
            buf_size = max(buf_size, batch_size)
        # ~

        # generate data in this process ~
        if n_workers is None:
            inferencer = None
            player.model.eval().to(infer_device)
            player.device = infer_device

            generator = iter(_SelfPlay(game, player))
        # ~
        # or
        # create inferencer and generate data via workers ~
        else:
            inferencer = Inferencer(player.model, infer_device, infer_bs)
            player.model = inferencer.create_client()
            player.device = 'cpu'

            if USE_TORCH_LOADER:
                generator = iter(torch.utils.data.DataLoader(
                    _SelfPlay(game, player),
                    batch_size=None,
                    num_workers=n_workers,
                ))
            else:
                generator = DataQueue(_SelfPlay(game, player), n_workers)
        # ~

        # store generated data
        buffer = ReplayBuffer(generator, batch_size, buf_size, n_replay_times)

        super().__init__(buffer, device, model_update_interval)

        # for model updates
        s._inferencer = inferencer
        if inferencer is None:
            s._player = player

    def model_update(s, params: dict):
        # print('[INFO] updating inference params')
        if s._inferencer is None:
            s._player.model.load_state_dict(params)
        else:
            s._inferencer.update_params(params)


class _SelfPlay(torch.utils.data.IterableDataset):
    def __init__(s, game, player):
        super().__init__()
        s._game = game
        s._player = player

    def __iter__(s):
        return s

    def __next__(s):
        s._game.reset()
        replay = {'ob': [], 'pi': [], 'outcome': None}
        while s._game.outcome is None:
            replay['ob'].append(s._game.observe())
            action, pi = s._player.act(s._game, return_pi=True)
            replay['pi'].append(pi)
            s._game.step(action)
        replay['ob'] = torch.from_numpy(np.asarray(replay['ob']))
        replay['pi'] = torch.from_numpy(np.asarray(replay['pi']))
        replay['outcome'] = torch.tensor(np.int8(s._game.outcome))
        return replay
