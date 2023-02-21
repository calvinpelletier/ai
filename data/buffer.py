import random
import torch
from collections import defaultdict
from typing import Iterable, Dict


class DataBuffer:
    '''Buffer for storing generated data.'''

    def __init__(s, generator: Iterable, batch_size: int, buf_size: int):
        s._generator = generator
        s.batch_size = batch_size
        s.buf_size = buf_size

        s._buf = []

    def get_batch(s) -> Dict[str, torch.Tensor]:
        s._fill_buf()
        return s._create_batch()

    def _fill_buf(s):
        raise NotImplementedError()

    def _create_batch(s):
        raise NotImplementedError()


class ReplayBuffer(DataBuffer):
    '''Buffer for storing generated game replays.'''

    def __init__(s,
        generator: Iterable,
        batch_size: int,
        buf_size: int,
        n_replay_times: int,
    ):
        super().__init__(generator, batch_size, buf_size)
        s.n_replay_times = n_replay_times

    def _fill_buf(s):
        while len(s._buf) < s.buf_size:
            s._buf.append(_Replay(s, next(s._generator)))

    def _create_batch(s):
        if s.buf_size == s.batch_size:
            replays = s._buf
        else:
            replays = random.sample(s._buf, s.batch_size)

        batch = defaultdict(list)
        for replay in replays:
            state = replay.sample_state()
            batch['ob'].append(state.ob)
            batch['pi'].append(state.pi)
            batch['v'].append(state.to_play * replay.outcome)
        return {k: torch.stack(vals).float() for k, vals in batch.items()}

    def remove_replay(s, replay):
        s._buf.remove(replay)


class _Replay:
    def __init__(s, buf, data):
        s._buf = buf
        s.outcome = data['outcome']
        s._states = {
            i: _State(i, data['ob'][i], data['pi'][i])
            for i in range(len(data['ob']))
        }
        s._hit_count = defaultdict(int)

    def sample_state(s):
        i = random.choice(list(s._states.keys()))
        state = s._states[i]
        s._hit(i)
        return state

    def _hit(s, i):
        s._hit_count[i] += 1
        if s._hit_count[i] >= s._buf.n_replay_times:
            s._states.pop(i)
            if not s._states:
                s._buf.remove_replay(s)


class _State:
    def __init__(s, ply, ob, pi):
        s.ply = ply
        s.ob = ob
        s.pi = pi
        s.to_play = 1 if ply % 2 == 0 else -1 # TODO: non-2p games
