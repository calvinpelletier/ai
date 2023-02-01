import torch
import random
from collections import defaultdict

from ai.data.util import create_data_loader


class ReplayBuffer:
    def __init__(s,
        data_gen,
        device,
        batch_size,
        n_data_workers,
        buf_size,
        n_replay_times,
    ):
        s.generator = create_data_loader(
            data_gen,
            batch_size=None,
            device=device,
            n_workers=n_data_workers,
        )

        s.n_replay_times = n_replay_times
        s.batch_size = batch_size
        s.buf_size = max(buf_size, batch_size)
        s._buf = []

    def __iter__(s):
        return s

    def __next__(s):
        s._fill_buf()
        return s._create_batch()

    def remove_replay(s, replay):
        s._buf.remove(replay)

    def _fill_buf(s):
        while len(s._buf) < s.buf_size:
            s._buf.append(_Replay(s, next(s.generator)))

    def _create_batch(s):
        batch = defaultdict(list)
        for replay in random.sample(s._buf, s.batch_size):
            state = replay.sample_state()
            batch['ob'].append(state.ob)
            batch['pi'].append(state.pi)
            batch['v'].append(state.to_play * replay.outcome)
        return {k: torch.stack(vals).float() for k, vals in batch.items()}


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
        s.to_play = 1 if ply % 2 == 0 else -1 # TODO
