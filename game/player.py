import random
import numpy as np
import torch

from ai.game.mcts import MonteCarloTreeSearch
from ai.util import softmax_sample_idx
from ai.config import Config
from ai.game.game import Game
from ai.model import Model


class Player:
    def act(s, game: Game, return_pi: bool = False, greedy: bool = False):
        raise NotImplementedError()


class RandomPlayer(Player):
    def act(s, game: Game, return_pi: bool = False, greedy: bool = False):
        legal = game.get_legal_actions()
        action = random.choice(legal)
        if return_pi:
            pi = np.zeros(game.n_actions)
            for a in legal:
                pi[a] = 1 / len(legal)
            return action, pi
        return action


class MctsPlayer(Player):
    def __init__(s, cfg: Config, game: Game, model: Model):
        s._cfg = cfg
        s.model = model
        s.device = model.get_device()

        s._v_bounds = [float(x) for x in game.outcome_bounds]
        assert len(s._v_bounds) == 2
        s._needs_v_normalize = s._v_bounds[0] != -1. or s._v_bounds[1] != 1.

        s._mcts = MonteCarloTreeSearch(s._cfg.mcts, s, s._v_bounds)

    def __call__(s, *a, **kw) -> dict:
        if s._cfg.mcts.modeled_env:
            return s._pred_me(*a, **kw)
        else:
            return s._pred_mfe(*a, **kw)

    def act(s, game: Game, return_pi: bool = False, greedy: bool = False):
        with torch.no_grad():
            actions, counts = s._mcts.run(game)

        temperature = 0. if greedy else s._temperature(game.ply)
        action = _choose_action(actions, counts, temperature)

        if return_pi:
            return action, _get_policy(actions, counts, game.n_actions)
        return action

    def _pred_mfe(s, ob):
        ob = torch.from_numpy(ob).float().unsqueeze(0).to(s.device)
        with torch.no_grad():
            pred = s.model(ob)
        return {
            'pi': pred['pi'].squeeze(0).cpu().numpy(),
            'v': s._v_normalize(pred['v'].squeeze(0).item()),
        }

    def _pred_me(s, ob=None, state=None, action=None):
        if ob is not None:
            # initial inference
            assert state is None and action is None
            ob = torch.from_numpy(ob).float().unsqueeze(0).to(s.device)
            with torch.no_grad():
                pred = s.model(ob=ob)
        else:
            # recurrent inference
            assert state is not None and action is not None
            state = state.unsqueeze(0)
            action = torch.tensor(
                action,
                dtype=torch.long,
                device=s.device,
            ).unsqueeze(0)
            with torch.no_grad():
                pred = s.model(state=state, action=action)

        return {
            'pi': pred['pi'].squeeze(0).cpu().numpy(),
            'v': s._v_normalize(pred['v'].squeeze(0).item()),
            'r': s._v_normalize(pred['r'].squeeze(0).item()),
            'state': pred['state'].squeeze(0),
        }

    def _temperature(s, ply):
        greedy = s._cfg.greedy_ply
        return 1. if greedy is None or ply < greedy else 0.

    def _v_normalize(s, v):
        if not s._needs_v_normalize:
            return v
        a, b = s._v_bounds
        return a + (b - a) * (v + 1) / 2

def _choose_action(actions, counts, t):
    if t == 0.:
        return actions[np.argmax(counts)]

    if t == float('inf'):
        return np.random.choice(actions)

    p = counts ** (1. / t)
    p /= sum(p)
    return np.random.choice(actions, p=p)

def _get_policy(actions, counts, n_actions):
    pi = np.zeros(n_actions, dtype=np.float32)
    for a, count in zip(actions, counts):
        pi[a] = count
    return pi / np.sum(pi)
