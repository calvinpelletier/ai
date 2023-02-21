import torch
import math
import numpy as np
from copy import deepcopy


class MctsConfig:
    def __init__(s,
        modeled_env,
        n_sims,
        intermediate_rewards=False,
        discount=None,
        noise_alpha=.25,
        noise_frac=.25,
        pb_c_base=19652,
        pb_c_init=1.25,
    ):
        s.modeled_env = modeled_env
        s.intermediate_rewards = intermediate_rewards
        s.discount = discount
        s.n_sims = n_sims
        s.noise_alpha = noise_alpha
        s.noise_frac = noise_frac
        s.pb_c_init = pb_c_init
        s.pb_c_base = pb_c_base


class MonteCarloTreeSearch:
    def __init__(s, cfg, player, value_bounds=None):
        s._cfg = cfg
        s._player = player
        s._normalizer = _Normalizer(value_bounds)
        s.root = None

    def run(s, game):
        s._init_root(game)
        for _ in range(s._cfg.n_sims):
            s._simulate(s._game_for_simulating(game))
        return s._policy()

    def _init_root(s, game):
        s.root = _Node(s._cfg, 0, game.to_play)
        s.root.expand(s._initial_infer(game), game.get_legal_actions())
        s.root.add_noise()

    def _simulate(s, game):
        line, outcome = s._traverse(game)
        if outcome is None:
            value = s._expand(game, line)
        else:
            value = outcome
        s._backprop(line, value)

    def _traverse(s, game):
        node = s.root
        line = [node]
        while node.expanded:
            action, node = s._select_child(node)
            game.step(action)
            line.append(node)
            if not s._cfg.modeled_env and game.outcome is not None:
                return line, game.outcome
        return line, None

    def _expand(s, game, line):
        if s._cfg.modeled_env:
            pred = s._player(state=line[-2].state, action=game.history[-1])
            actions = list(range(game.n_actions))
        else:
            pred = s._player(game.observe())
            actions = game.get_legal_actions()
            assert actions
        line[-1].expand(pred, actions)
        return pred['v'] * game.to_play

    def _backprop(s, line, value):
        for node in reversed(line):
            node.value_sum += value * node.to_play
            node.visit_count += 1
            s._normalizer.update(node.value)
            if s._cfg.intermediate_rewards:
                value = node.reward + s._cfg.discount * value

    def _select_child(s, node):
        action, child = max([
            (s._calc_ucb_score(node, child), action, child)
            for action, child in node.children.items()
        ], key=lambda x: x[0])[1:]
        return action, child

    def _calc_ucb_score(s, node, child):
        pb_c = math.log(
            (node.visit_count + s._cfg.pb_c_base + 1) / s._cfg.pb_c_base,
        ) + s._cfg.pb_c_init
        pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = s._normalizer(-child.value)
            if s._cfg.intermediate_rewards:
                value_score = child.reward + s._cfg.discount * value_score
        else:
            value_score = 0

        return prior_score + value_score

    def _policy(s):
        actions = np.empty(len(s.root.children), dtype=np.int32)
        counts = np.empty(len(s.root.children), dtype=np.float32)
        for i, (a, child) in enumerate(s.root.children.items()):
            actions[i] = a
            counts[i] = child.visit_count
        return actions, counts

    def _initial_infer(s, game):
        ob = game.observe()
        return s._player(ob=ob) if s._cfg.modeled_env else s._player(ob)

    def _game_for_simulating(s, game):
        return game.as_gameinfo() if s._cfg.modeled_env else deepcopy(game)


class _Node:
    def __init__(s, cfg, prior, to_play):
        s._cfg = cfg
        s.prior = prior
        s.to_play = to_play
        s.visit_count = 0
        s.value_sum = 0
        if s._cfg.intermediate_rewards:
            s.reward = 0

        # initialized when expanded
        s.children = None
        if s._cfg.modeled_env:
            s.state = None

    @property
    def expanded(s):
        return s.children is not None

    @property
    def value(s):
        if s.visit_count == 0:
            return 0
        return s.value_sum / s.visit_count

    def expand(s, pred, actions):
        if s._cfg.intermediate_rewards:
            s.reward = pred['r']
        if s._cfg.modeled_env:
            s.state = pred['state']

        pi = {a: math.exp(pred['pi'][a]) for a in actions}
        pi_sum = sum(pi.values())
        s.children = {
            a: _Node(s._cfg, p / pi_sum, -s.to_play) for a, p in pi.items()
        }

    def add_noise(s):
        actions = list(s.children.keys())
        noise = np.random.dirichlet([s._cfg.noise_alpha] * len(actions))
        frac = s._cfg.noise_frac
        for a, x in zip(actions, noise):
            s.children[a].prior = s.children[a].prior * (1 - frac) + x * frac


class _Normalizer:
    def __init__(s, known_bounds=None):
        s._has_known_bounds = known_bounds is not None
        if s._has_known_bounds:
            assert len(known_bounds) == 2
            assert known_bounds[0] < known_bounds[1]
            s._minimum = float(known_bounds[0])
            s._maximum = float(known_bounds[1])
        else:
            s._minimum = float('inf')
            s._maximum = -float('inf')

    def __call__(s, value: float) -> float:
        if s._maximum > s._minimum:
            return (value - s._minimum) / (s._maximum - s._minimum)
        return value

    def update(s, value: float):
        if s._has_known_bounds:
            assert s._minimum <= value <= s._maximum
        else:
            s._maximum = max(s._maximum, value)
            s._minimum = min(s._minimum, value)
