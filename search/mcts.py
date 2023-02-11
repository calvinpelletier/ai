import torch
import math
import numpy as np
from copy import deepcopy

from ai.util import softmax_sample_idx


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


class MctsAgent:
    def __init__(s, game, model, cfg):
        s._cfg = cfg
        s._model = model
        s._outcome_bounds = game.outcome_bounds
        s._setup_mcts()

    @property
    def model(s):
        return s._model

    @model.setter
    def model(s, model):
        s._model = model
        s._setup_mcts()

    def act(s, game, return_pi=False):
        with torch.no_grad():
            actions, counts = s._mcts.run(game)

        i = softmax_sample_idx(
            counts,
            s._temperature(game.ply),
        )
        action = actions[i]

        if return_pi:
            pi = np.zeros(game.n_actions)
            for action, count in zip(actions, counts):
                pi[action] = count
            return action, pi
        return action

    def _temperature(s, ply):
        raise NotImplementedError()

    def _setup_mcts(s):
        s._mcts = MonteCarloTreeSearch(s._cfg, s._model, s._outcome_bounds)


class MonteCarloTreeSearch:
    def __init__(s, cfg, model, value_bounds=None):
        s.cfg = cfg
        s.model = s._wrap_model(model, value_bounds)
        s.normalizer = _Normalizer(value_bounds)
        s.root = None

    def run(s, game):
        s._init_root(game)
        for _ in range(s.cfg.n_sims):
            s._simulate(s._game_for_simulating(game))
        return s._policy()

    def _init_root(s, game):
        s.root = _Node(s.cfg, 0)
        s.root.expand(
            s._initial_inference(game),
            game.to_play,
            game.get_legal_actions(),
        )
        s.root.add_noise()

    def _simulate(s, game):
        line, outcome = s._traverse(game)
        if outcome is None:
            value = s._expand(game, line)
        else:
            value = outcome # TODO: mult by to_play?
        s._backprop(line, value, game.to_play)

    def _traverse(s, game):
        node = s.root
        line = [node]
        while node.expanded:
            action, node = s._select_child(node)
            game.step(action)
            line.append(node)
            if not s.cfg.modeled_env and game.outcome is not None:
                return line, game.outcome
        return line, None

    def _expand(s, game, line):
        if s.cfg.modeled_env:
            pred = s.model(state=line[-2].hidden_state, action=game.history[-1])
            actions = list(range(game.n_actions))
        else:
            pred = s.model(game.observe())
            actions = game.get_legal_actions()
            assert actions
        line[-1].expand(pred, game.to_play, actions)
        return pred['v']

    def _backprop(s, line, value, to_play):
        for node in reversed(line):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            s.normalizer.update(node.value)
            if s.cfg.intermediate_rewards:
                value = node.reward + s.cfg.discount * value

    def _select_child(s, node):
        action, child = max([
            (s._calc_ucb_score(node, child), action, child)
            for action, child in node.children.items()
        ], key=lambda x: x[0])[1:]
        return action, child

    def _calc_ucb_score(s, node, child):
        pb_c = math.log(
            (node.visit_count + s.cfg.pb_c_base + 1) / s.cfg.pb_c_base,
        ) + s.cfg.pb_c_init
        pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
        prior_score = pb_c * child.prior

        if child.visit_count > 0:
            value_score = s.normalizer(child.value)
            if s.cfg.intermediate_rewards:
                value_score = child.reward + s.cfg.discount * value_score
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

    def _initial_inference(s, game):
        ob = game.observe()
        return s.model(ob=ob) if s.cfg.modeled_env else s.model(ob)

    def _game_for_simulating(s, game):
        return game.as_gameinfo() if s.cfg.modeled_env else deepcopy(game)

    def _wrap_model(s, model, bounds):
        cls = _ModelWrapperME if s.cfg.modeled_env else _ModelWrapperMFE
        return cls(model, bounds)


class _Node:
    def __init__(s, cfg, prior):
        s.cfg = cfg
        s.prior = prior
        s.visit_count = 0
        s.value_sum = 0
        if s.cfg.intermediate_rewards:
            s.reward = 0

        # initialized when expanded
        s.children = None
        s.to_play = None
        if s.cfg.modeled_env:
            s.hidden_state = None

    @property
    def expanded(s):
        return s.children is not None

    @property
    def value(s):
        if s.visit_count == 0:
            return 0
        return s.value_sum / s.visit_count

    def expand(s, pred, to_play, actions):
        s.to_play = to_play
        if s.cfg.intermediate_rewards:
            s.reward = pred['r']
        if s.cfg.modeled_env:
            s.hidden_state = pred['state']

        pi = {a: math.exp(pred['pi'][a]) for a in actions}
        pi_sum = sum(pi.values())
        s.children = {a: _Node(s.cfg, p / pi_sum) for a, p in pi.items()}

    def add_noise(s):
        actions = list(s.children.keys())
        noise = np.random.dirichlet([s.cfg.noise_alpha] * len(actions))
        frac = s.cfg.noise_frac
        for a, x in zip(actions, noise):
            s.children[a].prior = s.children[a].prior * (1 - frac) + x * frac


class _ModelWrapper:
    def __init__(s, model, bounds):
        s.model = model

        s.bounds = None
        if bounds is not None:
            a, b = [float(x) for x in bounds]
            if a != -1. or b != 1.:
                s.bounds = (a, b)

    def _normalize(s, x):
        if s.bounds is None:
            return x
        a, b = s.bounds
        return a + (b - a) * (x + 1) / 2

class _ModelWrapperMFE(_ModelWrapper):
    def __call__(s, ob):
        ob = torch.from_numpy(ob).float().unsqueeze(0)
        pred = s.model(ob)
        return {
            'pi': pred['pi'].squeeze(0).numpy(),
            'v': s._normalize(pred['v'].squeeze(0).item()),
        }

class _ModelWrapperME(_ModelWrapper):
    def __call__(s, ob=None, state=None, action=None):
        if ob is not None:
            # initial inference
            assert state is None and action is None
            print(ob)
            ob = torch.from_numpy(ob).float().unsqueeze(0)
            pred = s.model(ob=ob)
        else:
            # recurrent inference
            assert state is not None and action is not None
            state = state.unsqueeze(0)
            action = torch.tensor(action, dtype=torch.long).unsqueeze(0)
            pred = s.model(state=state, action=action)

        return {
            'pi': pred['pi'].squeeze(0).numpy(),
            'v': s._normalize(pred['v'].squeeze(0).item()),
            'r': s._normalize(pred['r'].squeeze(0).item()),
            'state': pred['state'].squeeze(0),
        }


class _Normalizer:
    def __init__(s, known_bounds=None):
        s.has_known_bounds = known_bounds is not None
        if s.has_known_bounds:
            assert len(known_bounds) == 2
            assert known_bounds[0] < known_bounds[1]
            s.minimum = float(known_bounds[0])
            s.maximum = float(known_bounds[1])
        else:
            s.minimum = float('inf')
            s.maximum = -float('inf')

    def __call__(s, value: float) -> float:
        if s.maximum > s.minimum:
            return (value - s.minimum) / (s.maximum - s.minimum)
        return value

    def update(s, value: float):
        if s.has_known_bounds:
            assert s.minimum <= value <= s.maximum
        else:
            s.maximum = max(s.maximum, value)
            s.minimum = min(s.minimum, value)
