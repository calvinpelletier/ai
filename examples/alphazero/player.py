from ai.search import mcts
from ai import Config


class AlphaZeroPlayer(mcts.MctsAgent):
    TEST = Config({
        'n_mcts_sims': 8,
        'greedy_ply': 32,
    })

    def __init__(s, cfg, game, model):
        super().__init__(game, model, mcts.MctsConfig(
            modeled_env=False,
            n_sims=cfg.n_mcts_sims,
            intermediate_rewards=False,
            discount=None,
        ))

        s._greedy_ply = cfg.greedy_ply

    def _temperature(s, ply):
        return 1. if ply < s._greedy_ply else 0.
