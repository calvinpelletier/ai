import ai
from ai.examples.alphazero import run, ExampleConfig, AlphaZeroMLP


exp = ai.Experiment('/tmp/a0/hps', direction='max')


class CLI:
    def run(s, n):
        cfg = ExampleConfig()
        cfg.train.steplimit = 5000

        game = ai.game.TicTacToe()

        model = AlphaZeroMLP(game)

        exp.run(n, lambda trial: _run_trial(cfg, game, model, trial))

    def show(s, hp_name):
        exp.show_plot(hp_name)

    def clean(s):
        exp.clean()


def _run_trial(cfg, game, model, trial):
    hp = trial.hp

    cfg.model_update_interval = hp.log('mui', 10, 1000)
    cfg.opt.lr = hp.log('lr', 1e-4, 1e-2)
    cfg.loss.v_weight = hp.log('vw', .01, 10)
    cfg.data.n_replay_times = hp.lin('nrt', 1, 8)
    cfg.player.mcts.n_sims = hp.log('ns', 4, 16)

    win_rate_vs_random, draw_rate_vs_self = run(cfg, game, model)
    return win_rate_vs_random


if __name__ == '__main__':
    ai.run(CLI)
