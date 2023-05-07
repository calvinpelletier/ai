from pathlib import Path

import ai
from ai.examples.alphazero import AlphaZeroMLP


def run(cfg, game, model):
    model.init().train().to(cfg.device)
    player = ai.game.MctsPlayer(cfg.player, game, model)
    task = ai.task.GameTask(game, player, cfg.task.n_matches)
    trial = ai.Trial(cfg.outpath, task=task, clean=True)

    ai.Trainer(
        env=ai.train.RL(cfg.loss.v_weight),
        data=ai.data.SelfPlay.from_cfg(cfg, game, player),
    ).train(
        model,
        ai.opt.build(cfg.opt, model),
        trial.hook(),
        steplimit=cfg.train.steplimit,
        timelimit=cfg.train.timelimit,
    )

    return task()


def example_config():
    return ai.Config.load(Path(__file__).parent / 'config.yaml')


if __name__ == '__main__':
    cfg = example_config()
    game = ai.game.TicTacToe()
    model = AlphaZeroMLP(game)
    run(cfg, game, model)
