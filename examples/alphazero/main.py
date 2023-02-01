from pathlib import Path

import ai
import ai.examples.alphazero as a0


BASE_CONFIG_PATH = Path(__file__).parent / 'config.yaml'


class CLI:
    def tictactoe(s, output_path, **kw):
        run(ai.game.TicTacToe(), output_path, kw)


def run(game, output_path, cfg_override):
    cfg = ai.Config(BASE_CONFIG_PATH, cfg_override)
    print(cfg)

    trial = ai.lab.Trial(output_path, clean=True)
    print(f'path: {trial.path}\n')

    model = a0.AlphaZeroMLP(game).init().to(cfg.train.device)
    player = a0.AlphaZeroPlayer(cfg.player, game, model)

    trainer = a0.build_trainer(cfg, game, player)
    trainer.train(model, ai.opt.build(cfg.opt, model), trial.hook())


if __name__ == '__main__':
    ai.fire(CLI)
