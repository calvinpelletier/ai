from ai.game.player import RandomPlayer, PLAYER1, PLAYER2
from ai.util import no_op


class GameTask:
    def __init__(s, game, log=no_op, n_matches=32):
        assert game.n_players == 2, 'TODO: non-2-player games'
        s._game = game
        s._log = log
        s._n_matches = 32

    def __call__(s, player, step=None):
        opp = RandomPlayer()

        p1_stats = _Stats(PLAYER1)
        p2_stats = _Stats(PLAYER2)
        for i in range(s._n_matches):
            if i < s._n_matches // 2:
                p1, p2 = player, opp
                stats = p1_stats
            else:
                p1, p2 = opp, player
                stats = p2_stats

            outcome = _play(s._game, p1, p2)
            stats.update(outcome)

        win_rate = (p1_stats.wins + p2_stats.wins) / s._n_matches
        loss_rate = (p1_stats.losses + p2_stats.losses) / s._n_matches
        draw_rate = (p1_stats.draws + p2_stats.draws) / s._n_matches
        s._log(step, 'task.win_rate', win_rate)
        s._log(step, 'task.loss_rate', loss_rate)
        s._log(step, 'task.draw_rate', draw_rate)

        s._log(step, 'task.p1.win_rate', p1_stats.wins / p1_stats.total)
        s._log(step, 'task.p1.loss_rate', p1_stats.losses / p1_stats.total)
        s._log(step, 'task.p1.draw_rate', p1_stats.draws / p1_stats.total)

        s._log(step, 'task.p2.win_rate', p2_stats.wins / p2_stats.total)
        s._log(step, 'task.p2.loss_rate', p2_stats.losses / p2_stats.total)
        s._log(step, 'task.p2.draw_rate', p2_stats.draws / p2_stats.total)

        return win_rate


def _play(game, p1, p2):
    game.reset()
    while game.outcome is None:
        player = p1 if game.to_play == 1 else p2
        action, _ = player.act(game)
        game.step(action)
    return game.outcome


class _Stats:
    def __init__(s, player):
        s._player = player

        s.wins = 0
        s.losses = 0
        s.draws = 0
        s.total = 0

    def update(s, outcome):
        s.total += 1
        if outcome == 0:
            s.draws += 1
        elif outcome == s._player:
            s.wins += 1
        else:
            s.losses += 1
