from copy import deepcopy

from ai.game.player import RandomPlayer
from ai.util import no_op


class GameTask:
    def __init__(s, game, player, n_matches=256):
        assert game.n_players == 2, 'TODO: non-2-player games'
        s._game = deepcopy(game)
        s._player = player
        s._n_matches = n_matches

    def __call__(s, _model=None, log=no_op):
        win_rate_vs_random, _, _ = s._run('vs_random', RandomPlayer(), log)
        _, _, draw_rate_vs_self = s._run('vs_self', s._player, log)
        return win_rate_vs_random, draw_rate_vs_self

    def _run(s, prefix, opp, log):
        p1_stats = _Stats(s._game.PLAYER1)
        p2_stats = _Stats(s._game.PLAYER2)
        for i in range(s._n_matches):
            if i < s._n_matches // 2:
                p1, p2 = s._player, opp
                stats = p1_stats
            else:
                p1, p2 = opp, s._player
                stats = p2_stats
            stats.update(s._play(p1, p2))

        win_rate = (p1_stats.wins + p2_stats.wins) / s._n_matches
        loss_rate = (p1_stats.losses + p2_stats.losses) / s._n_matches
        draw_rate = (p1_stats.draws + p2_stats.draws) / s._n_matches
        log(f'{prefix}.win_rate', win_rate)
        log(f'{prefix}.loss_rate', loss_rate)
        log(f'{prefix}.draw_rate', draw_rate)

        log(f'{prefix}.p1.win_rate', p1_stats.wins / p1_stats.total)
        log(f'{prefix}.p1.loss_rate', p1_stats.losses / p1_stats.total)
        log(f'{prefix}.p1.draw_rate', p1_stats.draws / p1_stats.total)

        log(f'{prefix}.p2.win_rate', p2_stats.wins / p2_stats.total)
        log(f'{prefix}.p2.loss_rate', p2_stats.losses / p2_stats.total)
        log(f'{prefix}.p2.draw_rate', p2_stats.draws / p2_stats.total)

        return win_rate, loss_rate, draw_rate

    def _play(s, p1, p2):
        s._game.reset()
        while s._game.outcome is None:
            player = p1 if s._game.to_play == 1 else p2
            action = player.act(s._game, greedy=True)
            s._game.step(action)
        return s._game.outcome


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
