import numpy as np

from ai.util.path import dataset_path


def create_subset(inpath, outpath, n_games):
    games = np.load(inpath / 'games.npy')
    actions = np.load(inpath / 'actions.npy')
    times = np.load(inpath / 'times.npy')

    assert len(actions) == len(times)
    games = games[:n_games]
    start = games[0, 0]
    end = games[-1, 1]
    assert start == 0
    actions = actions[:end]
    times = times[:end]
    print(len(games), len(actions), len(times))

    np.save(outpath / 'game.npy', games)
    np.save(outpath / 'action.npy', actions)
    np.save(outpath / 'time.npy', times)


if __name__ == '__main__':
    create_subset(
        dataset_path('chess/lichess/2023-01/0'),
        dataset_path('chess/test/10k'),
        10_000,
    )
