import random


PLAYER1 = 1
PLAYER2 = -1


class RandomPlayer:
    def act(s, game, return_pi=False):
        legal = game.get_legal_actions()
        action = random.choice(legal)
        if return_pi:
            pi = np.zeros(game.n_actions)
            for a in legal:
                pi[a] = 1 / len(legal)
            return action, pi
        return action
