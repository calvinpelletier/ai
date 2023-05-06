from collections import deque


class EarlyStopper:
    def __init__(s, improvement=0.03, patience=3):
        s._improvement = improvement
        s._patience = patience
        s._vals = deque()

    def __call__(s, step, val_loss):
        s._vals.append(val_loss)
        if len(s._vals) < s._patience:
            return False

        if len(s._vals) > s._patience:
            s._vals.popleft()

        x = (s._vals[0] - s._vals[-1]) / s._vals[0]
        stop = x < s._improvement
        if stop:
            print('[INFO] early stopping (improvement {:.2f} < {:.2f})'.format(
                x,
                s._improvement,
            ))
        return stop
