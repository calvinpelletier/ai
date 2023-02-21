import multiprocessing as mp

from ai.util.worker import kill_worker


class DataQueue:
    def __init__(s, dg, n_workers, max_size=32):
        s._q = mp.Queue(max_size)

        s._workers = []
        for _ in range(n_workers):
            p = mp.Process(target=_worker_fn, args=(s._q, dg))
            p.start()
            s._workers.append(p)

    def __iter__(s):
        return s

    def __next__(s):
        return s._q.get(block=True, timeout=None)

    def __del__(s):
        for worker in s._workers:
            kill_worker(worker)


def _worker_fn(q, dg):
    for data in dg:
        q.put(data, block=True, timeout=None)
