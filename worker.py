import multiprocessing as mp


def launch_worker(entry, *args, **kwargs):
    mp.set_start_method('spawn', force=True)
    p = mp.Process(target=entry, args=args, kwargs=kwargs)
    p.start()
    return p

def kill_worker(p):
    p.terminate()
    p.join()
