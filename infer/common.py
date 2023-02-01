from collections import namedtuple
import pickle


QUEUES = namedtuple('_', 'ping infer update debug')(
    'ping_queue', 'infer_queue', 'update_queue', 'debug_queue')


def encode(obj):
    return pickle.dumps(obj)

def decode(obj):
    if obj is None:
        return None
    return pickle.loads(obj)
