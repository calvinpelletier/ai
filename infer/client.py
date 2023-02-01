import asyncio
from redis import Redis

from ai.util import gen_uuid
from ai.util.timer import Timer
from ai.infer.common import QUEUES, encode, decode


SLEEP = 0.1


class InferencerUnresponsive(Exception):
    pass

class InferenceClient:
    def __init__(s, redis_cfg):
        s._redis_cfg = redis_cfg
        s._broker = None # redis client initialized JIT because of pickle

    def __call__(s, *args, **kwargs):
        return s.infer(*args, **kwargs)

    def ping(s):
        req_id = gen_uuid()
        s._add_to_queue(QUEUES.ping, req_id)
        resp = s.wait_for_resp(req_id, timeout=5)
        if resp != 'pong':
            raise InferencerUnresponsive()

    def debug(s):
        req_id = gen_uuid()
        s._add_to_queue(QUEUES.debug, req_id)
        return s.wait_for_resp(req_id)

    def update_weights(s, weights):
        s._add_to_queue(QUEUES.update, weights)

    def infer(s, *args, **kwargs):
        req_id = s.infer_async(*args, **kwargs)
        return s.wait_for_resp(req_id)

    def multi_infer(s, reqs):
        req_ids = s.multi_infer_async(reqs)
        return s.wait_for_resps(req_ids)

    def infer_async(s, *args, **kwargs):
        req_id = gen_uuid()
        s._add_to_queue(QUEUES.infer, (req_id, args, kwargs))
        return req_id

    def multi_infer_async(s, reqs):
        req_ids = []
        for args, kwargs in reqs:
            req_id = gen_uuid()
            s._add_to_queue(QUEUES.infer, (req_id, args, kwargs))
            req_ids.append(req_id)
        return req_ids

    def wait_for_resp(s, req_id, timeout=None):
        resp = asyncio.run(_async_wait_for_resp(s._broker, req_id, timeout))
        return decode(resp)

    def wait_for_resps(s, req_ids, timeout=None):
        resps = asyncio.run(_async_wait_for_resps(s._broker, req_ids, timeout))
        return [decode(resp) for resp in resps]

    def _add_to_queue(s, queue, req):
        if s._broker is None:
            s._setup_broker()
        s._broker.rpush(queue, encode(req))

    def _setup_broker(s):
        host, port, db = s._redis_cfg
        s._broker = Redis(host=host, port=port, db=db)

async def _async_wait_for_resps(broker, keys, timeout=None):
    return await asyncio.gather(*[
        _async_wait_for_resp(broker, key, timeout) for key in keys])

async def _async_wait_for_resp(broker, key, timeout=None):
    timer = Timer(timeout)
    while 1:
        resp = broker.get(key)
        if resp is not None or timer():
            break
        await asyncio.sleep(SLEEP)
    return resp
