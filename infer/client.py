import asyncio
from redis import Redis
from uuid import UUID
from typing import Optional, Union, List

from ai.util import gen_uuid
from ai.util.timer import Timer
from ai.infer.common import QUEUES, encode, decode


SLEEP = 0.1


class InferencerUnresponsive(Exception):
    '''Raised by InferenceClient.ping() after n seconds without a response.'''


class InferenceClient:
    '''Client of an inference worker.

    The __call__ method can be used as if it is the model, and
    it will send a request to the worker. For example:
    ```
    z = model(x, y)
    ```
    is the same as
    ```
    z = inference_client(x, y)
    ```

    NOTE: should only be created by an Inferencer.

    NOTE: InferencerClient sets up the Redis client JIT because it can't be
    pickled. So don't use the InferencerClient until after it's been sent to the
    data worker.

    NOTE: if you have multiple requests, it's much faster to use the multi_infer
    method instead of using __call__ multiple times in succession.
    '''

    def __init__(s, redis_cfg: Union[list, tuple]):
        '''
        redis_cfg : list/tuple of length 3
            redis host (str), redis port (int), db number (int)
        '''

        assert len(redis_cfg) == 3
        s._redis_cfg = redis_cfg
        s._broker = None # redis client initialized JIT because of pickle

    def __call__(s, *args, **kwargs):
        '''Run inference request.'''

        return s.infer(*args, **kwargs)

    def ping(s, timeout: int = 5):
        '''Raise InferencerUnresponsive exception if inferencer isnt running.

        timeout : int
            timeout in seconds
        '''

        req_id = gen_uuid()
        s._add_to_queue(QUEUES.ping, req_id)
        resp = s.wait_for_resp(req_id, timeout=timeout)
        if resp != 'pong':
            raise InferencerUnresponsive()

    def debug(s) -> dict:
        '''Fetch debug information from the worker.'''

        req_id = gen_uuid()
        s._add_to_queue(QUEUES.debug, req_id)
        return s.wait_for_resp(req_id)

    def update_params(s, params: dict):
        '''Update the parameters of the model.

        params : dict
            model.state_dict()
        '''

        s._add_to_queue(QUEUES.update, params)

    def infer(s, *args, **kwargs):
        '''Run inference request.'''

        req_id = s.infer_async(*args, **kwargs)
        return s.wait_for_resp(req_id)

    def multi_infer(s, reqs):
        '''Bundle multiple inference requests.

        reqs : list of tuples
            each item is a tuple of (args, kwargs) where args is a list of
            arguments and kwargs is a dict of keyword arguments
        '''

        req_ids = s.multi_infer_async(reqs)
        return s.wait_for_resps(req_ids)

    def infer_async(s, *args, **kwargs) -> UUID:
        '''Send inference request and return request id.

        Use InferenceClient.wait_for_resp(request_id) to get response.
        '''

        req_id = gen_uuid()
        s._add_to_queue(QUEUES.infer, (req_id, args, kwargs))
        return req_id

    def multi_infer_async(s, reqs: List[tuple]) -> List[UUID]:
        '''Same as infer_async but for multiple requests.

        Use InferenceClient.wait_for_resps(request_ids) to get response.

        reqs : list of tuples
            each item is a tuple of (args, kwargs) where args is a list of
            arguments and kwargs is a dict of keyword arguments
        '''

        req_ids = []
        for args, kwargs in reqs:
            req_id = gen_uuid()
            s._add_to_queue(QUEUES.infer, (req_id, args, kwargs))
            req_ids.append(req_id)
        return req_ids

    def wait_for_resp(s, req_id: UUID, timeout: Optional[int] = None):
        '''Wait for a response to be available for the given request id.

        req_id : uuid
            the request id returned from infer_async
        timeout : int or null
            optional timeout in seconds (returns None on timeout)
        '''

        resp = asyncio.run(_async_wait_for_resp(s._broker, req_id, timeout))
        return decode(resp)

    def wait_for_resps(s, req_ids: List[UUID], timeout: Optional[int] = None):
        '''Same as wait_for_resp but for multiple requests.'''

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
