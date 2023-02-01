from redis import Redis
import math
import asyncio
import torch
from time import sleep
from collections import defaultdict

from ai.infer.common import QUEUES, encode, decode
from ai.util.timer import Timer


SLEEP = 0.1


def inference_worker(model, device, batch_size, redis_cfg, debug):
    worker = InferenceWorker(model, device, batch_size, redis_cfg, debug)
    worker.run()

class InferenceWorker:
    def __init__(s, model, device, batch_size, redis_cfg, debug):
        s._model = model.to(device).eval()

        s._batcher = _Batcher(batch_size, device)

        host, port, db = redis_cfg
        s._broker = Redis(host=host, port=port, db=db)

        s._debug = debug
        s._info = _Info() if debug else None

    def run(s):
        with torch.no_grad():
            while 1:
                s._respond_to_pings()
                s._respond_to_debugs()
                s._check_for_weights_update()
                s._run_batch()

    def _respond_to_pings(s):
        while 1:
            req_id = s._get_req(QUEUES.ping)
            if req_id is None:
                break
            s._send_resp(req_id, 'pong')

    def _respond_to_debugs(s):
        while 1:
            req_id = s._get_req(QUEUES.debug)
            if req_id is None:
                break
            assert s._debug
            s._send_resp(req_id, s._info.export())

    def _check_for_weights_update(s):
        weight_updates = s._get_reqs(QUEUES.update, 0, math.inf)
        if weight_updates:
            s._model.load_state_dict(weight_updates[-1])

    def _run_batch(s):
        # fetch
        reqs = s._get_reqs(QUEUES.infer, 0, s._batcher.batch_size)
        if not reqs:
            sleep(SLEEP)
            return
        if s._debug:
            s._info.update_bs_avg(len(reqs))

        # run
        ids, model_args, model_kwargs = s._batcher.batch(reqs)
        model_output = s._model(*model_args, **model_kwargs)
        resps = s._batcher.unbatch(ids, model_output)

        # respond
        for id, resp in zip(ids, resps):
            s._send_resp(id, resp)

    def _send_resp(s, id, resp):
        s._broker.set(id, encode(resp))

    def _get_req(s, queue, wait=False, timeout=None):
        req = (
            asyncio.run(_async_wait_for_req(s._broker, queue, timeout))
            if wait else s._broker.lpop(queue)
        )
        if req is not None:
            req = decode(req)
        return req

    def _get_reqs(s, queue, min_, max_):
        reqs = []

        # fetch immediately available requests up to max amount
        while len(reqs) < max_:
            req = s._get_req(queue)
            if req is None:
                break
            reqs.append(req)

        # wait for future requests until at min amount
        while len(reqs) < min_:
            reqs.append(s._get_req(queue, wait=True))

        return reqs

async def _async_wait_for_req(broker, queue, timeout=None):
    timer = Timer(timeout)
    while 1:
        req = broker.lpop(queue)
        if req is not None or timer():
            break
        await asyncio.sleep(SLEEP)
    return req


class _Batcher:
    def __init__(s, batch_size, device):
        s.batch_size = batch_size
        s._device = device

    def batch(s, reqs):
        assert len(reqs) <= s.batch_size

        if len(reqs) == 1:
            id, args, kwargs = reqs[0]
            args = [a.to(s._device) for a in args]
            kwargs = {k: v.to(s._device) for k, v in kwargs.items()}
            return [id], args, kwargs

        id_batch = []
        args_batch = []
        kwargs_batch = defaultdict(list)
        for i, (id, args, kwargs) in enumerate(reqs):
            id_batch.append(id)

            for j, a in enumerate(args):
                assert a.shape[0] == 1
                if i == 0:
                    args_batch.append([a])
                else:
                    args_batch[j].append(a)

            for k, v in kwargs:
                assert v.shape[0] == 1
                kwargs_batch[k].append(v)

        for i in range(len(args_batch)):
            args_batch[i] = torch.cat(args_batch[i], dim=0).to(s._device)
        for k in kwargs_batch:
            kwargs_batch[k] = torch.cat(kwargs_batch[k], dim=0).to(s._device)

        return id_batch, args_batch, kwargs_batch

    def unbatch(s, ids, output):
        bs = len(ids)

        if isinstance(output, dict):
            ret = [{} for _ in range(bs)]
            for k, v in output.items():
                v = v.cpu()
                assert v.shape[0] == bs
                for i in range(bs):
                    ret[i][k] = v[i]
            return ret

        assert torch.is_tensor(output)
        assert output.shape[0] == bs
        output = output.cpu()
        return [output[i] for i in range(bs)]


class _Info:
    def __init__(s):
        s._bs_sum = 0
        s._bs_count = 0

    def update_bs_avg(s, batch_size):
        s._bs_sum += batch_size
        s._bs_count += 1

    def export(s):
        return {
            'avg_batch_size': s._get_avg_bs(),
        }

    def _get_avg_bs(s):
        if s._bs_count > 0:
            return s._bs_sum / s._bs_count
        return None
