from redis import Redis

from ai.worker import launch_worker, kill_worker
from ai.infer.worker import inference_worker
from ai.infer.client import InferenceClient


class Inferencer:
    def __init__(s,
        model,
        device='cuda',
        batch_size=1,
        redis_cfg=('127.0.0.1', 6379, 0),
        debug=False,
    ):
        host, port, db = redis_cfg
        s._broker = Redis(host=host, port=port, db=db)
        s._broker.flushdb()

        s._worker = launch_worker(inference_worker,
            model, device, batch_size, redis_cfg, debug)

        s._client = InferenceClient(redis_cfg)
        s._client.ping()

        s._redis_cfg = redis_cfg

    def create_client(s):
        return InferenceClient(s._redis_cfg)

    def __del__(s):
        kill_worker(s._worker)
        s._broker.flushdb()

    def __call__(s, *args, **kwargs):
        return s._client.infer(*args, **kwargs)

    def multi_infer(s, reqs):
        return s._client.multi_infer(reqs)

    def update_weights(s, weights):
        s._client.update_weights(weights)

    def debug(s):
        return s._client.debug()
