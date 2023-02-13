from redis import Redis

from ai.worker import launch_worker, kill_worker
from ai.infer.worker import inference_worker
from ai.infer.client import InferenceClient
from ai.model import Model


class Inferencer:
    '''Remote model inferencer.

    Launches a worker in another process which pulls batches of inference
    requests from an intermediate message broker (Redis) and runs them through
    a model.

    The __call__ method of the inferencer can be used as if it is the model, and
    it will send a request to the worker. For example:
    ```
    z = model(x, y)
    ```
    is the same as
    ```
    inferencer = Inferencer(model)
    z = inferencer(x, y)
    ```

    You can also use Inferencer.create_client() to create an InferencerClient
    object. Unlike Inferencer, it doesn't hold a reference to the worker so it
    can be safely passed around to other processes.

    A common use case is reinforcement learning, where data is being generated
    by the model while it is simultaneously being trained on that data. In this
    case, the data workers will each have an InferencerClient and the trainer
    will periodically call Inferencer.update_weights() to refresh the model.

    NOTE: InferencerClient sets up the Redis client JIT because it can't be
    pickled. So don't use the InferencerClient until after it's been sent to the
    data worker.

    NOTE: if you have multiple requests, it's much faster to use the multi_infer
    method instead of using __call__ multiple times in succession.
    '''

    def __init__(s,
        model: Model,
        device: str = 'cuda',
        batch_size: int = 1,
        redis_cfg: list | tuple = ('127.0.0.1', 6379, 0),
        debug: bool = False,
    ):
        '''
        model : ai.model.Model
        device : str
            the device that the inference worker will keep the model on
        batch_size : int
            max size of inference batches pulled by the worker
        redis_cfg : list/tuple of length 3
            redis host (str), redis port (int), db number (int)
        debug : bool
            if true, the worker will collect stats such as average batch size,
            which can be fetched via Inferencer.debug()
        '''

        # setup broker
        host, port, db = redis_cfg
        s._broker = Redis(host=host, port=port, db=db)
        s._broker.flushdb()

        # launch worker
        s._worker = launch_worker(inference_worker,
            model, device, batch_size, redis_cfg, debug)

        # create client and test that it's working
        s._client = InferenceClient(redis_cfg)
        s._client.ping()

        s._redis_cfg = redis_cfg

    def __del__(s):
        kill_worker(s._worker)
        s._broker.flushdb()

    def create_client(s) -> InferenceClient:
        '''Create an InferencerClient object.'''

        return InferenceClient(s._redis_cfg)

    def __call__(s, *args, **kwargs):
        '''Run an inference request (like calling the model directly)'''

        return s._client.infer(*args, **kwargs)

    def multi_infer(s, reqs: list | tuple):
        '''Bundle multiple inference requests.'''

        return s._client.multi_infer(reqs)

    def update_weights(s, weights):
        '''Update the parameters of the model.'''

        s._client.update_weights(weights)

    def debug(s):
        '''Fetch debug information from the worker.'''

        return s._client.debug()
