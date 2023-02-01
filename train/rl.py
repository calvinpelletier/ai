from copy import deepcopy

from ai.train.trainer import Trainer
from ai.infer import Inferencer
from ai.data.buffer import ReplayBuffer


class RLTrainer(Trainer):
    def __init__(s, cfg, env, agent, data_generator):
        s._refresh_interval = cfg.train.refresh

        # create inferencer
        inferencer = Inferencer(
            agent.model,
            device=cfg.inference.device,
            batch_size=cfg.inference.bs,
        )
        data_generator.agent = deepcopy(agent)
        data_generator.agent.model = inferencer.create_client()

        # create data iterator
        data = ReplayBuffer(
            data_generator,
            device=cfg.train.device,
            batch_size=cfg.train.bs,
            n_data_workers=cfg.data.n_workers,
            buf_size=cfg.data.buf_size,
            n_replay_times=cfg.data.n_replay_times,
        )

        super().__init__(env, data)

        s._inferencer = inferencer

    def _post_step(s, step, model, opt, hook):
        if step > 0 and step % s._refresh_interval == 0:
            s._inferencer.update_weights(model.state_dict())

        return super()._post_step(step, model, opt, hook)
