import ai


def build_trainer(cfg, game, player):
    return ai.train.RLTrainer(
        cfg,
        AlphaZeroEnv(cfg.train.loss.v_weight),
        player,
        ai.data.SelfPlay(game),
    )


class AlphaZeroEnv(ai.train.Env):
    def __init__(s, v_weight=1.):
        super().__init__()
        s._pi_loss_fn = ai.loss.CrossEntropyLoss()
        s._v_loss_fn = ai.loss.L2Loss()
        s._v_weight = v_weight

    def __call__(s, model, batch, step=0):
        pred = model(batch['ob'])

        pi_loss = s._pi_loss_fn(pred['pi'], batch['pi'])
        s.log('loss.policy', pi_loss)

        v_loss = s._v_loss_fn(pred['v'], batch['v'])
        s.log('loss.value', v_loss)

        return pi_loss + v_loss * s._v_weight
