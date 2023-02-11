

class Env:
    def __init__():
        pass
    def __call__(model, batch, step):
        train_log(x)
        return loss

class Trainer:
    def __init__(env, train_data):
        pass
    def train(model, opt):
        hook.setup(model, opt)
        stop = hook.step(step, done)
        loss = env(model, batch)
        train_log(loss)
        return step
    def validate(model, val_data):
        loss = env(model, batch)
        val_log(avg_loss)
        return avg_loss

class HookConfig:
    log=(fn,sch) # _special_
    save=(fn,sch) # sch:fn(_step,_model,_opt)
    validate=(sch,stopper) # sch:_validate(_model,_data)->stopper?
    task=(fn,sch,stopper) # sch:fn(_model%,_data?)->stopper?
    sample=(fn,sch) # sch:fn(_model)
class Hook:
    def __init__(cfg, val_data):
        pass
    def setup(model, opt, validate):
        pass
    def step(step, done):
        # train log enable/disable
        save(model, opt)
        val_loss = validate(model, val_data)
        return stopper(val_loss)

    def train_log(k, v):
        # check enabled
        log(step, 'train.'+k, v)
    def val_log(k, v):
        log(step, 'val.'+k, v)
    def task_log(k, v):
        log(step, 'task.'+k, v)

class Trial:
    def __init__(path, hook_cfg):
        log
    def hook(val_data):
        return Hook(cfg, val_data)
    def save(step, model, opt):
        pass
