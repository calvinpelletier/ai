

def build(type_to_cls, c, cfg):
    return type_to_cls[cfg.type](c, cfg)
