

def on_interval(i: int, interval: int) -> bool:
    return interval is not None and i % interval == 0
