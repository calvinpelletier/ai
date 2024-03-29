from ai.lab.base import LabEntity
from ai.lab.trial import Trial
from ai.lab.exp import Experiment


class Study(LabEntity):
    def __init__(s, path, clean=False):
        super().__init__(path, clean)

    def trial(s, name, *a, **kw):
        return Trial(s.path / name, *a, **kw)

    def experiment(s, name, *a, **kw):
        return Experiment(s.path / name, *a, **kw)
