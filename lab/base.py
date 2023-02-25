from shutil import rmtree

from ai.path import lab_path


class LabEntity:
    def __init__(s, path):
        s.path = lab_path(path)
        s.path.mkdir(parents=True, exist_ok=True)

    def clean(s):
        rmtree(s.path)
        s.path.mkdir()
        return s
