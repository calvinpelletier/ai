from shutil import rmtree

from ai.util.path import lab_path


class LabEntity:
    def __init__(s, path, clean):
        s.path = lab_path(path)
        if clean and s.path.exists():
            rmtree(s.path)
        s.path.mkdir(parents=True, exist_ok=True)
