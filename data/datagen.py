import torch


class RLDataGenerator(torch.utils.data.IterableDataset):
    def __init__(s, agent=None):
        super().__init__()
        s._agent = agent

    @property
    def agent(s):
        return s._agent

    @agent.setter
    def agent(s, agent):
        s._agent = agent

    def __iter__(s):
        assert s._agent is not None
        return s

    def __next__(s):
        raise NotImplementedError()
