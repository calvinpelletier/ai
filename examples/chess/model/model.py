import ai.model as m

from .encode import (
    BoardEncoder,
    MetaEncoder,
    BoardMetaEncoder,
    HistoryEncoder,
)
from .decode import (
    ValueDecoder,
    PolicyDecoder,
    ValuePolicyDecoder,
    BoardDecoder,
    ValuePost,
    PolicyPost,
    LegalPost,
    MetaPost,
)
from .process import Processor


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOARD
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Board2Value(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardEncoder(cfg),
            ValuePost(cfg),
        ))

class Board2Policy(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardEncoder(cfg),
            PolicyPost(cfg),
        ))

class Board2Legal(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardEncoder(cfg),
            LegalPost(cfg),
        ))

class Board2ValuePolicy(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardEncoder(cfg, cfg.encode),
            ValuePolicyDecoder(cfg, cfg.decode),
        ))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# META
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Meta2Value(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            MetaEncoder(cfg, cfg.encode),
            ValuePost(cfg, cfg.decode),
        ))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HISTORY
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class History2Meta(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            HistoryEncoder(cfg),
            MetaPost(cfg),
        ))

class History2Board(m.Model):
    def __init__(s, cfg):
        super().__init__()
        s._encoder = HistoryEncoder(cfg, cfg.encode)
        s._decoder = BoardDecoder(cfg, cfg.decode)

    def forward(s, history, history_len):
        x = s._encoder(history, history_len)
        return s._decoder(x)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BOARD/META
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BoardMeta2Policy(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardMetaEncoder(cfg, cfg.encode),
            PolicyDecoder(cfg, cfg.decode),
        ))

class BoardMeta2ValuePolicy(m.Model):
    def __init__(s, cfg):
        super().__init__(m.seq(
            BoardMetaEncoder(cfg, cfg.encode),
            Processor(cfg, cfg.main),
            ValuePolicyDecoder(cfg, cfg.decode),
        ))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


MAP = {
    'b2l': Board2Legal,
    'h2b': History2Board,
    'b2a': Board2Policy,
}
def build_model(cfg):
    return MAP[cfg.task](cfg.model)
