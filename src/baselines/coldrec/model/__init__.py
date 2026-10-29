from .AGNN import AGNN
from .FSGNN import FSGNN
from .MF import MF

AVAILABLE_MODELS = {
    'AGNN': AGNN,
    'FSGNN': FSGNN,
    'MF': MF,
}

__all__ = list(AVAILABLE_MODELS.keys()) + ['AVAILABLE_MODELS']
