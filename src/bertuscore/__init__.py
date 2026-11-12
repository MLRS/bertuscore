"""BERTUScore: a BERTScore-style evaluation metric for Maltese."""

from .scorer import DEFAULT_MODEL, DEFAULT_NUM_LAYERS, BERTUScore, score

__version__ = "0.1.0"

__all__ = ["BERTUScore", "score", "DEFAULT_MODEL", "DEFAULT_NUM_LAYERS", "__version__"]
