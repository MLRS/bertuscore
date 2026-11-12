"""Core BERTUScore implementation.

BERTUScore is BERTScore (Zhang et al., 2020) computed with the Maltese BERTu
encoder (Micallef et al., 2022) instead of an English or multilingual model.
This module is a thin, documented wrapper around the ``bert-score`` library.
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence, Tuple, Union

DEFAULT_MODEL = "MLRS/BERTu"
# BERTu is a BERT-base model (12 transformer layers). Layer 12 is the last one.
DEFAULT_NUM_LAYERS = 12

Texts = Union[str, Sequence[str]]


def _as_list(texts: Texts, name: str) -> List[str]:
    """Accept a single string or a sequence of strings and return a list."""
    if isinstance(texts, str):
        return [texts]
    texts = list(texts)
    if not all(isinstance(t, str) for t in texts):
        raise TypeError(f"`{name}` must be a string or a sequence of strings.")
    return texts


@contextlib.contextmanager
def _quiet_model_loading():
    """Hide the weight-loading progress bars and the "LOAD REPORT" table.

    ``transformers`` prints a report listing ``pooler.*`` as MISSING and
    ``cls.predictions.*`` as UNEXPECTED when loading BERTu. Both are expected
    and harmless: BERTScore only uses the encoder's hidden states, not the
    pooler or the masked-language-model head. Real errors still surface.
    """
    from transformers.utils import logging as hf_logging

    previous = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.set_verbosity(previous)
        hf_logging.enable_progress_bar()


def _cap_tokenizer_length(scorer) -> None:
    """Give the tokenizer a real maximum length.

    The BERTu tokenizer does not declare ``model_max_length``, so
    ``transformers`` reports a huge placeholder (~1e30). Recent ``transformers``
    versions cannot convert that to an integer when ``bert_score`` asks for
    truncation, which crashes with ``OverflowError: int too big to convert``.
    Capping it at the model's real limit (512 positions) fixes the crash and
    also means over-long sentences are truncated instead of breaking the model.
    """
    tokenizer = getattr(scorer, "_tokenizer", None)
    model = getattr(scorer, "_model", None)
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if tokenizer is not None and limit and tokenizer.model_max_length > limit:
        tokenizer.model_max_length = limit


class BERTUScore:
    """Compute BERTUScore precision, recall and F1 for Maltese text.

    The BERTu model is downloaded from the Hugging Face Hub the first time it
    is needed (about 500 MB) and then cached locally. It is loaded lazily, so
    creating a ``BERTUScore`` object is instant, and loaded only once, so
    scoring several batches with the same object is fast.

    Args:
        model_type: Hugging Face model id used for the embeddings.
            Defaults to ``"MLRS/BERTu"``.
        num_layers: Which transformer layer's embeddings to use. Defaults to
            12, the final layer of BERTu.
        device: ``"cpu"``, ``"cuda"``, ... If ``None``, a GPU is used when one
            is available, otherwise the CPU.
        batch_size: Number of sentences embedded at once. Lower it if you run
            out of memory.

    Example:
        >>> scorer = BERTUScore()
        >>> P, R, F1 = scorer.score(
        ...     ["Il-qattus qagħad fuq it-tapit."],
        ...     ["Qattus kien qed joqgħod fuq it-tapit."],
        ... )
        >>> float(F1[0])  # doctest: +SKIP
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL,
        num_layers: int = DEFAULT_NUM_LAYERS,
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        self.model_type = model_type
        self.num_layers = num_layers
        self.device = device
        self.batch_size = batch_size
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            from bert_score import BERTScorer  # imported lazily: it pulls in torch

            with _quiet_model_loading():
                scorer = BERTScorer(
                    model_type=self.model_type,
                    num_layers=self.num_layers,
                    device=self.device,
                )
            _cap_tokenizer_length(scorer)
            self._scorer = scorer
        return self._scorer

    def score(
        self,
        candidates: Texts,
        references: Texts,
        verbose: bool = False,
    ):
        """Score candidate sentences against reference sentences.

        Args:
            candidates: The sentences to evaluate (e.g. machine translations),
                as one string or a list of strings.
            references: The gold sentences, in the same order and the same
                number as ``candidates``.
            verbose: Show a progress bar.

        Returns:
            A ``(precision, recall, f1)`` tuple of 1-D ``torch.Tensor``s with
            one value per sentence pair. Use ``.mean().item()`` for a corpus
            average or ``.tolist()`` for plain Python numbers.

        Raises:
            ValueError: If the inputs are empty or have different lengths.
        """
        cands = _as_list(candidates, "candidates")
        refs = _as_list(references, "references")
        if len(cands) != len(refs):
            raise ValueError(
                f"Got {len(cands)} candidates but {len(refs)} references; "
                "they must be the same length (one reference per candidate)."
            )
        if not cands:
            raise ValueError("No sentences to score.")

        return self._get_scorer().score(
            cands, refs, verbose=verbose, batch_size=self.batch_size
        )

    def f1(self, candidates: Texts, references: Texts) -> List[float]:
        """Convenience wrapper returning only the per-sentence F1 as floats."""
        return self.score(candidates, references)[2].tolist()


def score(
    candidates: Texts,
    references: Texts,
    model_type: str = DEFAULT_MODEL,
    num_layers: int = DEFAULT_NUM_LAYERS,
    device: Optional[str] = None,
    batch_size: int = 64,
    verbose: bool = False,
) -> Tuple["object", "object", "object"]:
    """One-shot functional interface, mirroring ``bert_score.score``.

    This reloads the model on every call. If you score more than once, create a
    :class:`BERTUScore` object and reuse it instead.
    """
    return BERTUScore(
        model_type=model_type,
        num_layers=num_layers,
        device=device,
        batch_size=batch_size,
    ).score(candidates, references, verbose=verbose)
