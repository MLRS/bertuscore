"""Fast unit tests (no model download) plus one opt-in end-to-end test.

Run the fast tests:      pytest
Run the real-model test: pytest -m slow     (downloads MLRS/BERTu, ~500 MB)
"""

import json

import pytest
import torch

import bertuscore
from bertuscore import BERTUScore, cli

CAND = "Il-qattus qagħad fuq it-tapit."
REF = "Qattus kien qed joqgħod fuq it-tapit."


class FakeScorer:
    """Stands in for bert_score.BERTScorer so tests need no model."""

    def __init__(self):
        self.calls = []

    def score(self, cands, refs, verbose=False, batch_size=64):
        self.calls.append((list(cands), list(refs), batch_size))
        n = len(cands)
        return torch.full((n,), 0.8), torch.full((n,), 0.7), torch.full((n,), 0.75)


@pytest.fixture
def fake(monkeypatch):
    fake_scorer = FakeScorer()
    monkeypatch.setattr(BERTUScore, "_get_scorer", lambda self: fake_scorer)
    return fake_scorer


def test_defaults():
    s = BERTUScore()
    assert s.model_type == "MLRS/BERTu"
    assert s.num_layers == 12


def test_model_is_lazy():
    # Constructing the object must not load anything.
    assert BERTUScore()._scorer is None


def test_single_strings_are_wrapped(fake):
    P, R, F1 = BERTUScore().score(CAND, REF)
    assert fake.calls[0][:2] == ([CAND], [REF])
    assert F1.shape == (1,)


def test_lists_and_f1_helper(fake):
    out = BERTUScore().f1([CAND, CAND], [REF, REF])
    assert out == pytest.approx([0.75, 0.75])


def test_batch_size_is_forwarded(fake):
    BERTUScore(batch_size=8).score([CAND], [REF])
    assert fake.calls[0][2] == 8


def test_length_mismatch_raises(fake):
    with pytest.raises(ValueError, match="same length"):
        BERTUScore().score([CAND, CAND], [REF])


def test_empty_raises(fake):
    with pytest.raises(ValueError, match="No sentences"):
        BERTUScore().score([], [])


def test_non_string_raises(fake):
    with pytest.raises(TypeError):
        BERTUScore().score([CAND, 3], [REF, REF])


def test_cli_strings(fake, capsys):
    assert cli.main(["-c", CAND, "-r", REF]) == 0
    out = capsys.readouterr().out
    assert "F1:" in out and "0.7500" in out


def test_cli_files_per_sentence_json(fake, tmp_path, capsys):
    c, r = tmp_path / "c.txt", tmp_path / "r.txt"
    c.write_text(CAND + "\n" + CAND + "\n", encoding="utf-8")
    r.write_text(REF + "\n" + REF + "\n", encoding="utf-8")
    assert cli.main(["--file", "-c", str(c), "-r", str(r), "--json", "--per-sentence"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_pairs"] == 2
    assert len(data["sentences"]) == 2
    assert data["f1"] == pytest.approx(0.75)


def test_cli_reports_mismatch(fake, tmp_path, capsys):
    c, r = tmp_path / "c.txt", tmp_path / "r.txt"
    c.write_text("a\nb\n", encoding="utf-8")
    r.write_text("a\n", encoding="utf-8")
    assert cli.main(["--file", "-c", str(c), "-r", str(r)]) == 2
    assert "same length" in capsys.readouterr().err


def test_cli_missing_file(capsys):
    assert cli.main(["--file", "-c", "nope.txt", "-r", "nope2.txt"]) == 2
    assert "error:" in capsys.readouterr().err


def test_tokenizer_length_is_capped():
    """Regression: BERTu's tokenizer reports ~1e30 as max length, which crashes
    recent `transformers` with OverflowError. We must cap it at the model limit."""
    from types import SimpleNamespace

    from bertuscore.scorer import _cap_tokenizer_length

    tok = SimpleNamespace(model_max_length=int(1e30))
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    _cap_tokenizer_length(SimpleNamespace(_tokenizer=tok, _model=model))
    assert tok.model_max_length == 512


def test_tokenizer_length_left_alone_when_already_valid():
    from types import SimpleNamespace

    from bertuscore.scorer import _cap_tokenizer_length

    tok = SimpleNamespace(model_max_length=128)
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    _cap_tokenizer_length(SimpleNamespace(_tokenizer=tok, _model=model))
    assert tok.model_max_length == 128


def test_version_exposed():
    assert bertuscore.__version__


@pytest.mark.slow
def test_real_model_end_to_end():
    """Runs the actual BERTu model. Identical sentences must outscore different ones."""
    scorer = BERTUScore(device="cpu")
    _, _, f1 = scorer.score([REF, CAND], [REF, REF])
    assert f1[0].item() > 0.99          # a sentence compared with itself
    assert f1[0].item() > f1[1].item()  # ...beats a different sentence


@pytest.mark.slow
def test_real_model_handles_overlong_input():
    """Sentences beyond BERTu's 512-token limit are truncated, not a crash."""
    long_text = " ".join([REF] * 200)
    _, _, f1 = BERTUScore(device="cpu").score([long_text], [long_text])
    assert f1[0].item() > 0.99
