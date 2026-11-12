# BERTUScore

**BERTUScore scores how close a Maltese sentence is to a reference Maltese sentence.**
It is [BERTScore](https://arxiv.org/abs/1904.09675) computed with
[BERTu](https://huggingface.co/MLRS/BERTu), a BERT model trained specifically on Maltese text,
including its spelling (ċ, ġ, ħ, ż, għ), instead of on English. The main use is
**evaluating machine translation into Maltese**, but it works for any task where you compare
generated Maltese text with a reference. How closely its scores track human judgments is not
tested in this repository (see "Good to know" below).

You give it candidate sentences and reference sentences. You get back a precision, recall and
F1 score between 0 and 1 for every pair. Higher means more similar in meaning.

---

## Install

BERTUScore is not on PyPI yet, so install it straight from GitHub (needs Python 3.10+):

```bash
pip install git+https://github.com/MLRS/bertuscore.git
```

This also installs `torch`, `transformers` and `bert-score`. The first time you score
something, the BERTu model (about 500 MB) is downloaded automatically and cached, so later
runs start quickly. No account or API key is needed.

## Use it from the command line

Compare two sentences:

```bash
bertuscore -c "Il-qattus qagħad fuq it-tapit." -r "Qattus kien qed joqgħod fuq it-tapit."
```

```
BERTUScore (MLRS/BERTu, layer 12) over 1 sentence pair(s)
  Precision: 0.8549
  Recall:    0.8264
  F1:        0.8404
```

`-c` is the **c**andidate (the text being judged, e.g. a machine translation) and `-r` is the
**r**eference (the correct or human version).

Score whole files, with one sentence per line in each file and the same number of lines in
both:

```bash
bertuscore --file -c hypotheses.txt -r references.txt
```

Add `--per-sentence` to also see a score for every line, or `--json` for machine-readable
output. Try it on the bundled example files:

```bash
bertuscore --file -c examples/hypotheses.txt -r examples/references.txt --per-sentence
```

```
BERTUScore (MLRS/BERTu, layer 12) over 3 sentence pair(s)
  Precision: 0.9511
  Recall:    0.9428
  F1:        0.9464

line	P	R	F1
1	0.9585	0.8935	0.9248
2	1.0000	1.0000	1.0000
3	0.8949	0.9348	0.9144
```

## Use it from Python

```python
from bertuscore import BERTUScore

scorer = BERTUScore()   # loads the model once; reuse this object

candidates = ["Il-qattus qagħad fuq it-tapit.", "Illum it-temp huwa sabiħ ħafna."]
references = ["Il-qattus kien qed joqgħod fuq it-tapit.", "Illum it-temp huwa sabiħ ħafna."]

precision, recall, f1 = scorer.score(candidates, references)

print(f1.tolist())        # one F1 per sentence pair -> [0.9248..., 1.0000...]
print(f1.mean().item())   # a single corpus-level number -> 0.9624...
```

A runnable copy is in [`examples/quickstart.py`](examples/quickstart.py). A single pair of
strings also works (`scorer.score("candidate", "reference")`), and `scorer.f1(...)` returns
just the F1 values as plain Python floats.

## Reading the scores

- **1.0 means identical.** The same sentence compared with itself scores 1.0.
- **Precision** asks: how much of the *candidate* is supported by the reference?
  **Recall** asks: how much of the *reference* is covered by the candidate?
  **F1** combines the two and is the number most people report.
- **Scores are compressed towards the top.** Even loosely related sentences score around
  0.8, as in the first example above. BERTUScore does not apply BERTScore's optional
  "baseline rescaling", so don't read 0.84 as "84% correct". Use the scores to **compare
  systems or sentences against each other under the same settings**, not as an absolute
  grade.
- Only compare scores produced with the same model and layer.

## Options

| What | CLI flag | Python argument | Default |
|---|---|---|---|
| Model used for embeddings | `--model` | `model_type=` | `MLRS/BERTu` |
| Transformer layer | `--layer` | `num_layers=` | `12` (the last layer) |
| Sentences per batch | `--batch-size` | `batch_size=` | `64` (lower it if memory runs out) |
| `cpu` or `cuda` | `--device` | `device=` | GPU if available, else CPU |
| Show progress bars | `--verbose` | `score(..., verbose=True)` | off |

## What it does exactly

BERTUScore is a thin, documented wrapper around the
[`bert-score`](https://github.com/Tiiiger/bert_score) library with these fixed choices, which
keep results reproducible:

- model `MLRS/BERTu`, embeddings from layer 12;
- greedy token matching with cosine similarity (standard BERTScore);
- no IDF weighting and no baseline rescaling.

## Good to know

- **Maltese only.** Both candidate and reference should be Maltese; BERTu was trained on Maltese text only.
  other languages.
- **Long inputs are truncated.** BERTu reads at most 512 tokens per sentence; anything beyond
  that is cut off rather than causing an error.
- **GPU is optional.** It runs on a normal laptop CPU; a GPU just makes large files faster.
- **Windows console shows `?` or crashes on ħ/ġ/ż?** Your console is not using UTF-8. The
  `bertuscore` command handles this itself. In your own scripts, start with
  `sys.stdout.reconfigure(encoding="utf-8")`.
- **Seeing a "LOAD REPORT" about `pooler` or `cls.predictions` weights?** BERTUScore hides
  it, but if you load BERTu yourself you may see it. It is expected and harmless, because
  BERTScore does not use those parts of the model.
- **Not validated here.** This repository provides the metric implementation. It does not
  include a study of how well the scores agree with human judgments of Maltese translation
  quality.

## Development

```bash
git clone https://github.com/MLRS/bertuscore.git
cd bertuscore
pip install -e ".[dev]"
pytest                 # fast tests, no download
pytest -m slow         # also runs the real BERTu model end to end
```

## Licence and credits

The code in this repository is released under the [MIT licence](LICENSE).

The BERTu model is **not** part of this repository; it is downloaded from the Hugging Face
Hub and has its own licence, **CC BY-NC-SA 4.0 (non-commercial)**. Check that licence before
using BERTUScore in a commercial setting.

If you use this, please also credit the work it builds on:

- **BERTu**: Micallef, K., Gatt, A., Tanti, M., van der Plas, L., Borg, C. (2022).
  *Pre-training Data Quality and Quantity for a Low-Resource Language: New Corpus and BERT
  Models for Maltese.* DeepLo 2022. [ACL Anthology](https://aclanthology.org/2022.deeplo-1.10)
- **BERTScore**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., Artzi, Y. (2020).
  *BERTScore: Evaluating Text Generation with BERT.* ICLR 2020.
  [arXiv:1904.09675](https://arxiv.org/abs/1904.09675)

To cite this software, use the "Cite this repository" button on GitHub (see
[`CITATION.cff`](CITATION.cff)).
