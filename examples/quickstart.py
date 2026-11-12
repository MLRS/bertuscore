"""Smallest possible BERTUScore example.  Run:  python examples/quickstart.py"""

import sys

from bertuscore import BERTUScore

# Maltese letters (ħ, ġ, ż, ċ, għ) need a UTF-8 console. Older Windows consoles
# default to cp1252 and would crash on print(); this makes the example portable.
sys.stdout.reconfigure(encoding="utf-8")

scorer = BERTUScore()  # the BERTu model downloads on first use (~500 MB)

candidates = ["Il-qattus qagħad fuq it-tapit.", "Illum it-temp huwa sabiħ ħafna."]
references = ["Il-qattus kien qed joqgħod fuq it-tapit.", "Illum it-temp huwa sabiħ ħafna."]

precision, recall, f1 = scorer.score(candidates, references)

for cand, f in zip(candidates, f1.tolist()):
    print(f"{f:.4f}  {cand}")
print(f"\nCorpus-level F1: {f1.mean().item():.4f}")
