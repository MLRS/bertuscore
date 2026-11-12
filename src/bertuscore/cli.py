"""Command-line interface: ``bertuscore -c "candidate" -r "reference"``."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .scorer import DEFAULT_MODEL, DEFAULT_NUM_LAYERS, BERTUScore


def read_lines(path: str) -> List[str]:
    """Read one sentence per line (UTF-8), ignoring a trailing blank line."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bertuscore",
        description="Compute BERTUScore (BERTScore with the Maltese BERTu model).",
        epilog=(
            "examples:\n"
            '  bertuscore -c "Il-qattus qagħad fuq it-tapit." -r "Qattus kien qed joqgħod fuq it-tapit."\n'
            "  bertuscore --file -c hyp.txt -r ref.txt --per-sentence"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-c", "--candidate", required=True,
                   help="Candidate sentence (or a file path with --file).")
    p.add_argument("-r", "--reference", required=True,
                   help="Reference sentence (or a file path with --file).")
    p.add_argument("--file", action="store_true",
                   help="Treat -c and -r as UTF-8 text files with one sentence per line.")
    p.add_argument("--per-sentence", action="store_true",
                   help="Also print P/R/F1 for every sentence pair.")
    p.add_argument("--json", action="store_true",
                   help="Print machine-readable JSON instead of text.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Hugging Face model id (default: {DEFAULT_MODEL}).")
    p.add_argument("--layer", type=int, default=DEFAULT_NUM_LAYERS,
                   help=f"Transformer layer to use (default: {DEFAULT_NUM_LAYERS}).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Sentences per batch (default: 64). Lower it if memory is tight.")
    p.add_argument("--device", default=None,
                   help='"cpu" or "cuda". Default: GPU if available, else CPU.')
    p.add_argument("--verbose", action="store_true",
                   help="Show model progress bars and timing.")
    p.add_argument("--version", action="version", version=f"bertuscore {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Make non-ASCII Maltese characters (ċ ġ ħ ż għ) print correctly on Windows.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)

    try:
        if args.file:
            cands, refs = read_lines(args.candidate), read_lines(args.reference)
        else:
            cands, refs = [args.candidate], [args.reference]

        scorer = BERTUScore(
            model_type=args.model,
            num_layers=args.layer,
            device=args.device,
            batch_size=args.batch_size,
        )
        P, R, F1 = scorer.score(cands, refs, verbose=args.verbose)
    except (OSError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    summary = {
        "model": args.model,
        "layer": args.layer,
        "num_pairs": len(cands),
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
    }

    if args.json:
        if args.per_sentence:
            summary["sentences"] = [
                {"precision": p, "recall": r, "f1": f}
                for p, r, f in zip(P.tolist(), R.tolist(), F1.tolist())
            ]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    print(f"BERTUScore ({args.model}, layer {args.layer}) over {len(cands)} sentence pair(s)")
    print(f"  Precision: {summary['precision']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f}")
    print(f"  F1:        {summary['f1']:.4f}")
    if args.per_sentence:
        print("\nline\tP\tR\tF1")
        for i, (p, r, f) in enumerate(zip(P.tolist(), R.tolist(), F1.tolist()), start=1):
            print(f"{i}\t{p:.4f}\t{r:.4f}\t{f:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
