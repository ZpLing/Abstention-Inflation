"""Aggregate S5 — w/o "Unknown" Option Rerun.

For every main-experiment cell under ``results/``, restrict to the Abstention
Inflation set (S2 == "Unknown") and report the accuracy the model reaches once
the "Unknown" option is removed. The paper's claim is that this sits well above
the 50% random baseline (~64% pooled), i.e. the abstention hid a recoverable
answer rather than a genuine inability.

Reads both current and pre-rename summary files through
:func:`infra.result_schema.load_summary`, so it works on every result directory
in this repo.

Usage::

    python experiments/C2_deny_yet_capable/S5_without_unknown_option_rerun/analyze_S5.py
    python .../analyze_S5.py --results_root results --datasets FLD FOLIO
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from infra.result_schema import iter_cells, load_cell  # noqa: E402

RANDOM_BASELINE_TFQ = 0.50


def _binom_p(k: int, n: int, p0: float) -> float:
    """Two-sided one-sample binomial p-value; falls back to None sans scipy."""
    try:
        from scipy.stats import binomtest
    except ImportError:
        return float("nan")
    return binomtest(k, n, p0, alternative="two-sided").pvalue


def _letter(answer_idx: int) -> str:
    return chr(ord("A") + answer_idx) if answer_idx >= 0 else "UNKNOWN"


def collect(results_root: Path, datasets: set[str] | None):
    """Return {(model, dataset): [n_correct, n_total]} over the S5 rerun rows."""
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    files = []
    for ds_, model_, slug_, tt_ in iter_cells(results_root):
        try:
            s = load_cell(ds_, model_, slug_, tt_, results_root)
        except Exception as e:  # noqa: BLE001 — a malformed cell must not stop the sweep
            print(f"  [skip] {ds_}/{model_}: {e}")
            continue
        files.append(f"{slug_}/{ds_}_{model_}")
        rows = s.get("s5_rerun") or []
        if not rows:
            continue
        ds, model = s.get("dataset", "?"), s.get("model", "?")
        if datasets and ds not in datasets:
            continue
        cell = cells[(model, ds)]
        for r in rows:
            pred = r.get("pred_s5_rerun")
            if pred in (None, "", "UNPARSEABLE"):
                continue
            cell[1] += 1
            if pred == _letter(r.get("answer_idx", -1)):
                cell[0] += 1
    return cells, files


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the S5 rerun accuracy.")
    ap.add_argument("--results_root", default="results")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Restrict to these dataset names (default: all).",
    )
    args = ap.parse_args()

    cells, files = collect(
        Path(args.results_root), set(args.datasets) if args.datasets else None
    )
    if not cells:
        print(
            f"No S5 rerun rows found under {args.results_root}/ "
            f"({len(files)} summary files scanned)."
        )
        return

    header = f"{'Model':<34} {'Dataset':<16} {'n':>6} {'Acc(S5)':>9} {'Δ vs 50%':>10} {'p':>10}"
    print(header)
    print("-" * len(header))
    pooled_correct = pooled_total = 0
    for (model, ds), (correct, total) in sorted(cells.items()):
        if total == 0:
            continue
        acc = correct / total
        p = _binom_p(correct, total, RANDOM_BASELINE_TFQ)
        print(
            f"{model[:34]:<34} {ds[:16]:<16} {total:>6} {acc:>8.1%} "
            f"{acc - RANDOM_BASELINE_TFQ:>+9.1%} {p:>10.3g}"
        )
        pooled_correct += correct
        pooled_total += total

    if pooled_total:
        acc = pooled_correct / pooled_total
        p = _binom_p(pooled_correct, pooled_total, RANDOM_BASELINE_TFQ)
        print("-" * len(header))
        print(
            f"{'POOLED':<34} {'':<16} {pooled_total:>6} {acc:>8.1%} "
            f"{acc - RANDOM_BASELINE_TFQ:>+9.1%} {p:>10.3g}"
        )


if __name__ == "__main__":
    main()
