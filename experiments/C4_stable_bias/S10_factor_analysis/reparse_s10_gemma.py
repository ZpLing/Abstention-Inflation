"""Re-derive every S10(c) label from the generation text already on disk.

Why this is not a re-run
------------------------
``run_S10_gemma_cuda.py`` stores ``raw_s1``/``raw_s2`` next to each prediction,
so a parser fix can be applied to finished cells without touching a GPU. The
fix that prompted this tool is ``_runon_verb``: a detokenisation artifact in the
instruction-tuned runs welded the answer verb to trailing junk
(``Final answer: Trueout``), which defeated the scheme's ``\\bTRUE\\b`` and
scored an explicit commitment as UNPARSEABLE.

Only the derived fields are rewritten -- ``pred_*``, ``prov_*``, ``metrics``,
``provenance`` and ``n_abstention_inflation``. The generations themselves are
never modified, so this is idempotent and can be re-run after later cells land.

    python reparse_s10_gemma.py [dir ...]        # default: the three S10(c) dirs

A cell whose labels do not change is left alone rather than rewritten, so the
file mtimes still say which cells the fix actually moved.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.label_scheme import get_scheme                # noqa: E402
from core.result_schema import TRUSTED_TIERS            # noqa: E402

#: The runner is loaded by path rather than imported as a module: it lives in a
#: directory that is not a package, and `_classify` is the single definition of
#: the tier ladder. Re-implementing it here is exactly the drift this tool would
#: then be unable to detect.
_spec = importlib.util.spec_from_file_location(
    "_s10_runner", Path(__file__).with_name("run_S10_gemma_cuda.py"))
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)

DEFAULT_DIRS = [ROOT / "results" / d for d in
                ("s10_gemma_mnt3072", "s10_gemma_p2", "s10_gemma_n500")]


def reparse(summary: dict) -> tuple[dict, int]:
    """(summary, n_changed) with every derived field recomputed in place.

    ``n_changed`` counts label *and* provenance changes."""
    scheme = get_scheme(summary["dataset"])
    rows = summary.get("per_sample") or []
    changed = 0
    for r in rows:
        for raw_k, pred_k, prov_k, with_unknown in (
                ("raw_s1", "pred_s1", "prov_s1", False),
                ("raw_s2", "pred_s2", "prov_s2", True)):
            if raw_k not in r:
                continue
            pred, prov = _runner._classify(r[raw_k], scheme, with_unknown)
            # Provenance counts as a change even when the label is unchanged:
            # `trusted_share` and `abs_rate_strict` are both computed from the
            # tier, so a row that keeps its label but moves between tiers still
            # has to be written back. Counting only labels left stale tiers on
            # disk in cells where no prediction happened to flip.
            if r.get(pred_k) != pred or r.get(prov_k) != prov:
                changed += 1
            r[pred_k], r[prov_k] = pred, prov

    n = len(rows)
    if not n:
        return summary, changed

    def acc(pk: str) -> float:
        ok = sum(1 for r in rows
                 if (r["answer_idx"] == 0 and r[pk] == "A")
                 or (r["answer_idx"] == 1 and r[pk] == "B"))
        return ok / n

    n_unk = sum(1 for r in rows if r["pred_s2"] == "UNKNOWN")
    n_strict = sum(1 for r in rows
                   if r["pred_s2"] == "UNKNOWN" and r["prov_s2"] in TRUSTED_TIERS)
    trusted = sum(1 for r in rows if r["prov_s2"] in TRUSTED_TIERS)

    summary["n_abstention_inflation"] = n_unk
    summary["metrics"] = {
        "S1": {"label_acc": acc("pred_s1")},
        "S2": {"label_acc": acc("pred_s2"),
               "abs_rate": n_unk / n,
               "abs_rate_strict": n_strict / n,
               "trusted_share": trusted / n},
    }
    summary["provenance"] = {
        "s1": dict(Counter(r["prov_s1"] for r in rows if "prov_s1" in r)),
        "s2": dict(Counter(r["prov_s2"] for r in rows if "prov_s2" in r)),
    }
    return summary, changed


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS
    touched = total = 0
    for d in dirs:
        if not d.exists():
            print(f"[skip] {d} does not exist")
            continue
        print(f"\n=== {d.name} ===")
        for path in sorted(d.glob("ab_summary_*.json")):
            summary = json.loads(path.read_text())
            before = summary["metrics"]["S2"].copy()
            summary, changed = reparse(summary)
            total += 1
            if not changed:
                print(f"  {path.name:<46} unchanged")
                continue
            touched += 1
            after = summary["metrics"]["S2"]
            path.write_text(json.dumps(summary, indent=2))
            print(f"  {path.name:<46} {changed:>4} field(s) changed  "
                  f"Acc {before['label_acc']:.1%}->{after['label_acc']:.1%}  "
                  f"strict {before['abs_rate_strict']:.1%}->{after['abs_rate_strict']:.1%}  "
                  f"trusted {before['trusted_share']:.1%}->{after['trusted_share']:.1%}")
    print(f"\n{touched} of {total} cell(s) rewritten")


if __name__ == "__main__":
    main()
