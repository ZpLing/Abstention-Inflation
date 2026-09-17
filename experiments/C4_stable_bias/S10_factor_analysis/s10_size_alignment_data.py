"""Derive the S10(c) size x alignment table (Figure 9) from result files.

The figure used to carry 16 hardcoded rows with no link to any run, which is
how a suspicious FLD/IT column (three different scales sharing Abs Rate 0.555
to three decimals) survived unnoticed, and how three cells could be flagged
UNRELIABLE in the plotting code while the paper text made claims across "all
four model sizes" without mentioning the exclusion.

Two things this module does that the hardcoded table could not:

* **Abs Rate as a bracket.** ``run_S10_local_sweep.py`` records which parser
  tier produced each label, so a cell reports ``abs_rate`` (every UNKNOWN --
  the old, unaudited quantity) *and* ``abs_rate_strict`` (only explicit
  commitments). Base models echo the S2 prompt template, which literally
  contains the word "Unknown", so for them the two can differ enormously.

* **Reliability is derived, not asserted.** A cell is unreliable when too
  little of its output carries an explicit commitment -- ``trusted_share``
  below ``TRUST_FLOOR`` -- rather than being listed by hand.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.result_schema import get_field, load_summary   # noqa: E402

#: Below this share of explicitly-committed answers, a cell's Abs Rate is
#: whatever the whole-text fallback happened to find, and is not reportable.
TRUST_FLOOR = 0.70

SIZES = ["E2B", "E4B", "26B-A4B", "31B"]
DATASETS = ["FLD", "FOLIO"]

#: Where to read cells from: the n=500 sweep the paper reports. Every row still
#: carries its own ``n`` so a caller can check what it is comparing.
RESULTS_DIR = ROOT / "results" / "s10_gemma"


def _tag(size: str, is_it: bool) -> str:
    return f"gemma-4-{size}-it" if is_it else f"gemma-4-{size}"


def load_cells(results_dir: Path = RESULTS_DIR):
    """[{dataset, size, is_it, acc_s1, acc_s2, abs_rate, abs_rate_strict,
        trusted_share, reliable, n}] -- one row per (dataset, size, variant).

    Missing runs are reported rather than silently filled: a caller that gets
    fewer than 16 rows is looking at an incomplete sweep.
    """
    cells, missing = [], []
    for ds in DATASETS:
        for size in SIZES:
            for is_it in (False, True):
                tag = _tag(size, is_it)
                path = results_dir / f"ab_summary_{ds}_{tag}.json"
                if not path.exists():
                    missing.append(f"{ds}/{tag}")
                    continue
                s = load_summary(path)
                m = s.get("metrics") or {}
                s2 = m.get("S2") or {}
                n = s.get("n_total") or 0
                abs_rate = s2.get("abs_rate")
                if abs_rate is None:      # pre-rerun files carry only the count
                    abs_rate = (get_field(s, "n_abstention_inflation", 0) / n) if n else None
                trusted = s2.get("trusted_share")
                cells.append({
                    "dataset": ds, "size": size, "is_it": is_it, "model": tag,
                    "n": n,
                    "acc_s1": (m.get("S1") or {}).get("label_acc"),
                    "acc_s2": s2.get("label_acc"),
                    "abs_rate": abs_rate,
                    "abs_rate_strict": s2.get("abs_rate_strict"),
                    "trusted_share": trusted,
                    "reliable": (trusted is not None and trusted >= TRUST_FLOOR),
                    "provenance": (s.get("provenance") or {}).get("s2"),
                })
    return cells, missing


def check_claims(cells, metric: str = "abs_rate_strict"):
    """Test the paper's claims about this figure, per cell.

    ``metric`` defaults to ``abs_rate_strict`` on purpose. Testing these claims
    on the plain ``abs_rate`` reproduces the very bias this re-run exists to
    remove: an IT model almost always emits a `Final answer:` line, so its
    upper bound is its true value, whereas a base model's is inflated by
    template echo and whole-text fallback. Comparing the two directly puts an
    inflated base number against a clean IT one -- which does not merely add
    noise, it *manufactures support* for "IT Abs Rate varies less than base",
    because the base range is widened by parser artifacts.

    Cells whose ``trusted_share`` is below :data:`TRUST_FLOOR` are excluded
    rather than annotated: a cell where most answers were guessed by the
    fallback has no Abs Rate worth comparing. Excluded cells are reported so a
    claim resting on a thin sample is visible.
    """
    by = {(c["dataset"], c["size"], c["is_it"]): c for c in cells}
    out = []

    # Every comparison below is restricted to a single sample size: putting a
    # 500-item estimate against a differently-sized one and calling the gap a
    # finding is exactly the error this guards against.
    sizes = Counter(c["n"] for c in cells if c["n"])
    ref_n = sizes.most_common(1)[0][0] if sizes else None
    off_size = [c for c in cells if c["n"] != ref_n]

    def val(c):
        """The claim-testing quantity, or None when the cell is not comparable."""
        if c is None or not c["reliable"] or c["n"] != ref_n:
            return None
        return c.get(metric)

    # C-a: IT Abs Rate varies less across size than base does.
    #
    # Both ranges must span the SAME sizes. Filtering the two lists
    # independently would compare, say, an IT range over four scales against a
    # base range over two -- the wider scale span alone makes the first larger,
    # and the verdict would be about which cells happen to be finished rather
    # than about instruction tuning.
    for ds in DATASETS:
        usable = [s for s in SIZES
                  if val(by.get((ds, s, True))) is not None
                  and val(by.get((ds, s, False))) is not None]
        dropped = [s for s in SIZES if s not in usable]
        if len(usable) > 1:
            it = [val(by[(ds, s, True)]) for s in usable]
            bs = [val(by[(ds, s, False)]) for s in usable]
            r_it, r_bs = max(it) - min(it), max(bs) - min(bs)
            note = f"  [over {', '.join(usable)}"
            note += f"; {', '.join(dropped)} not comparable]" if dropped else "]"
            out.append(("IT Abs Rate varies less than base",
                        "OK" if r_it < r_bs else "CONTRADICTED",
                        f"{ds}: IT range={r_it:.3f} vs base range={r_bs:.3f} "
                        f"[{metric}]{note}"))
        else:
            out.append(("IT Abs Rate varies less than base", "UNTESTABLE",
                        f"{ds}: only {len(usable)} size(s) have a base *and* an "
                        f"IT cell that are both reportable at n={ref_n}"))

    # C-b: instruction tuning raises accuracy AND Abs Rate together.
    for ds in DATASETS:
        for size in SIZES:
            b, i = by.get((ds, size, False)), by.get((ds, size, True))
            if not b or not i:
                continue
            vb, vi = val(b), val(i)
            if vb is None or vi is None or b["acc_s1"] is None or i["acc_s1"] is None:
                why = []
                for lbl, c in (("base", b), ("IT", i)):
                    if not c["reliable"]:
                        why.append(f"{lbl} below the trust floor")
                    elif c["n"] != ref_n:
                        why.append(f"{lbl} is n={c['n']}, not {ref_n}")
                out.append(("acc gain accompanied by higher Abs Rate", "UNTESTABLE",
                            f"{ds}/{size}: " + ("; ".join(why) or "metric missing")))
                continue
            d_acc, d_abs = i["acc_s1"] - b["acc_s1"], vi - vb
            verdict = "OK" if (d_acc > 0 and d_abs > 0) else "CONTRADICTED"
            # Flag when the reported upper bound would have said something else:
            # that gap is the parser artifact, not a property of the models.
            alt = ""
            if b.get("abs_rate") is not None and i.get("abs_rate") is not None:
                d_abs_up = i["abs_rate"] - b["abs_rate"]
                if (d_abs > 0) != (d_abs_up > 0):
                    alt = (f"  <-- opposite verdict on the unaudited abs_rate "
                           f"(dAbsRate={d_abs_up:+.3f})")
            out.append(("acc gain accompanied by higher Abs Rate", verdict,
                        f"{ds}/{size}: dAcc={d_acc:+.3f} dAbsRate={d_abs:+.3f} "
                        f"[{metric}]{alt}"))
    return out


if __name__ == "__main__":
    cells, missing = load_cells()
    if missing:
        print(f"MISSING {len(missing)} cell(s): {', '.join(missing)}\n")
    if not cells:
        print(f"No results under {RESULTS_DIR} -- run the sweep first.")
        sys.exit(0)

    ns = {c["n"] for c in cells}
    print(f"reading {RESULTS_DIR.relative_to(ROOT)}   [{len(cells)} cells]")
    if len(ns) > 1:
        print(f"  [warn] cells disagree on sample size: {sorted(ns)} — "
              f"they are not comparable")
    hdr = (f"{'dataset':<7} {'size':<8} {'var':<5} {'n':>4} {'Acc S1':>7} {'Acc S2':>7} "
           f"{'AbsR':>7} {'AbsR strict':>12} {'trusted':>8}  ok?")
    print(hdr); print("-" * len(hdr))
    for c in cells:
        f = lambda v, w=7, p=1: (f"{v:>{w}.{p}%}" if v is not None else " " * (w - 1) + "-")
        print(f"{c['dataset']:<7} {c['size']:<8} {'IT' if c['is_it'] else 'base':<5} "
              f"{c['n']:>4} {f(c['acc_s1'])} {f(c['acc_s2'])} {f(c['abs_rate'])} "
              f"{f(c['abs_rate_strict'],12)} {f(c['trusted_share'],8)}  "
              f"{'yes' if c['reliable'] else 'LOW-TRUST'}")

    print(f"\n=== paper claims, tested on abs_rate_strict "
          f"(trust floor {TRUST_FLOOR:.0%}) ===")
    for claim, verdict, detail in check_claims(cells):
        print(f"  [{verdict:<11}] {claim:<44} {detail}")
    print("\n=== same claims on the unaudited abs_rate, for comparison only ===")
    print("    (this is the quantity the original Figure 9 used; where the two")
    print("     tables disagree, the disagreement IS the parser artifact)")
    for claim, verdict, detail in check_claims(cells, metric="abs_rate"):
        print(f"  [{verdict:<11}] {claim:<44} {detail}")
