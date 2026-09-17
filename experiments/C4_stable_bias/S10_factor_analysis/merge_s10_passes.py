"""Merge the two S10(c) passes into one n=500 summary per (dataset, model).

The sweep ran in two passes because the original gemma configs only ever
covered 200 of the 500 items the paper reports (`sample_limits: {true: 100,
false: 100}`). Pass 1 is those 200; pass 2 is the remaining 300
(`--class_offset 100 --n_per_class 150`). Re-running the first 200 would have
thrown away hours of finished GPU time, so they are merged here instead.

Metrics are recomputed over the union rather than averaged: Abs Rate and
accuracy are ratios, and averaging two ratios with different denominators
(200 and 300) would silently reweight the result.

    python merge_s10_passes.py [pass1_dir] [pass2_dir] [out_dir]
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))   # this tool is run from its own directory

#: Explicit opt-out of the generation-budget check, for summaries too old to
#: carry a run_config. Off by default: an unverifiable merge is refused.
ALLOW_UNVERIFIED = "--allow-unverified-budget" in sys.argv
_argv = [a for a in sys.argv if a != "--allow-unverified-budget"]

P1 = Path(_argv[1]) if len(_argv) > 1 else ROOT / "results" / "s10_gemma_mnt3072"
P2 = Path(_argv[2]) if len(_argv) > 2 else ROOT / "results" / "s10_gemma_p2"
OUT = Path(_argv[3]) if len(_argv) > 3 else ROOT / "results" / "s10_gemma_n500"

#: A merged cell must land on exactly this many items, or it is incomplete.
EXPECT_N = int(_argv[4]) if len(_argv) > 4 else 500

from core.result_schema import TRUSTED_TIERS as TRUSTED  # noqa: E402


class IncompatiblePasses(Exception):
    """The two passes must not be spliced together."""


def check_compatible(a: dict, b: dict, expect_n: int,
                     allow_unverified_budget: bool = False):
    """Refuse to merge passes that were not produced the same way.

    Two failure modes this guards against, both silent otherwise:

    * **Incomplete.** A pass-2 job killed part-way leaves a short summary; a
      merge would publish, say, n=350 as a finished cell and every downstream
      table and figure would treat it as complete.

    * **Stale / mixed.** The 1024-token and 3072-token sweeps write *identical
      file names* into different directories. Pointing the merger at the wrong
      one splices 200 items generated under one budget onto 300 under another
      -- and the budget moves Abs Rate by tens of points, so the result looks
      plausible and is wrong.
    """
    for key in ("model", "dataset"):
        if a.get(key) != b.get(key):
            raise IncompatiblePasses(
                f"{key} differs: {a.get(key)!r} vs {b.get(key)!r}")

    ca, cb = a.get("run_config") or {}, b.get("run_config") or {}
    if not (ca and cb) and not allow_unverified_budget:
        # Fail closed. Warning and proceeding would wave through precisely the
        # case this check exists for: the 1024-token sweep predates the
        # run_config field, so a merge pointed at it has nothing to compare and
        # would silently splice two generation budgets together.
        missing = " and ".join(n for n, c in (("pass 1", ca), ("pass 2", cb)) if not c)
        raise IncompatiblePasses(
            f"no run_config in {missing}, so the generation budget cannot be "
            f"verified. Backfill it from the job log, or pass "
            f"--allow-unverified-budget if you have checked by hand that both "
            f"passes used the same settings")
    for key in ("max_new_tokens", "use_chat_template", "class_offset"):
        # Absent on both sides compares equal, which would wave a partial
        # run_config through; require the key to actually be there.
        if key not in ca or key not in cb:
            if allow_unverified_budget:
                continue
            side = " and ".join(n for n, c in (("pass 1", ca), ("pass 2", cb))
                                if key not in c)
            raise IncompatiblePasses(
                f"run_config has no {key!r} in {side}; the passes cannot be "
                f"verified as comparable")
        if key != "class_offset" and ca[key] != cb[key]:
            raise IncompatiblePasses(
                f"{key} differs between passes: {ca[key]!r} vs {cb[key]!r}")
    if ca.get("class_offset") == cb.get("class_offset"):
        raise IncompatiblePasses(
            f"both passes used class_offset={ca.get('class_offset')!r} — "
            "they cover the same items, not complementary ones")

    ids_a = {r["id"] for r in a.get("per_sample") or []}
    ids_b = {r["id"] for r in b.get("per_sample") or []}
    overlap = ids_a & ids_b
    if overlap:
        raise IncompatiblePasses(
            f"{len(overlap)} item(s) appear in both passes "
            f"(e.g. {sorted(overlap)[0]}) — the passes are not complementary")

    total = len(ids_a | ids_b)
    if total != expect_n:
        raise IncompatiblePasses(
            f"merged size is {total}, expected {expect_n} "
            f"(pass1={len(ids_a)}, pass2={len(ids_b)}) — a pass is incomplete")

    counts = Counter(r["answer_idx"] for r in
                     list(a.get("per_sample") or []) + list(b.get("per_sample") or []))
    if counts[0] != counts[1]:
        raise IncompatiblePasses(
            f"class balance is off: {counts[0]} true vs {counts[1]} false")


def merge_one(a: dict, b: dict) -> dict:
    """Union the per-sample rows of two passes and recompute every metric.

    Call :func:`check_compatible` first; this does no validation of its own.
    """
    rows, seen = [], set()
    for src in (a, b):
        for r in src.get("per_sample") or []:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            rows.append(r)
    n = len(rows)

    def acc(pk):
        ok = sum(1 for r in rows
                 if (r["answer_idx"] == 0 and r[pk] == "A")
                 or (r["answer_idx"] == 1 and r[pk] == "B"))
        return ok / n if n else 0.0

    n_unk = sum(1 for r in rows if r["pred_s2"] == "UNKNOWN")
    n_strict = sum(1 for r in rows
                   if r["pred_s2"] == "UNKNOWN" and r["prov_s2"] in TRUSTED)
    trusted = sum(1 for r in rows if r["prov_s2"] in TRUSTED)

    out = dict(a)
    out.update({
        "n_total": n,
        "n_abstention_inflation": n_unk,
        "merged_from": {
            "pass1_n": len(a.get("per_sample") or []),
            "pass2_n": len(b.get("per_sample") or []),
            "pass1_run_config": a.get("run_config") or {},
            "pass2_run_config": b.get("run_config") or {},
        },
        "metrics": {
            "S1": {"label_acc": acc("pred_s1")},
            "S2": {"label_acc": acc("pred_s2"),
                   "abs_rate": n_unk / n if n else 0.0,
                   "abs_rate_strict": n_strict / n if n else 0.0,
                   "trusted_share": trusted / n if n else 0.0},
        },
        "provenance": {"s1": dict(Counter(r["prov_s1"] for r in rows)),
                       "s2": dict(Counter(r["prov_s2"] for r in rows))},
        "per_sample": rows,
    })
    return out


def main():
    if not P2.exists():
        sys.exit(f"pass-2 directory not found: {P2}")
    for name, d in (("pass-1", P1), ("pass-2", P2)):
        if OUT.resolve() == d.resolve():
            sys.exit(f"output directory is the {name} directory ({d}). "
                     f"This tool deletes merged cells that stop validating, so "
                     f"it must never write into an input directory.")
    OUT.mkdir(parents=True, exist_ok=True)
    merged = skipped = refused = 0
    def _is_ours(path: Path) -> bool:
        """True only for a file this tool wrote (it stamps `merged_from`).

        Deletion is scoped to those. The output directory is a caller-supplied
        path, and removing anything else in it would mean this tool can destroy
        real result files when pointed somewhere unintended.
        """
        try:
            return "merged_from" in json.loads(path.read_text())
        except Exception:
            return False

    def drop_stale(name: str, why: str):
        """Remove a previously merged cell that no longer validates.

        Refusing to write is not enough: an earlier run may have left a valid
        n=500 file here, and every downstream table and figure would go on
        reading it as current. A cell that cannot be produced now must not
        appear to exist.
        """
        stale = OUT / name
        if not stale.exists():
            return
        if not _is_ours(stale):
            print(f"            [keep] {name} in the output dir was not written "
                  f"by this tool; not deleting it — resolve by hand")
            return
        stale.unlink()
        print(f"            removed stale merged cell ({why})")

    for f1 in sorted(P1.glob("ab_summary_*.json")):
        f2 = P2 / f1.name
        if not f2.exists():
            print(f"  [wait] {f1.name}: pass 2 not finished")
            drop_stale(f1.name, "pass 2 no longer present")
            skipped += 1
            continue
        a, b = json.loads(f1.read_text()), json.loads(f2.read_text())
        try:
            check_compatible(a, b, EXPECT_N, ALLOW_UNVERIFIED)
        except IncompatiblePasses as exc:
            print(f"  [REFUSED] {f1.name}: {exc}")
            drop_stale(f1.name, "inputs no longer validate")
            refused += 1
            continue
        m = merge_one(a, b)
        dest = OUT / f1.name
        if dest.exists() and not _is_ours(dest):
            # The guard on deletion was not enough: writing would clobber a real
            # result file just as surely, and the output path is caller-supplied.
            print(f"  [REFUSED] {f1.name}: a file of that name in the output "
                  f"directory was not written by this tool; refusing to "
                  f"overwrite it")
            refused += 1
            continue
        dest.write_text(json.dumps(m, indent=2))
        s2 = m["metrics"]["S2"]
        info = m["merged_from"]
        print(f"  {m['model']:<22}{m['dataset']:<7} "
              f"{info['pass1_n']}+{info['pass2_n']}={m['n_total']:<4} "
              f"Acc_S1={m['metrics']['S1']['label_acc']:>6.1%} "
              f"AbsR={s2['abs_rate']:>6.1%} strict={s2['abs_rate_strict']:>6.1%} "
              f"trusted={s2['trusted_share']:>6.1%}")
        merged += 1
    # Anything this tool wrote that no longer has a pass-1 input is orphaned.
    expected = {f.name for f in P1.glob("ab_summary_*.json")}
    for stale in sorted(OUT.glob("ab_summary_*.json")):
        if stale.name in expected or not _is_ours(stale):
            continue
        stale.unlink()
        print(f"  [stale] removed {stale.name}: no pass-1 input")

    tail = ""
    if skipped:
        tail += f"; {skipped} still waiting on pass 2"
    if refused:
        tail += f"; {refused} REFUSED — see above, nothing was written for those"
    print(f"\nmerged {merged} cell(s) -> {OUT}{tail}")
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    main()
