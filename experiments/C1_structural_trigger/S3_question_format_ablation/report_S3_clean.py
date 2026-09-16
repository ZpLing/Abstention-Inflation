"""Reliability gate + Table 1 fill-in for the clean-prompt S3 rerun.

A cell is publishable only if the responses are actual measurements. Three ways
they are not, all of which the earlier run hid:
  * the endpoint refused the call            -> excluded_rate
  * the reply was cut off before any answer  -> unparseable (tail window empty)
  * the answer had to be guessed from a
    whole-text keyword scan                  -> rejected by the runner

Prints the per-cell verdict and, for the cells that pass, the LaTeX cell text
for Table 1's FLD_MCQ / FOLIO_MCQ columns with the arrow subscript recomputed
against that column's own Acc (S1) baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLEAN = ROOT / "results/s3_clean_n500"

MODELS = [("DeepSeek-R1", "deepseek-r1-distill-llama-8b"),
          ("GPT-5.4-nano", "gpt-5.4-nano"),
          ("Gemini-3.1-Flash-Lite", "gemini-3.1-flash-lite")]
DATASETS = ["FLD", "FOLIO"]

# Table 1, Acc (S1) row, FLD_MCQ / FOLIO_MCQ columns (unchanged by this rerun).
S1_BASELINE = {
    ("DeepSeek-R1", "FLD"): 66.6,  ("DeepSeek-R1", "FOLIO"): 82.6,
    ("GPT-5.4-nano", "FLD"): 53.8, ("GPT-5.4-nano", "FOLIO"): 84.0,
    ("Gemini-3.1-Flash-Lite", "FLD"): 64.6,
    ("Gemini-3.1-Flash-Lite", "FOLIO"): 86.5,
}

MAX_REFUSED = 0.05        # endpoint refusals
MAX_TRUNCATED = 0.15      # replies that hit the output cap before answering
MAX_UNMAPPED = 0.02       # an answer was found but could not be mapped
MAX_DEPTH_SKEW = 0.75     # mean proof-depth gap allowed between kept/dropped

DATASET_CACHE = {}


def _depths(ds):
    if ds not in DATASET_CACHE:
        rows = json.loads((ROOT / f"dataset/{ds}.json").read_text())
        out = {}
        for r in rows:
            for k in ("depth", "steps", "n_steps", "proof_depth"):
                if isinstance(r.get(k), int):
                    out[r["id"]] = r[k]
                    break
        DATASET_CACHE[ds] = out
    return DATASET_CACHE[ds]


def score(d, ds):
    """Split responses into measurements vs. measurement failures.

    A reply that ran past the output cap without ever reaching an answer is not
    a model decision -- it is the same kind of non-measurement as an endpoint
    refusal, so it leaves the denominator instead of being scored. That is only
    legitimate if the dropped items are not a harder or differently-labelled
    subset, so the depth/label skew is checked and reported alongside.
    """
    rows = [r for r in d["per_sample"] if not r.get("excluded")]
    truncated = [r for r in rows if r["tier"] == "rejected_no_answer_in_tail"]
    unmapped = [r for r in rows
                if r["pred"] == "UNPARSEABLE" and r not in truncated]
    kept = [r for r in rows if r["pred"] != "UNPARSEABLE"]

    n_resp = d["n"]
    refused_rate = d["excluded_rate"]
    trunc_rate = len(truncated) / len(rows) if rows else 1.0
    unmapped_rate = len(unmapped) / len(rows) if rows else 1.0

    dep = _depths(ds)
    dk = [dep[r["id"]] for r in kept if r["id"] in dep]
    dd = [dep[r["id"]] for r in truncated if r["id"] in dep]
    skew = abs(sum(dk)/len(dk) - sum(dd)/len(dd)) if dk and dd else 0.0

    n = len(kept)
    acc = 100 * sum((r["pred"] == "A" and r["answer_idx"] == 0) or
                    (r["pred"] == "B" and r["answer_idx"] == 1)
                    for r in kept) / n if n else 0.0
    n_abs = sum(1 for r in kept if r["pred"] == "UNKNOWN")
    absr = 100 * n_abs / n if n else 0.0
    # sensitivity: if every dropped reply had (not) been an abstention
    lo = 100 * n_abs / (n + len(truncated)) if rows else 0.0
    hi = 100 * (n_abs + len(truncated)) / (n + len(truncated)) if rows else 0.0

    reasons = []
    if not d["length_ok"]:
        reasons.append("response count mismatch")
    if refused_rate > MAX_REFUSED:
        reasons.append(f"{refused_rate:.1%} refused by endpoint")
    if trunc_rate > MAX_TRUNCATED:
        reasons.append(f"{trunc_rate:.1%} hit the output cap before answering")
    if unmapped_rate > MAX_UNMAPPED:
        reasons.append(f"{unmapped_rate:.1%} answer found but unmappable")
    if skew > MAX_DEPTH_SKEW:
        reasons.append(f"dropped items skew {skew:.1f} steps in depth")
    return dict(n=n, acc=acc, absr=absr, lo=lo, hi=hi, trunc=len(truncated),
                trunc_rate=trunc_rate, unmapped=len(unmapped), skew=skew,
                ok=not reasons, reasons=reasons, n_resp=n_resp)


def main():
    print(f"{'cell':32} {'n':>4} {'Acc':>6} {'Abs':>6} {'[lo,hi]':>13} "
          f"{'trunc':>11} {'skew':>5} {'maxtok':>7}  verdict")
    print("-" * 118)
    rows = {}
    for disp, mid in MODELS:
        for ds in DATASETS:
            p = CLEAN / f"summary_s3_{ds}_{mid}.json"
            if not p.exists():
                print(f"{disp+' x '+ds:32} {'-':>4} {'-':>6} {'-':>6} "
                      f"{'-':>13} {'-':>11} {'-':>5} {'-':>7}  MISSING")
                continue
            d = json.loads(p.read_text())
            r = score(d, ds)
            rows[(disp, ds)] = r
            print(f"{disp+' x '+ds:32} {r['n']:4} {r['acc']:6.1f} {r['absr']:6.1f} "
                  f"[{r['lo']:5.1f},{r['hi']:5.1f}] "
                  f"{r['trunc']:4}({r['trunc_rate']*100:4.1f}%) {r['skew']:5.2f} "
                  f"{d.get('max_tokens','?'):>7}  "
                  f"{'PASS' if r['ok'] else 'FAIL: ' + '; '.join(r['reasons'])}")

    print("\n\n=== Table 1 cell text (FLD_MCQ / FOLIO_MCQ columns) ===")
    for disp, _ in MODELS:
        for ds in DATASETS:
            if (disp, ds) not in rows:
                continue
            r = rows[(disp, ds)]
            acc, absr, ok = r["acc"], r["absr"], r["ok"]
            base = S1_BASELINE[(disp, ds)]
            delta = acc - base
            arrow = f"\\dn{{{abs(delta):.1f}}}" if delta < 0 else f"\\up{{{delta:.1f}}}"
            tag = "" if ok else "   <-- DOES NOT PASS, do not publish"
            print(f"  {disp:24} {ds+'_MCQ':12} "
                  f"Acc (S2) = {acc:.1f}{arrow}   Abs Rate (S2) = {absr:.1f}"
                  f"   (n={r['n']}){tag}")
            if r["trunc"]:
                print(f"  {'':24} {'':12} {r['trunc']} replies hit the output cap "
                      f"before answering and left the denominator; Abs Rate is "
                      f"{r['lo']:.1f}-{r['hi']:.1f} across the two extreme "
                      f"assumptions about them")
            print(f"  {'':24} {'':12} (S1 baseline {base} is from the earlier "
                  f"run, so this delta is not a paired contrast)")


if __name__ == "__main__":
    main()
