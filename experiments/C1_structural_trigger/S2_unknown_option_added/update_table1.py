"""Write Table 1 in `paper/acl_latex.tex` from the result files.

The table lays each model out as three rows -- Acc (S1), Acc (S2), Abs Rate
(S2) -- over eight columns:

    FLD | FLD_MCQ | FOLIO | FOLIO_MCQ | ARC | MedQA | MMLU | LogiQA

Where each column comes from:

* **FLD / FOLIO** -- the native TFQ cells. S1 is `summary_s1_*`; S2 is the
  S11 run with the abstain verb in slot C, which is `build_judge_s2_prompt`
  byte for byte, so the S2 column is the S2 prompt rather than a look-alike.
* **FLD_MCQ / FOLIO_MCQ** -- S3, the same items re-rendered as A/B/C letters.
  Its Acc (S1) cell is the native S1 above, so the delta down that column
  mixes two prompts; it is written as the S3 accuracy against the S1
  baseline and the caption should say so.
* **ARC / MedQA / MMLU / LogiQA** -- the MCQ cells, pooled by sample id.

A cell with no result file is left exactly as it is in the tex, so this can
run while collection is still in flight. Scoring matches
`build_mcq_n500_table.py`: an abstention is incorrect, a turn that carries no
answer is dropped from the pair.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.evaluator import Evaluator

PAPER = ROOT / "paper/acl_latex.tex"
TFQ_DIR = ROOT / "results/positional_bias_n500"
S3_DIR = ROOT / "results/s3_clean_n500"

#: row anchor in the tex -> model id that was queried
MODELS = {
    r"\multirow{3}{*}{DeepSeek-V4-Flash}": "deepseek-v4-flash",
    r"\multirow{3}{*}{GPT-5.4-nano}": "gpt-5.4-nano",
    r"\multirow{3}{*}{\makecell[l]{Gemini-3.1-\\Flash-Lite}}": "gemini-3.1-flash-lite",
}
COLUMNS = ["FLD", "FLD_MCQ", "FOLIO", "FOLIO_MCQ", "ARC", "MedQA", "MMLU", "LogiQA"]
MCQ_DIRS = ["results/mcq_n500"]


# ----------------------------------------------------------------- MCQ cells
def _mcq_cell(model: str, bench: str):
    ids = {i["id"] for i in json.loads((ROOT / f"dataset/{bench}.json").read_text())}
    recs = {}
    for d in MCQ_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*summary*.json"), key=lambda p: p.stat().st_mtime):
            s = json.loads(f.read_text())
            if s.get("model") != model:
                continue
            for r in s.get("per_sample", []):
                if r.get("id") in ids and "pred_s1" in r and "pred_s2" in r:
                    recs[r["id"]] = r
    scored = [r for r in recs.values()
              if not _unanswered(r, "s1") and not _unanswered(r, "s2")]
    if not scored:
        return None
    n = len(scored)
    ok = lambda p, a: len(p) == 1 and p.isalpha() and ord(p) - ord("A") == a
    return (100 * sum(ok(r["pred_s1"], r["answer_idx"]) for r in scored) / n,
            100 * sum(ok(r["pred_s2"], r["answer_idx"]) for r in scored) / n,
            100 * sum(r["pred_s2"] == "UNKNOWN" for r in scored) / n)


def _unanswered(rec, setting):
    recorded = (rec.get("unanswered") or {}).get(setting.upper())
    if recorded:
        return recorded != "no_commitment"
    if rec.get(f"pred_{setting}") != "UNPARSEABLE":
        return False
    return Evaluator.classify_unanswered(rec.get(f"raw_{setting}")) != "no_commitment"


# ----------------------------------------------------------------- TFQ cells
def _tfq_rows(path: Path):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return [r for r in d["per_sample"] if not r.get("excluded")]


def _tfq_cell(model: str, dataset: str, s2_path: Path):
    """Paired S1 vs S2 over the items valid in both."""
    s1 = _tfq_rows(TFQ_DIR / f"summary_s1_{dataset}_{model}.json")
    s2 = _tfq_rows(s2_path)
    if s1 is None or s2 is None:
        return None
    a = {r["id"]: r for r in s1}
    b = {r["id"]: r for r in s2}
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    n = len(ids)
    ok = lambda r: (r["pred"] == "A" and r["answer_idx"] == 0) or \
                   (r["pred"] == "B" and r["answer_idx"] == 1)
    return (100 * sum(ok(a[i]) for i in ids) / n,
            100 * sum(ok(b[i]) for i in ids) / n,
            100 * sum(b[i]["pred"] == "UNKNOWN" for i in ids) / n)


def cells_for(model: str) -> dict:
    out = {}
    for ds in ("FLD", "FOLIO"):
        out[ds] = _tfq_cell(model, ds,
                            TFQ_DIR / f"summary_unknown_C_{ds}_{model}.json")
        out[f"{ds}_MCQ"] = _tfq_cell(model, ds,
                                     S3_DIR / f"summary_s3_{ds}_{model}.json")
    for ds in ("ARC", "MedQA", "MMLU", "LogiQA"):
        out[ds] = _mcq_cell(model, ds)
    return out


# ----------------------------------------------------------------- tex edit
def _split(line: str):
    body, _, tail = line.rpartition(r"\\")
    return body.split("&"), tail


def _join(parts, tail):
    return "&".join(parts) + r"\\" + tail


def main():
    lines = PAPER.read_text().split("\n")
    written = skipped = 0
    for anchor, model in MODELS.items():
        try:
            start = next(i for i, l in enumerate(lines) if l.strip() == anchor.strip())
        except StopIteration:
            print(f"[warn] row not found: {anchor}")
            continue
        idx = {}
        for i in range(start, min(start + 8, len(lines))):
            for key, label in (("s1", "Acc (S1)"), ("s2", "Acc (S2)"),
                               ("abs", "Abs Rate (S2)")):
                if label in lines[i] and "&" in lines[i] and not lines[i].lstrip().startswith("%"):
                    idx.setdefault(key, i)
        data = cells_for(model)
        for col_no, col in enumerate(COLUMNS, start=2):
            cell = data.get(col)
            if cell is None:
                skipped += 1
                continue
            acc1, acc2, abs2 = cell
            d = acc2 - acc1
            arrow = (r"\dn{%.1f}" % -d) if d < 0 else (r"\up{%.1f}" % d)
            for key, value in (("s1", f"{acc1:.1f}"),
                               ("s2", f"{acc2:.1f}{arrow}"),
                               ("abs", f"{abs2:.1f}")):
                parts, tail = _split(lines[idx[key]])
                parts[col_no] = f" {value} "
                lines[idx[key]] = _join(parts, tail)
            written += 1
    PAPER.write_text("\n".join(lines))
    print(f"Table 1: wrote {written} cells, left {skipped} untouched (no results yet).")


if __name__ == "__main__":
    main()
