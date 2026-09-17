"""Rewrite Table 1 of the paper from the result files.

One row triplet per model: S1 Acc, S2 Acc with its delta from S1, and Abs Rate
under S2. The TFQ_MCQ columns are S3 (the question-format ablation), so they
share S1's baseline and report S3's accuracy and Abs Rate.
"""
import glob
import json
import re
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.result_schema import paired_keep_ids   # noqa: E402
TEX = ROOT / "paper/acl_latex.tex"

MODELS = [("dsv4flash", "deepseek-v4-flash", "DeepSeek-V4-Flash"),
          ("nano", "gpt-5.4-nano", "GPT-5.4-nano"),
          ("gemini31", "gemini-3.1-flash-lite", "Gemini-3.1-Flash-Lite")]
COLUMNS = ["FLD", "FLD_MCQ", "FOLIO", "FOLIO_MCQ", "ARC", "MedQA", "MMLU", "LogiQA"]
ROW_LABEL = {"dsv4flash": "DeepSeek-V4-Flash", "nano": "GPT-5.4-nano",
             "gemini31": "\\makecell[l]{Gemini-3.1-\\\\Flash-Lite}"}


def _rates(summary, second):
    """(S1 Acc, `second` Acc, `second` Abs Rate) on the paired keep-set.

    The TFQ and MCQ summaries were written by different code paths and only the
    TFQ one records its own n_scored, so both are re-scored here through the
    one keep-set rule. Reading each family's stored metrics instead puts two
    denominators in one table.
    """
    keep = paired_keep_ids(summary)
    rows = [r for r in summary["per_sample"] if r["id"] in keep]
    field = {"S1": "pred_s1", "S2": "pred_s2", "S3": "pred_s3_format"}
    n = len(rows)

    def acc(name):
        f = field[name]
        return 100 * sum(r[f] == chr(ord("A") + r["answer_idx"]) for r in rows) / n

    abs_rate = 100 * sum(r[field[second]] == "UNKNOWN" for r in rows) / n
    return acc("S1"), acc(second), abs_rate


def cell(slug, model, col):
    """(S1 Acc, S2-or-S3 Acc, Abs Rate) for one column, as percentages."""
    if col.endswith("_MCQ"):
        path = ROOT / f"results/tfq_n500/{slug}/ab_summary_{col[:-4]}_{model}.json"
        return _rates(json.loads(path.read_text()), "S3")
    if col in ("FLD", "FOLIO"):
        path = ROOT / f"results/tfq_n500/{slug}/ab_summary_{col}_{model}.json"
    else:
        path = Path(glob.glob(str(ROOT / f"results/mcq_n500/*/ab_summary_{col}_{model}.json"))[0])
    return _rates(json.loads(path.read_text()), "S2")


def delta(a1, a2):
    d = a2 - a1
    return f"\\up{{{d:.1f}}}" if d >= 0 else f"\\dn{{{-d:.1f}}}"


def main():
    table, cells = [], {}
    for slug, model, label in MODELS:
        vals = [cell(slug, model, c) for c in COLUMNS]
        cells[label] = dict(zip(COLUMNS, vals))
        head = f"\\multirow{{3}}{{*}}{{{ROW_LABEL[slug]}}}"
        s1 = " & ".join(f"{v[0]:.1f}" for v in vals)
        s2 = " & ".join(f"{v[1]:.1f}{delta(v[0], v[1])}" for v in vals)
        ab = " & ".join(f"{v[2]:.1f}" for v in vals)
        table.append(f"{head}\n  & Acc (S1) & {s1} \\\\\n"
                     f"  & Acc (S2) & {s2} \\\\\n"
                     f"  & Abs Rate (S2) & {ab} \\\\")

    body = "\n\\midrule\n".join(table)
    t = TEX.read_text()
    start = t.index("\\multirow{3}{*}{DeepSeek-V4-Flash}")
    end = t.index("\\bottomrule", start)
    t = t[:start] + body + "\n" + t[end:]
    TEX.write_text(t)
    print("Table 1 rewritten from results/\n")

    tfq = [(lab, c, v) for lab, d in cells.items() for c, v in d.items() if c in ("FLD", "FOLIO")]
    s1_all = [v[0] for _l, _c, v in tfq]
    print(f"TFQ S1 Acc range      : {min(s1_all):.1f}--{max(s1_all):.1f}%")
    for ds in ("FLD", "FOLIO"):
        ab = [d[ds][2] for d in cells.values()]
        print(f"{ds:6s} Abs Rate range  : {min(ab):.1f}--{max(ab):.1f}%")
    for c in ("ARC", "MedQA", "MMLU", "LogiQA"):
        a = [d[c][0] for d in cells.values()]
        print(f"{c:6s} S1 Acc range    : {min(a):.1f}--{max(a):.1f}%")


if __name__ == "__main__":
    main()
