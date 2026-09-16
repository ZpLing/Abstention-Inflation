"""Write the clean-prompt S3 rerun into Table 1's FLD_MCQ / FOLIO_MCQ columns.

Table 1 lays each model out as three rows -- Acc (S1), Acc (S2), Abs Rate (S2)
-- over the columns

    FLD | FLD_MCQ | FOLIO | FOLIO_MCQ | ARC | MedQA | MMLU | LogiQA

The two *_MCQ columns are the S3 condition, so only those are touched. The
arrow subscript follows the convention already used in the table: the signed
gap between the Acc (S2) cell and the Acc (S1) cell of the SAME column,
rendered \\dn{} when accuracy fell and \\up{} when it rose. It is recomputed
here rather than carried over, so the arrow can never drift from the number
above it.

Only cells that clear the reliability gate in report_S3_clean.py are written;
anything else is left untouched and named in the run log.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/acl_latex.tex"

_spec = importlib.util.spec_from_file_location(
    "_rep", Path(__file__).with_name("report_S3_clean.py"))
_rep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rep)

# Column index after splitting a table row on "&": 0 is the multirow gutter,
# 1 is the metric label, then the eight benchmark columns.
COL = {"FLD": 3, "FOLIO": 5}          # the _MCQ column of each TFQ benchmark
ROW_LABELS = {"acc_s1": "Acc (S1)", "acc_s2": "Acc (S2)", "abs_s2": "Abs Rate (S2)"}

MODEL_ANCHOR = {
    "DeepSeek-V4-Flash": r"\multirow{3}{*}{DeepSeek-V4-Flash}",
    "GPT-5.4-nano": r"\multirow{3}{*}{GPT-5.4-nano}",
    "Gemini-3.1-Flash-Lite": r"\multirow{3}{*}{\makecell[l]{Gemini-3.1-\\Flash-Lite}}",
}


def find_block(lines, anchor):
    """Return the indices of the Acc (S1) / Acc (S2) / Abs Rate (S2) rows."""
    start = next(i for i, l in enumerate(lines) if l.strip() == anchor.strip())
    idx = {}
    for i in range(start, min(start + 8, len(lines))):
        for key, label in ROW_LABELS.items():
            if label in lines[i] and "&" in lines[i]:
                idx.setdefault(key, i)
    missing = set(ROW_LABELS) - set(idx)
    if missing:
        raise RuntimeError(f"{anchor}: could not locate rows {sorted(missing)}")
    return idx


def get_cell(line, col):
    return line.split("&")[col].strip()


def set_cell(line, col, value):
    parts = line.split("&")
    old = parts[col]
    lead = re.match(r"^\s*", old).group(0) or " "
    trail = "" if col == len(parts) - 1 else " "
    parts[col] = f"{lead}{value}{trail}"
    return "&".join(parts)


def arrow(new_acc, baseline):
    d = new_acc - baseline
    return (f"\\dn{{{abs(d):.1f}}}" if d < 0 else f"\\up{{{d:.1f}}}"), d


def main():
    text = PAPER.read_text(encoding="utf-8")
    lines = text.split("\n")
    written, skipped = [], []

    for disp, mid in _rep.MODELS:
        idx = find_block(lines, MODEL_ANCHOR[disp])
        for ds in _rep.DATASETS:
            p = _rep.CLEAN / f"summary_s3_{ds}_{mid}.json"
            if not p.exists():
                skipped.append(f"{disp} x {ds}: no result file yet")
                continue
            import json
            r = _rep.score(json.loads(p.read_text()), ds)
            if not r["ok"]:
                skipped.append(f"{disp} x {ds}: {'; '.join(r['reasons'])}")
                continue

            col = COL[ds]
            base = float(get_cell(lines[idx["acc_s1"]], col))
            mark, d = arrow(r["acc"], base)

            before_acc = get_cell(lines[idx["acc_s2"]], col)
            before_abs = get_cell(lines[idx["abs_s2"]], col)
            lines[idx["acc_s2"]] = set_cell(lines[idx["acc_s2"]], col,
                                            f"{r['acc']:.1f}{mark}")
            lines[idx["abs_s2"]] = set_cell(lines[idx["abs_s2"]], col,
                                            f"{r['absr']:.1f}")
            written.append(
                f"{disp:24} {ds+'_MCQ':11} "
                f"Acc  {before_acc:>16} -> {r['acc']:.1f}{mark}"
                f"   (S1 {base}, delta {d:+.1f})\n"
                f"{'':24} {'':11} Abs  {before_abs:>16} -> {r['absr']:.1f}"
                f"   (n={r['n']}"
                + (f", {r['trunc']} dropped for hitting the output cap)" if r["trunc"] else ")")
            )

    PAPER.write_text("\n".join(lines), encoding="utf-8")
    print("WRITTEN")
    for w in written:
        print("  " + w)
    print("\nSKIPPED")
    for s in skipped or ["  (none)"]:
        print("  " + s)


if __name__ == "__main__":
    main()
