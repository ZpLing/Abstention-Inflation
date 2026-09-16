"""Bring App. A's S3 sentences in line with the clean-prompt rerun.

The published paragraph was built around a DeepSeek-V4-Flash x FLD "outlier" whose
Table 1 cell (Acc 20.0 / Abs Rate 25.0) matches no result file and implies a
below-chance non-abstention accuracy. The rerun puts that cell at 45.2 / 39.8,
in line with every other cell, so the outlier -- and the prose defending it --
goes away.

Reads Table 1 straight out of the LaTeX source so the per-cell deltas cannot drift
from the table, then rewrites the numbers in the App. A paragraph and drops the
multiple-comparisons carve-out that only existed for that cell.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper/acl_latex.tex"

MODELS = ["DeepSeek-V4-Flash", "GPT-5.4-nano", "Gemini-3.1-Flash-Lite"]
# column index after splitting a Table 1 row on "&"
COLS = {"FLD": 2, "FLD_MCQ": 3, "FOLIO": 4, "FOLIO_MCQ": 5}


def read_table(lines):
    """Return {model: {row: {col: value}}} for the three metric rows."""
    out, cur, model = {}, None, None
    for l in lines:
        m = re.search(r"\\multirow\{3\}\{\*\}\{(?:\\makecell\[l\]\{)?([^}\\]+)", l)
        if m:
            model = m.group(1).strip()
            model = next((x for x in MODELS if x.startswith(model.rstrip("-"))), model)
            cur = out.setdefault(model, {})
        if cur is None or "&" in l is False:
            continue
        for key, label in (("s1", "Acc (S1)"), ("s2", "Acc (S2)"),
                           ("abs", "Abs Rate (S2)")):
            if label in l and "&" in l:
                parts = l.split("&")
                cur[key] = {c: parts[i].strip().rstrip("\\").strip()
                            for c, i in COLS.items()}
    return out


def num(cell):
    return float(re.match(r"([\d.]+)", cell).group(1))


def main():
    text = PAPER.read_text(encoding="utf-8")
    lines = text.split("\n")
    tbl = read_table(lines)

    deltas, worst = [], (None, 0.0)
    print(f"{'cell':34} {'S2 Abs':>7} {'S3 Abs':>7} {'dAbs':>7} "
          f"{'S2 Acc':>7} {'S3 Acc':>7} {'dAcc':>7}")
    for model in MODELS:
        for ds in ("FLD", "FOLIO"):
            r = tbl.get(model)
            if not r:
                continue
            s2a, s3a = num(r["abs"][ds]), num(r["abs"][ds + "_MCQ"])
            s2c, s3c = num(r["s2"][ds]), num(r["s2"][ds + "_MCQ"])
            deltas.append(abs(s3a - s2a))
            if abs(s3a - s2a) > worst[1]:
                worst = (f"{model} x {ds}", abs(s3a - s2a))
            print(f"{model+' x '+ds:34} {s2a:7.1f} {s3a:7.1f} {s3a-s2a:+7.1f} "
                  f"{s2c:7.1f} {s3c:7.1f} {s3c-s2c:+7.1f}")
    pooled = sum(deltas) / len(deltas)
    cap = max(deltas)
    print(f"\npooled mean |dAbs| = {pooled:.1f} over {len(deltas)} cells; "
          f"largest = {cap:.1f} ({worst[0]})")

    # Idempotent: each pattern matches the published wording OR the wording a
    # previous run of this script left behind, so the numbers can be refreshed
    # whenever a cell is re-measured.
    reps = [
        (r"S3 yields a small or null format effect (?:on 5 of 6 cells|in all 6 cells) "
         r"\(each with \$\|\\Delta\\text\{Abs Rate\}\|\\leq [\d.]+\$ points, none "
         r"significant at \$p\{=\}0\.05\$\)(?:; the DeepSeek-V4-Flash \$\\times\$ FLD cell is an "
         r"outlier with [^.]*\(McNemar \$p<0\.01\$\))?\. The pooled mean "
         r"\$\|\\Delta\\text\{Abs Rate\}\|\$ across (?:all 6 cells|the 6 cells) is [\d.]+ "
         r"points, (?:far smaller|an order of magnitude smaller) than the 31\.6\\% "
         r"S1\$\\to\$S2 Abs Rate jump, so format alone cannot account for "
         r"\\emph\{Abstention Inflation\}(?:; we read the DeepSeek--FLD anomaly as a "
         r"model-specific MCQ-format sensitivity rather than a counterexample)?\.",
         (f"S3 yields a small or null format effect in all 6 cells (each with "
          f"$|\\Delta\\text{{Abs Rate}}|\\leq {cap:.1f}$ points, none significant at "
          f"$p{{=}}0.05$). The pooled mean $|\\Delta\\text{{Abs Rate}}|$ across the 6 "
          f"cells is {pooled:.1f} points, an order of magnitude smaller than the 31.6\\% "
          f"S1$\\to$S2 Abs Rate jump, so format alone cannot account for "
          f"\\emph{{Abstention Inflation}}.")),
        (r"every test would need \$p<0\.05/36 \\approx 0\.0014\$ to reject; (?:apart from the "
         r"DeepSeek-V4-Flash \$\\times\$ FLD format cell noted above, )?the observed minimum \$p\$ "
         r"is 0\.21, so the structural null survives even under strict correction\.",
         ("every test would need $p<0.05/36 \\approx 0.0014$ to reject; the observed "
          "minimum $p$ is 0.21, so the structural null survives even under strict "
          "correction.")),
    ]
    for pat, new in reps:
        text, n = re.subn(pat, lambda _m, v=new: v, text, count=1)
        print(("  patched: " if n else "  NOT FOUND: ") + new[:64].replace("\n", " "))

    PAPER.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
