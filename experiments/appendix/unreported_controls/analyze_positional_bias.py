"""Aggregate results from scripts/run_positional_bias.py."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scipy.stats import binomtest
from core.metrics import label_acc, label_macro_f1, judge_classes

RESULT_DIR = ROOT / "results/positional_bias"

MODELS = [
    ("nano", "gpt-5.4-nano"),
    ("gemini", "gemini-2.5-flash-lite"),
    ("deepseek", "deepseek-r1-distill-llama-8b"),
]
DATASETS = ["FLD", "FOLIO"]
POSITIONS = ["A", "B", "C"]

C_FALLBACK_SOURCES = {
    ("nano", "FLD"): [
        "results/ab_gpt5_nano/ab_summary_FLD_gpt-5.4-nano.json",
        "results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json",
    ],
    ("nano", "FOLIO"): [
        "results/ab_gpt5_nano/ab_summary_FOLIO_gpt-5.4-nano.json",
        "results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json",
    ],
    ("gemini", "FLD"): [
        "results/ab_gemini_flash_lite/ab_summary_FLD_gemini-2.5-flash-lite.json",
        "results/ab_gemini_batch2/ab_summary_FLD_gemini-2.5-flash-lite.json",
    ],
    ("gemini", "FOLIO"): [
        "results/ab_gemini_flash_lite/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
        "results/ab_gemini_batch2/ab_summary_FOLIO_gemini-2.5-flash-lite.json",
    ],
    # DeepSeek's original FLD batch1 did not include pred_s5; ab_s5 fills that
    # exact first-batch sample set, and ab_deepseek_batch2 fills the second.
    ("deepseek", "FLD"): [
        "results/ab_s5/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
        "results/ab_deepseek_batch2/ab_summary_FLD_deepseek-r1-distill-llama-8b.json",
    ],
    ("deepseek", "FOLIO"): [
        "results/ab_followup/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
        "results/ab_deepseek_batch2/ab_summary_FOLIO_deepseek-r1-distill-llama-8b.json",
    ],
}


def load_summary(model_name: str, dataset: str, position: str, result_dir: Path = None):
    result_dir = result_dir or RESULT_DIR
    path = result_dir / f"summary_unknown_{position}_{dataset}_{model_name}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def load_c_fallback(model_key: str, model_name: str, dataset: str):
    paths = C_FALLBACK_SOURCES.get((model_key, dataset), [])
    per_sample = []
    seen = set()
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for row in data.get("per_sample", []):
            if "pred_s5" not in row:
                continue
            sid = row["id"]
            if sid in seen:
                continue
            seen.add(sid)
            pred = row["pred_s5"]
            raw_letter = "C" if pred == "UNKNOWN" else pred if pred in ("A", "B") else None
            per_sample.append({
                "id": sid,
                "source": row.get("source", dataset),
                "answer_idx": row["answer_idx"],
                "pred": pred,
                "raw_letter": raw_letter,
                "tier": "fallback_s5",
                "raw": row.get("raw_s5", ""),
            })
    if not per_sample:
        return None
    preds = [r["pred"] for r in per_sample]
    answer_idxs = [r["answer_idx"] for r in per_sample]
    raw_letters = [r["raw_letter"] for r in per_sample]
    metrics = {
        "n": len(per_sample),
        "label_acc": label_acc(preds, answer_idxs),
        "label_f1": label_macro_f1(preds, answer_idxs, judge_classes(with_unknown=True)),
        "abstain_rate": sum(p == "UNKNOWN" for p in preds) / len(preds),
        "counts": {
            "A": preds.count("A"),
            "B": preds.count("B"),
            "UNKNOWN": preds.count("UNKNOWN"),
            "UNPARSEABLE": preds.count("UNPARSEABLE"),
        },
    }
    return {
        "experiment": "positional_bias",
        "model_key": model_key,
        "model": model_name,
        "dataset": dataset,
        "unknown_position": "C",
        "n": len(per_sample),
        "metrics": metrics,
        "raw_letter_counts": {
            "A": raw_letters.count("A"),
            "B": raw_letters.count("B"),
            "C": raw_letters.count("C"),
            "null": raw_letters.count(None),
        },
        "source_summaries": paths,
        "per_sample": per_sample,
        "source_note": "C loaded from original AB/S5 pred_s5 summaries",
    }


# Row order by model_key. Row LABELS are taken from the actual model id in the
# data (never a hardcoded alias) so the report cannot mislabel what was run.
DISPLAY_ORDER = ["deepseek", "nano", "gemini"]

# Expected full coverage = every model_key × dataset. Used to mark a report as
# partial when some cells did not complete.
EXPECTED_CELLS = len(DISPLAY_ORDER) * len(DATASETS)


def write_markdown_report(table_rows, incomplete, out_path: Path, incompatible=None):
    """Emit the markdown table from complete rows only, labeled by real model id.

    table_rows carries model_key/model(id)/dataset/A/B/C/Max-Min for cells that
    finished cleanly AND passed the cross-position compatibility gate; incomplete
    and incompatible cells are listed as caveats rather than tabulated, so a
    partial, stale, or mislabeled number never lands in the paper table.
    """
    incompatible = incompatible or []
    by = {(r["model_key"], r["dataset"]): r for r in table_rows}
    partial = len(table_rows) < EXPECTED_CELLS
    # Title's label claim is data-driven: only assert "unified" if every
    # tabulated row actually ran with unified labels.
    all_unified = bool(table_rows) and all(r.get("unified_labels") is True for r in table_rows)
    label_note = "unified True/False/Unknown" if all_unified else "mixed/native labels — verify"
    lines = [
        f"# Positional Bias Control: Unknown Option Position (n=500, {label_note})",
        "",
    ]
    if partial:
        lines += [
            f"> ⚠ PARTIAL: {len(table_rows)}/{EXPECTED_CELLS} model×benchmark "
            f"rows are complete AND mutually compatible. Incomplete/incompatible "
            f"cells are listed at the bottom and are NOT tabulated. Rerun before "
            f"using as final.",
            "",
        ]
    lines += [
        "Abstention rate with the `Unknown` option at positions A, B, and C on "
        "FLD and FOLIO. All cells use the same 500 balanced samples (250 True + "
        "250 False) and the unified True/False/Unknown label set; C is measured "
        "directly (not taken from Table 1). Rows are labeled with the **actual "
        "API model id** that was queried.",
        "",
        "## Abstention Rate by Position",
        "",
        "| Model (API id) | Benchmark | A: Unknown | B: Unknown | C: Unknown | Max–Min (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    pos_means = {p: [] for p in POSITIONS}
    min_nvalid_overall = None
    for mk in DISPLAY_ORDER:
        for ds in DATASETS:
            r = by.get((mk, ds))
            if not r:
                continue
            lines.append(
                f"| `{r['model']}` | {ds} | {r['A']:.1f}% | {r['B']:.1f}% | "
                f"{r['C']:.1f}% | {r['max_minus_min_pp']:.1f} pp |"
            )
            for p in POSITIONS:
                pos_means[p].append(r[p])
            mnv = r.get("min_n_valid")
            if mnv is not None:
                min_nvalid_overall = mnv if min_nvalid_overall is None else min(min_nvalid_overall, mnv)
    if min_nvalid_overall is not None and min_nvalid_overall < 500:
        lines += [
            "",
            f"*Abs Rate is computed over the valid subset (n_valid ≥ "
            f"{min_nvalid_overall}/500). A small number of FLD prompts are "
            f"deterministically refused by the API content filter (\"Sensitive "
            f"word detected\") and are excluded from the denominator.*",
        ]
    lines += [
        "",
        "*Model ids are the exact strings sent to the API. If the paper uses "
        "different display names (e.g. \"DeepSeek-R1\", \"Gemini-3.1-Flash-"
        "Lite\"), map them deliberately — do not assume `deepseek-r1-distill-"
        "llama-8b` or `gemini-2.5-flash-lite` equal those names.*",
    ]
    if any(pos_means[p] for p in POSITIONS):
        lines += [
            "",
            "## Mean Abstention Rate",
            "",
            "| Unknown position | Mean Abs Rate |",
            "|---|---:|",
        ]
        for p in POSITIONS:
            vals = pos_means[p]
            mean = sum(vals) / len(vals) if vals else float("nan")
            lines.append(f"| {p} | {mean:.1f}% |")
    if incomplete:
        lines += ["", "## Incomplete cells (excluded — rerun before trusting)", ""]
        lines += [f"- {c}" for c in incomplete]
    if incompatible:
        lines += ["", "## Incompatible rows (excluded — stale/mismatched data)", ""]
        lines += [f"- {c}" for c in incompatible]
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def validate_group(model_name, group, expected_n=None, require_unified=True):
    """Check that the A/B/C cells of one model×dataset are safe to certify.

    A completion flag per cell is not enough, and neither is cross-position
    agreement (three cells that are all STALE the same way — e.g. all n=200, all
    missing unified_labels, all empty per_sample — would "agree" and slip
    through). So each cell is first checked in ABSOLUTE terms (well-formed,
    complete, right label mode, expected n, per_sample covers n), and only then
    cross-checked for pairing (same model, same n, same label mode, same sample
    id set — required for a meaningful A/B/C Max-Min). Fails CLOSED on anything
    missing or malformed. Returns (ok, reasons, shared_meta).
    """
    reasons = []
    present = {p: g for p, g in group.items() if isinstance(g, dict)}
    if len(present) < len(POSITIONS):
        miss = [p for p in POSITIONS if p not in present]
        return False, [f"missing/malformed positions {miss}"], {}

    # Absolute per-cell well-formedness (fail closed on missing/stale fields).
    for p, g in present.items():
        n = g.get("n")
        if not isinstance(n, int) or n <= 0:
            reasons.append(f"{p}: missing/invalid n ({n!r})")
        if g.get("complete") is not True:
            reasons.append(f"{p}: not complete")
        ps = g.get("per_sample")
        if not isinstance(ps, list) or not ps:
            reasons.append(f"{p}: missing/empty per_sample")
        elif isinstance(n, int) and len(ps) != n:
            reasons.append(f"{p}: per_sample len {len(ps)} != n {n}")
        if g.get("model") != model_name:
            reasons.append(f"{p}: model {g.get('model')!r} != expected {model_name!r}")
        if require_unified and g.get("unified_labels") is not True:
            reasons.append(f"{p}: unified_labels != True ({g.get('unified_labels')!r})")
        if expected_n is not None and n != expected_n:
            reasons.append(f"{p}: n {n!r} != expected {expected_n}")

    # Cross-position pairing (only meaningful once cells are well-formed).
    ns = {p: g.get("n") for p, g in present.items()}
    if len(set(ns.values())) > 1:
        reasons.append(f"n differs across positions: {ns}")
    ul = {p: g.get("unified_labels") for p, g in present.items()}
    if len(set(ul.values())) > 1:
        reasons.append(f"unified_labels differ across positions: {ul}")
    idsets = {
        p: frozenset(r.get("id") for r in (g.get("per_sample") or []))
        for p, g in present.items()
    }
    if len({s for s in idsets.values()}) > 1:
        sizes = {p: len(s) for p, s in idsets.items()}
        reasons.append(f"sample-id sets differ across positions (sizes {sizes})")

    shared = {"unified_labels": next(iter(ul.values())), "n": next(iter(ns.values()))}
    return (not reasons), reasons, shared


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def mcnemar_on_abstain(a_rows, b_rows):
    """Paired McNemar for abstain vs non-abstain decisions."""
    b_by_id = {r["id"]: r for r in b_rows}
    n10 = n01 = 0
    for row in a_rows:
        other = b_by_id.get(row["id"])
        if not other:
            continue
        a_abs = row["pred"] == "UNKNOWN"
        b_abs = other["pred"] == "UNKNOWN"
        if a_abs and not b_abs:
            n10 += 1
        elif (not a_abs) and b_abs:
            n01 += 1
    d = n10 + n01
    p = 1.0 if d == 0 else binomtest(n01, n=d, p=0.5, alternative="two-sided").pvalue
    return n10, n01, p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--result-dir", default=str(RESULT_DIR),
        help="Directory of summary_unknown_*.json files "
             "(e.g. results/positional_bias_n500 for the 500-sample run).",
    )
    ap.add_argument(
        "--no-c-fallback", action="store_true",
        help="Do not synthesize C from old AB summaries; require a real C run. "
             "Use for the n500 run where all three positions are measured.",
    )
    ap.add_argument(
        "--expect-n", type=int, default=500,
        help="Absolute per-cell sample count required to certify a row "
             "(guards against stale runs of a different size). Default 500.",
    )
    ap.add_argument(
        "--allow-native-labels", action="store_true",
        help="Permit non-unified (native Proved/Disproved/Uncertain) label "
             "summaries. By default the gate requires unified_labels=True.",
    )
    args = ap.parse_args()
    result_dir = Path(args.result_dir)
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    incomplete = []
    incomplete_keys = set()
    incompatible = []
    summaries = {}  # (model_key, ds, pos) -> summary, for cross-position validation
    print(f"Result dir: {result_dir}")
    print(
        f"{'Model':<10} {'DS':<6} {'Pos':<3} {'n':>4} "
        f"{'Abs Rate':>7} {'Acc':>7} {'F1':>7} "
        f"{'raw A/B/C':>15} {'counts A/B/U/UP':>18}"
    )
    print("-" * 96)
    for model_key, model_name in MODELS:
        for ds in DATASETS:
            loaded = {}
            for pos in POSITIONS:
                summary = load_summary(model_name, ds, pos, result_dir)
                if summary is None and pos == "C" and not args.no_c_fallback:
                    summary = load_c_fallback(model_key, model_name, ds)
                if not summary:
                    continue
                # Flag runs that did not finish cleanly so a partial result is
                # never silently reported as a final number.
                # Trust the runner's completion flag: it already accounts for
                # content-filter exclusions (a complete cell may have a few
                # refused samples that were dropped from the denominator, which
                # is fine — only a systemic failure sets complete=False).
                if summary.get("complete") is False:
                    incomplete_keys.add((model_key, ds, pos))
                    incomplete.append(
                        f"{model_key}/{ds}/U={pos} "
                        f"(complete=False, "
                        f"excluded={summary.get('excluded', '?')}, "
                        f"invalid={summary.get('invalid_responses', '?')})"
                    )
                loaded[pos] = summary
                summaries[(model_key, ds, pos)] = summary
                m = summary["metrics"]
                raw_counts = summary["raw_letter_counts"]
                counts = m["counts"]
                print(
                    f"{model_key:<10} {ds:<6} {pos:<3} {summary['n']:>4d} "
                    f"{m['abstain_rate']:>6.1%} {m['label_acc']:>6.1%} {m['label_f1']:>6.1%} "
                    f"{raw_counts.get('A', 0):>4}/{raw_counts.get('B', 0):<4}/{raw_counts.get('C', 0):<4} "
                    f"{counts['A']:>4}/{counts['B']:<4}/{counts['UNKNOWN']:<4}/{counts['UNPARSEABLE']:<4}"
                )
                rows.append({
                    "model": model_key,
                    "model_name": model_name,
                    "dataset": ds,
                    "position": pos,
                    "n": summary["n"],
                    "n_valid": summary.get("n_valid", summary["n"]),
                    "excluded": summary.get("excluded", 0),
                    "abstain_rate": m["abstain_rate"],
                    "label_acc": m["label_acc"],
                    "label_f1": m["label_f1"],
                    "counts": counts,
                    "raw_letter_counts": raw_counts,
                })

            if len(loaded) >= 2:
                print(f"  paired abstain McNemar ({model_key}/{ds}):")
                for left, right in [("A", "B"), ("A", "C"), ("B", "C")]:
                    if left in loaded and right in loaded:
                        n10, n01, p = mcnemar_on_abstain(
                            loaded[left]["per_sample"], loaded[right]["per_sample"]
                        )
                        print(
                            f"    U={left} vs U={right}: "
                            f"{left}-only={n10}, {right}-only={n01}, p={p:.4g} {sig(p)}"
                        )
            print()

    # Rebuttal table: A / B / C Abs Rate + Max-Min (pp) per model×dataset.
    print("\n" + "=" * 60)
    print("REBUTTAL TABLE  (Abs Rate by Unknown position)")
    print("=" * 60)
    print(f"{'Model':<28}{'Bench':<7}{'A':>7}{'B':>7}{'C':>7}{'Max-Min':>9}")
    print("-" * 65)
    abs_rate = {(r["model"], r["dataset"], r["position"]): r["abstain_rate"] for r in rows}
    n_by = {(r["model"], r["dataset"], r["position"]): r["n"] for r in rows}
    nvalid_by = {(r["model"], r["dataset"], r["position"]): r.get("n_valid", r["n"]) for r in rows}
    table_rows = []
    for model_key, model_name in MODELS:
        for ds in DATASETS:
            vals = {p: abs_rate.get((model_key, ds, p)) for p in POSITIONS}
            missing = [p for p in POSITIONS if vals[p] is None]
            incs = [p for p in POSITIONS if (model_key, ds, p) in incomplete_keys]
            # Never publish a number (or a Max-Min) for a row whose cells are
            # missing or did not finish cleanly — that is exactly the invalid
            # result the completion gate exists to suppress.
            if missing or incs:
                print(f"{model_name:<28}{ds:<7}  INCOMPLETE — rerun "
                      f"(missing={missing}, incomplete={incs})")
                continue
            # Even when all three cells are individually complete, refuse to
            # certify the row unless they are mutually compatible (same model,
            # label scheme, n, and sample set). Blocks stale/mixed data.
            group = {p: summaries.get((model_key, ds, p)) for p in POSITIONS}
            ok, reasons, shared = validate_group(
                model_name, group,
                expected_n=args.expect_n,
                require_unified=not args.allow_native_labels)
            if not ok:
                print(f"{model_name:<28}{ds:<7}  INCOMPATIBLE — {'; '.join(reasons)}")
                incompatible.append(f"{model_key}/{ds}: {'; '.join(reasons)}")
                continue
            pcts = {p: vals[p] * 100 for p in POSITIONS}
            span = max(pcts.values()) - min(pcts.values())
            ns = {p: n_by.get((model_key, ds, p)) for p in POSITIONS}
            nvs = {p: nvalid_by.get((model_key, ds, p)) for p in POSITIONS}
            min_nvalid = min(v for v in nvs.values() if v is not None)
            print(f"{model_name:<28}{ds:<7}"
                  f"{pcts['A']:>6.1f}%{pcts['B']:>6.1f}%{pcts['C']:>6.1f}%"
                  f"{span:>7.1f}pp   (n_valid≥{min_nvalid})")
            table_rows.append({
                "model_key": model_key, "model": model_name, "dataset": ds,
                "A": round(pcts["A"], 1), "B": round(pcts["B"], 1),
                "C": round(pcts["C"], 1), "max_minus_min_pp": round(span, 1),
                "n_per_position": ns, "n_valid_per_position": nvs,
                "min_n_valid": min_nvalid,
                "unified_labels": shared.get("unified_labels"),
            })

    if incomplete:
        print("\n*** WARNING: incomplete/errored cells (rerun before trusting): ***")
        for c in incomplete:
            print(f"    - {c}")
    if incompatible:
        print("\n*** WARNING: incompatible rows (stale/mismatched — excluded): ***")
        for c in incompatible:
            print(f"    - {c}")

    out_path = result_dir / "positional_bias_summary.json"
    out_path.write_text(json.dumps(
        {"rows": rows, "rebuttal_table": table_rows,
         "incomplete": incomplete, "incompatible": incompatible},
        indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out_path}")

    md_path = write_markdown_report(
        table_rows, incomplete, result_dir / "positional_bias_report.md",
        incompatible=incompatible)
    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
