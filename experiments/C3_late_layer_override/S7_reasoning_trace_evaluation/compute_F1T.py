"""Recompute trace_f1 for existing FOLIO ab_summary_*.json files.

After P-FOLIO alignment + trace_extractors.folio_* registration, FOLIO is now
a SOFT (BERTScore) trace family. Existing summaries stored raw_s1/s2/s3/s5 per
sample, so we can rescore offline without re-running any LLM calls.

Usage:
    python -m scripts.recompute_folio_trace_f1
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import metrics, trace_extractors  # noqa: E402
from core.dataset_loader import Sample  # noqa: E402


FOLIO_JSON = ROOT / "data" / "Judge" / "FOLIO.json"


def _build_id_to_extra(folio_path: Path):
    """Map FOLIO_xxxx id → extra dict containing gold_proof_steps."""
    samples = json.loads(folio_path.read_text())
    out = {}
    for i, s in enumerate(samples):
        sid = f"FOLIO_{i:04d}"
        out[sid] = {
            "Conclusion": s.get("Conclusion", ""),
            "Facts":      s.get("Facts", ""),
            "gold_proof_steps": s.get("gold_proof_steps"),
        }
    return out


def _make_sample(rec, extra) -> Sample:
    return Sample(
        id=rec["id"],
        source="FOLIO",
        question=extra.get("Conclusion", ""),
        options=[],
        answer_idx=rec.get("answer_idx", -1),
        extra=extra,
    )


def rescore_one(summary_path: Path, id_to_extra) -> dict:
    data = json.loads(summary_path.read_text())
    if data.get("dataset") != "FOLIO":
        return None
    per_sample = data.get("per_sample", [])
    if not per_sample:
        return None

    samples = []
    raws = {"s1": [], "s2": [], "s3": [], "s5": []}
    for ps in per_sample:
        sid = ps.get("id")
        extra = id_to_extra.get(sid)
        if extra is None:
            continue
        samples.append(_make_sample(ps, extra))
        for k in raws:
            raws[k].append(ps.get(f"raw_{k}", "") or "")

    if not samples:
        return None

    # Coverage
    n_with_gold = sum(1 for s in samples if s.extra.get("gold_proof_steps"))

    gold_texts = [trace_extractors.extract_gold(s) for s in samples]

    out = {"n_total_in_summary": len(per_sample),
           "n_aligned":           len(samples),
           "n_with_gold_proof":   n_with_gold}

    for setting in ("s1", "s2", "s3", "s5"):
        pred_texts = [trace_extractors.extract_pred(r, s)
                      for r, s in zip(raws[setting], samples)]
        f1 = metrics.mean_trace_bertscore_f1(pred_texts, gold_texts)
        out[f"trace_f1_{setting.upper()}"] = round(float(f1), 4)

    # Update the summary file in-place: write new trace_f1 into metrics block.
    metr = data.get("metrics", {})
    for setting in ("S1", "S2", "S3", "S5"):
        if setting in metr:
            metr[setting]["trace_f1"] = out[f"trace_f1_{setting}"]
    data["trace_family"] = "soft"
    summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


def main():
    if not FOLIO_JSON.exists():
        print(f"ERROR: {FOLIO_JSON} not found", file=sys.stderr)
        sys.exit(1)
    id_to_extra = _build_id_to_extra(FOLIO_JSON)
    print(f"Loaded gold_proof_steps for {sum(1 for v in id_to_extra.values() if v['gold_proof_steps'])} / {len(id_to_extra)} FOLIO samples\n")

    targets = sorted(ROOT.glob("results/**/ab_summary_FOLIO*.json"))
    if not targets:
        print("No FOLIO summary files found.")
        return
    for path in targets:
        print(f"--- {path.relative_to(ROOT)} ---")
        result = rescore_one(path, id_to_extra)
        if result is None:
            print("  (skipped)")
            continue
        for k, v in result.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
