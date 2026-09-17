"""Analyze how MedQA-USMLE is categorized."""

import json
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "raw dataset"


def analyze_medqa():
    print("\n[MedQA-USMLE-4-options]")
    rows = json.loads((DATA / "medqa_usmle_4opt.json").read_text())
    by_split = Counter(r["split"] for r in rows)
    print(f"splits: {dict(by_split)}")
    print(f"total: {len(rows)}")

    meta_counter = Counter(r.get("meta_info") for r in rows)
    print("\nmeta_info distribution (the only categorical label MedQA exposes):")
    for k, v in meta_counter.most_common():
        print(f"  {k!r:<30} {v}")

    by_split_meta = Counter((r["split"], r.get("meta_info")) for r in rows)
    print("\nby (split, meta_info):")
    for (split, meta), v in sorted(by_split_meta.items()):
        print(f"  {split:<6} {str(meta):<25} {v}")

    sample = rows[0]
    print("\nFields:", list(sample.keys()))
    print(f"answer_idx domain: {sorted(set(r['answer_idx'] for r in rows))}")
    n_opts = Counter(len(r["options"]) for r in rows)
    print(f"#options per question: {dict(n_opts)}")
    sample_opts = sample["options"]
    print(f"option keys (sample): {list(sample_opts.keys())}")


if __name__ == "__main__":
    analyze_medqa()
