"""
C3 Step 2: Collect Abs Rate and CAR samples directly from OLMo-3-Instruct inference output.
No NLI probe needed — simple label-based filtering.

  Abstention Inflation samples: answerable (gold != UNKNOWN), model outputs UNKNOWN in S2
  CAR samples: Unknown-labeled (gold == UNKNOWN), model correctly outputs UNKNOWN

Input:  results/S8_logit_lens/olmo_inference_FLD.json
Output: results/S8_logit_lens/c3_samples.json

Run:
    python scripts/c3_collect_samples.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

INFERENCE_PATH = ROOT / "results" / "S8_logit_lens" / "olmo_inference_FLD.json"
OUT_PATH       = ROOT / "results" / "S8_logit_lens" / "c3_samples.json"


def main():
    samples = json.loads(INFERENCE_PATH.read_text())
    print(f"Loaded {len(samples)} samples")

    ai_samples = []
    car_samples = []

    for s in samples:
        gold  = s["proof_label"]
        pred2 = s.get("pred_s2", "")

        if pred2 == "UNKNOWN" and gold != "__UNKNOWN__":
            ai_samples.append({
                "id":          s["id"],
                "proof_label": gold,
                "gold_short":  gold.replace("__", "").upper()[:9],   # PROVED / DISPROVED
                "Conclusion":  s["Conclusion"],
                "Facts":       s["Facts"],
                "raw_s2":      s.get("raw_s2", ""),
                "sample_type": "ai",
            })
        elif pred2 == "UNKNOWN" and gold == "__UNKNOWN__":
            car_samples.append({
                "id":          s["id"],
                "proof_label": gold,
                "gold_short":  "UNKNOWN",
                "Conclusion":  s["Conclusion"],
                "Facts":       s["Facts"],
                "raw_s2":      s.get("raw_s2", ""),
                "sample_type": "CAR",
            })

    # balance: use same N for both groups in logit-lens
    n = min(len(ai_samples), len(car_samples))
    car_balanced = car_samples[:n]

    print(f"Abstention Inflation samples: {len(ai_samples)}")
    print(f"CAR samples: {len(car_samples)}  (using {n} balanced)")

    output = {
        "abstention_inflation_samples":   ai_samples,
        "correct_abstention_samples":   car_samples,
        "correct_abstention_balanced":  car_balanced,
        "stats": {
            "n_abstention_inflation":         len(ai_samples),
            "n_car":         len(car_samples),
            "n_car_balanced": n,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
