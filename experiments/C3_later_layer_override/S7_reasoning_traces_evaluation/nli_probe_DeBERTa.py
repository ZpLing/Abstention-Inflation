"""Probe whether the model's CoT (raw_s1 / raw_s2) already derives a definitive
PROVED / DISPROVED conclusion, using a pretrained NLI encoder.

Approach (no fine-tuning, no LLM judge):
    premise    = CoT body (raw_s1 or raw_s2 with the "Final answer: X" line stripped)
    hypothesis = sample.Conclusion (the claim being judged)
    NLI labels:
        entailment    → CoT derives PROVED   (CoT supports hypothesis)
        contradiction → CoT derives DISPROVED (CoT refutes hypothesis)
        neutral       → CoT genuinely undecided (UNKNOWN-like)

Paired design (default): pick N Abstention Inflation samples (S2 → UNKNOWN), read both raw_s1
and raw_s2 for each → 2N NLI calls → compare derived label across settings.

Usage:
    python -m scripts.probe_cot_conclusion_nli \
        --summary results/ab_nano_batch2/ab_summary_FOLIO_gpt-5.4-nano.json \
        --dataset_json data/Judge/FOLIO.json \
        --n 30
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# label index → name (HF mnli order)
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
DERIVED_FROM_NLI = {
    "entailment":    "PROVED",
    "contradiction": "DISPROVED",
    "neutral":       "NEUTRAL",
}


def strip_final_answer(raw: str) -> str:
    """Remove the trailing 'Final answer: X' line so NLI sees CoT only."""
    if not raw:
        return ""
    # cut at the last "Final answer" / "final answer:" if present
    m = re.search(r"\n?\s*final\s+answer\s*:", raw, re.IGNORECASE)
    if m:
        return raw[:m.start()].strip()
    return raw.strip()


def take_tail(text: str, max_chars: int = 1500) -> str:
    """NLI is 512-token limited; keep the last ~1500 chars (~300-400 tok) of CoT."""
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def load_dataset_id_map(dataset_json: Path) -> dict:
    """Map id → (Conclusion, proof_label) for FOLIO/FLD-style files."""
    items = json.loads(dataset_json.read_text())
    out = {}
    for i, r in enumerate(items):
        sid = f"{dataset_json.stem.upper()}_{i:04d}"  # e.g. FOLIO_0000
        out[sid] = {
            "Conclusion": r.get("Conclusion", ""),
            "proof_label": r.get("proof_label"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="ab_summary_*.json path")
    ap.add_argument("--dataset_json", required=True, help="data/Judge/<DS>.json path")
    ap.add_argument("--n", type=int, default=30, help="samples per setting")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Loading NLI model: {NLI_MODEL}")
    print("(downloads ~750MB on first run)")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    if torch.cuda.is_available():
        nli = nli.cuda()
    elif torch.backends.mps.is_available():
        nli = nli.to("mps")
    print(f"Model loaded on: {next(nli.parameters()).device}")

    # ---- load data ----
    summary = json.loads(Path(args.summary).read_text())
    id_map  = load_dataset_id_map(Path(args.dataset_json))
    per_sample = summary.get("per_sample", [])
    model_name = summary.get("model")
    dataset    = summary.get("dataset")
    print(f"\nSummary: model={model_name} dataset={dataset} n={len(per_sample)}")

    # ---- pick Abstention Inflation samples (paired design) ----
    ai_samples = [s for s in per_sample if s.get("pred_s2") == "UNKNOWN"]
    print(f"Abs Rate (S2=UNKNOWN) samples available: {len(ai_samples)}")

    import random
    random.seed(args.seed)
    n_pick = min(args.n, len(ai_samples))
    picked = random.sample(ai_samples, n_pick) if n_pick < len(ai_samples) else ai_samples
    print(f"Picked {n_pick} Abstention Inflation samples (paired across S1+S2 = {2*n_pick} NLI calls)")

    def nli_classify(premise: str, hypothesis: str) -> dict:
        inputs = tok(premise, hypothesis, truncation=True, max_length=512,
                     return_tensors="pt").to(next(nli.parameters()).device)
        with torch.no_grad():
            logits = nli(**inputs).logits[0]
            probs  = torch.softmax(logits, dim=-1).cpu().tolist()
        idx = max(range(3), key=lambda i: probs[i])
        return {
            "label":  LABEL_NAMES[idx],
            "derived": DERIVED_FROM_NLI[LABEL_NAMES[idx]],
            "p_entail":  probs[0],
            "p_neutral": probs[1],
            "p_contra":  probs[2],
            "confidence": probs[idx],
        }

    # ---- score ----
    rows = []
    for s in picked:
        sid       = s["id"]
        meta      = id_map.get(sid)
        if meta is None:
            continue
        hypothesis = meta["Conclusion"]
        gold       = meta["proof_label"]  # __PROVED__ / __DISPROVED__ / __UNKNOWN__
        gold_short = gold.replace("_", "").upper() if gold else "?"

        for setting in ("s1", "s2"):
            raw = s.get(f"raw_{setting}", "") or ""
            cot = take_tail(strip_final_answer(raw))
            if not cot:
                continue
            pred = s.get(f"pred_{setting}", "")
            res = nli_classify(cot, hypothesis)

            # Match: did NLI-derived match gold?
            derived_eq_gold = (res["derived"] == gold_short)
            # Mismatch flag: pred says UNKNOWN but NLI derived a definitive label
            cot_decisive_but_pred_unk = (
                setting == "s2"
                and pred == "UNKNOWN"
                and res["derived"] in ("PROVED", "DISPROVED")
            )

            rows.append({
                "id":         sid,
                "setting":    setting.upper(),
                "pred":       pred,
                "gold":       gold_short,
                "nli_label":  res["label"],
                "nli_derived": res["derived"],
                "confidence": res["confidence"],
                "derived_eq_gold":          derived_eq_gold,
                "cot_decisive_but_pred_unk": cot_decisive_but_pred_unk,
                "p_entail":  res["p_entail"],
                "p_neutral": res["p_neutral"],
                "p_contra":  res["p_contra"],
            })

    # ---- print per-row ----
    print(f"\n{'ID':<14} {'Set':<3} {'pred':<10} {'gold':<10} {'NLI':<10} {'derived':<10} {'conf':>5} {'γ?':>3}")
    print("-" * 90)
    for r in rows:
        gflag = "✓" if r["cot_decisive_but_pred_unk"] else ""
        print(f"{r['id']:<14} {r['setting']:<3} {r['pred']:<10} {r['gold']:<10} "
              f"{r['nli_label']:<10} {r['nli_derived']:<10} {r['confidence']:>5.2f} {gflag:>3}")

    # ---- aggregate ----
    print("\n" + "=" * 70)
    s1 = [r for r in rows if r["setting"] == "S1"]
    s2 = [r for r in rows if r["setting"] == "S2"]

    def agg(label, rs):
        n = len(rs)
        if n == 0: return
        decisive = sum(1 for r in rs if r["nli_derived"] in ("PROVED", "DISPROVED"))
        decisive_correct = sum(1 for r in rs if r["nli_derived"] in ("PROVED", "DISPROVED")
                                              and r["derived_eq_gold"])
        neutral = sum(1 for r in rs if r["nli_derived"] == "NEUTRAL")
        gamma = sum(1 for r in rs if r["cot_decisive_but_pred_unk"])
        gamma_correct = sum(1 for r in rs if r["cot_decisive_but_pred_unk"]
                                            and r["derived_eq_gold"])
        print(f"\n{label}  (n={n})")
        print(f"  CoT derived a definitive label (P or D):  {decisive}/{n} = {decisive/n:.1%}")
        print(f"    of which match gold:                    {decisive_correct}/{decisive} ({decisive_correct/max(decisive,1):.1%})")
        print(f"  CoT was NEUTRAL (no derivation):          {neutral}/{n} = {neutral/n:.1%}")
        if label == "S2":
            print(f"  γ-flag (CoT decisive BUT pred=UNKNOWN):   {gamma}/{n} = {gamma/n:.1%}")
            print(f"    of which CoT-derived also matches gold: {gamma_correct}/{gamma} ({gamma_correct/max(gamma,1):.1%})")
            print(f"    → γ-correct = 'reasoning correct, output-layer flip':    {gamma_correct}/{n} = {gamma_correct/n:.1%}")

    agg("S1 (no Unknown option)", s1)
    agg("S2 (with Unknown, Abstention Inflation subset)", s2)

    # ---- save ----
    out = args.out or f"results/probe/cot_nli_{model_name}_{dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model":    model_name,
        "dataset":  dataset,
        "nli_model": NLI_MODEL,
        "n_paired": n_pick,
        "rows":     rows,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
