"""Two-step CoT probe using tasksource/deberta-base-long-nli, kept inside the
model's actual training distribution.

Step 1 — VALIDATION (model competence on our data, native task framing):
    For each raw sample, run tasksource(Facts, Conclusion) and predict
    {entailment / neutral / contradiction} → mapped to {PROVED / UNKNOWN / DISPROVED}.
    Compare to gold proof_label. Report 3-way accuracy.
    → This is exactly the FOLIO/FLD task tasksource was trained on.

Step 2 — APPLICATION (γ probe, same model + same task framing):
    For each Abstention Inflation sample, run tasksource(CoT_body, Conclusion) for both raw_s1
    and raw_s2. Same 3-way label space. Categorize:
        γ = pred=UNKNOWN AND CoT entails/contradicts AND derived == gold
        β = pred=UNKNOWN AND CoT entails/contradicts AND derived != gold
        α = pred=UNKNOWN AND CoT neutral

The validation accuracy in Step 1 calibrates the application in Step 2 — we
only claim γ-rate up to the model's measured competence on the source task.

Usage:
    python -m scripts.probe_cot_tasksource \
        --summary results/ab_nano_batch2/ab_summary_FLD_gpt-5.4-nano.json \
        --dataset_json data/Judge/FLD.json \
        --validation_n 200 --n 999
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NLI_MODEL = "tasksource/deberta-base-long-nli"


# Standard NLI label names (read dynamically from model config; print if mismatch)
def label_to_idx(id2label: dict) -> dict:
    """Map canonical name → idx based on model's id2label."""
    out = {}
    for i, name in id2label.items():
        n = name.lower()
        if "entail" in n:
            out["entailment"] = int(i)
        elif "contradict" in n:
            out["contradiction"] = int(i)
        elif "neutral" in n:
            out["neutral"] = int(i)
    return out


def strip_final_answer(raw: str) -> str:
    if not raw: return ""
    m = re.search(r"\n?\s*final\s+answer\s*:", raw, re.IGNORECASE)
    return raw[:m.start()].strip() if m else raw.strip()


def take_tail(text: str, max_chars: int = 4000) -> str:
    """tasksource long-NLI handles ~1680 tokens; ~4000 chars is safe."""
    return text if len(text) <= max_chars else "..." + text[-max_chars:]


def gold_to_canonical(label: str) -> str:
    """__PROVED__ → entailment ; __DISPROVED__ → contradiction ; __UNKNOWN__ → neutral."""
    if not label: return "?"
    L = label.replace("_", "").upper()
    return {"PROVED": "entailment", "DISPROVED": "contradiction", "UNKNOWN": "neutral"}.get(L, "?")


def canonical_to_short(canonical: str) -> str:
    return {"entailment": "PROVED", "contradiction": "DISPROVED", "neutral": "NEUTRAL"}.get(canonical, "?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, nargs="+",
                    help="one or more ab_summary_*.json paths; per_sample lists are concatenated")
    ap.add_argument("--dataset_json", required=True)
    ap.add_argument("--n", type=int, default=999, help="max Abstention Inflation samples for CoT probe")
    ap.add_argument("--validation_n", type=int, default=200, help="raw samples for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"Loading {NLI_MODEL}")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
    if torch.cuda.is_available(): nli = nli.cuda()
    elif torch.backends.mps.is_available(): nli = nli.to("mps")
    device = next(nli.parameters()).device
    print(f"Device: {device}")
    print(f"Model id2label: {nli.config.id2label}")
    LBL = label_to_idx(nli.config.id2label)
    print(f"Canonical mapping → {LBL}")

    def classify(premise: str, hypothesis: str) -> dict:
        inputs = tok(premise, hypothesis, truncation=True, max_length=1680,
                     return_tensors="pt").to(device)
        with torch.no_grad():
            logits = nli(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        # find which canonical name has the highest prob
        scores = {name: probs[idx] for name, idx in LBL.items()}
        top = max(scores, key=scores.get)
        return {"canonical": top, "scores": scores, "confidence": scores[top]}

    # =================================================================
    # STEP 1 — Validation on raw dataset (model's native task)
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 1 — Validation: tasksource on raw (Facts, Conclusion) pairs")
    print("=" * 70)
    all_raw_items = json.loads(Path(args.dataset_json).read_text())
    raw_items = all_raw_items[: args.validation_n]  # validation only takes first N
    print(f"Loaded {len(raw_items)} raw samples for validation (out of {len(all_raw_items)} total)")

    val_rows = []
    for i, r in enumerate(raw_items):
        gold_label = r.get("proof_label")
        if not gold_label: continue
        gold_canon = gold_to_canonical(gold_label)
        if gold_canon == "?": continue
        facts = r.get("Facts", "")
        concl = r.get("Conclusion", "")
        if not facts or not concl: continue
        res = classify(facts, concl)
        val_rows.append({
            "idx": i, "gold": gold_canon, "pred": res["canonical"],
            "correct": res["canonical"] == gold_canon,
            "confidence": res["confidence"],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(raw_items)} processed")

    n_val = len(val_rows)
    val_acc = sum(1 for r in val_rows if r["correct"]) / max(n_val, 1)
    print(f"\nValidation: {sum(1 for r in val_rows if r['correct'])}/{n_val} = {val_acc:.1%}")
    # per-class breakdown
    for canon in ("entailment", "neutral", "contradiction"):
        sub = [r for r in val_rows if r["gold"] == canon]
        if sub:
            acc = sum(1 for r in sub if r["correct"]) / len(sub)
            short = canonical_to_short(canon)
            print(f"  gold={short:<10} n={len(sub):>3}  acc={acc:.1%}")

    # =================================================================
    # STEP 2 — Apply same model+task to (CoT, Conclusion) on Abstention Inflation samples
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2 — Apply: tasksource on (CoT_body, Conclusion) for Abstention Inflation samples")
    print("=" * 70)
    per_sample = []
    model_name = None; dataset = None
    for sp in args.summary:
        summary = json.loads(Path(sp).read_text())
        per_sample.extend(summary.get("per_sample", []))
        model_name = summary.get("model")
        dataset    = summary.get("dataset")
        print(f"  loaded {sp}: +{len(summary.get('per_sample', []))} samples")
    print(f"Pooled per_sample: model={model_name} dataset={dataset} n={len(per_sample)}")

    # Build id → (Conclusion, gold) map FROM FULL DATASET (not just validation subset)
    id_map = {}
    for i, r in enumerate(all_raw_items):
        sid = f"{dataset}_{i:04d}"
        id_map[sid] = {"Conclusion": r.get("Conclusion", ""), "proof_label": r.get("proof_label")}
    print(f"Built id→meta map for {len(id_map)} raw samples")

    ai_samples = [s for s in per_sample if s.get("pred_s2") == "UNKNOWN"]
    print(f"Abs Rate (S2=UNKNOWN) samples available: {len(ai_samples)}")

    import random
    random.seed(args.seed)
    n_pick = min(args.n, len(ai_samples))
    picked = random.sample(ai_samples, n_pick) if n_pick < len(ai_samples) else ai_samples
    print(f"Probing {n_pick} Abstention Inflation samples (paired S1+S2 = {2*n_pick} NLI calls)")

    rows = []
    for i, s in enumerate(picked):
        sid = s["id"]
        meta = id_map.get(sid)
        if meta is None: continue
        hypothesis = meta["Conclusion"]
        gold_canon = gold_to_canonical(meta["proof_label"])
        if gold_canon == "?": continue
        for setting in ("s1", "s2"):
            raw = s.get(f"raw_{setting}", "") or ""
            cot = take_tail(strip_final_answer(raw))
            if not cot: continue
            pred = s.get(f"pred_{setting}", "")
            res = classify(cot, hypothesis)
            derived_canon = res["canonical"]
            derived_eq_gold = derived_canon == gold_canon
            cot_decisive_but_pred_unk = (
                setting == "s2" and pred == "UNKNOWN"
                and derived_canon in ("entailment", "contradiction")
            )
            rows.append({
                "id": sid, "setting": setting.upper(), "pred": pred,
                "gold": canonical_to_short(gold_canon),
                "derived": canonical_to_short(derived_canon),
                "confidence": res["confidence"],
                "derived_eq_gold": derived_eq_gold,
                "cot_decisive_but_pred_unk": cot_decisive_but_pred_unk,
                "scores": res["scores"],
            })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_pick} samples processed")

    # ---- aggregate ----
    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    s1 = [r for r in rows if r["setting"] == "S1"]
    s2 = [r for r in rows if r["setting"] == "S2"]

    def agg(label, rs, is_s2=False):
        n = len(rs)
        if n == 0: return None
        decisive = sum(1 for r in rs if r["derived"] in ("PROVED", "DISPROVED"))
        decisive_correct = sum(1 for r in rs if r["derived"] in ("PROVED", "DISPROVED")
                                              and r["derived_eq_gold"])
        neutral = sum(1 for r in rs if r["derived"] == "NEUTRAL")
        gamma_total = sum(1 for r in rs if r["cot_decisive_but_pred_unk"])
        gamma_correct = sum(1 for r in rs if r["cot_decisive_but_pred_unk"]
                                            and r["derived_eq_gold"])
        beta = gamma_total - gamma_correct
        alpha = sum(1 for r in rs if r["setting"] == "S2" and r["pred"] == "UNKNOWN"
                                    and r["derived"] == "NEUTRAL")
        print(f"\n{label}  (n={n})")
        print(f"  CoT decisive (P or D):       {decisive}/{n} = {decisive/n:.1%}")
        print(f"    of which match gold:       {decisive_correct}/{decisive} ({decisive_correct/max(decisive,1):.1%})")
        print(f"  CoT NEUTRAL:                 {neutral}/{n} = {neutral/n:.1%}")
        if is_s2:
            print(f"  γ  (decisive AND correct, pred=UNK):    {gamma_correct}/{n} = {gamma_correct/n:.1%}")
            print(f"  β  (decisive but WRONG, pred=UNK):       {beta}/{n} = {beta/n:.1%}")
            print(f"  α  (CoT neutral, pred=UNK):              {alpha}/{n} = {alpha/n:.1%}")
        return {
            "n": n, "decisive": decisive, "decisive_correct": decisive_correct,
            "neutral": neutral, "gamma_correct": gamma_correct, "beta": beta, "alpha": alpha,
        }

    s1_stats = agg("S1 (no Unknown option)", s1, is_s2=False)
    s2_stats = agg("S2 (with Unknown, Abstention Inflation subset)", s2, is_s2=True)

    # ---- save ----
    out = args.out or f"results/probe/cot_tasksource_{model_name}_{dataset}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "nli_model": NLI_MODEL,
        "model": model_name, "dataset": dataset,
        "validation": {
            "n": n_val, "accuracy": val_acc,
            "per_class": {canonical_to_short(c): {
                "n": sum(1 for r in val_rows if r["gold"] == c),
                "acc": sum(1 for r in val_rows if r["gold"] == c and r["correct"]) /
                       max(sum(1 for r in val_rows if r["gold"] == c), 1),
            } for c in ("entailment", "neutral", "contradiction")},
        },
        "cot_probe": {"n_paired": n_pick, "s1_stats": s1_stats, "s2_stats": s2_stats},
        "rows": rows,
    }, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
