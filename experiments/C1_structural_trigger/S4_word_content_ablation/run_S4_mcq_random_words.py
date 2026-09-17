"""MCQ Latent Label Mapping experiment.

Tests whether the Latent Label Mapping effect (§4.3 RPC) generalises from
T/F tasks to MCQ tasks, and whether "None of the above" (NOTA) vs. a random
word triggers the same escape-valve response.

Conditions (all keeping A/B/C/D intact, adding one fifth option E):
    S1   — A/B/C/D only (baseline, no fifth option)
    S2   — A/B/C/D/E. Unknown          (main experiment reference)
    NOTA — A/B/C/D/E. None of the above
    C2   — A/B/C/D/E. Triangular       (random control word 1)
    C3   — A/B/C/D/E. Cerulean         (random control word 2)

Correct answer is ALWAYS one of A/B/C/D — choosing E is always wrong.

Key metric: OPT_X rate (% choosing E), split into:
    explicit_opt_x  — model output E or wrote the fifth-option text
    ghost_abstain   — model wrote "Unknown"/"Uncertain" when fifth option was NOT Unknown

Datasets: ARC-Challenge_250 (n=200) + MedQA (n=200)
Model:    deepseek-v4-flash (configurable)

Usage:
    python -m scripts.run_mcq_latent_label_mapping \
        --config configs/mcq_latent_label_mapping_deepseek.yaml
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from core.config_loader import load_config
from core.data_handler import DataHandler
from core.llm_handler import LLMHandler
from core.dataset_loader import load_arc, load_medqa
from core.prompts import (
    build_mcq_s1_prompt,
    build_mcq_s2_prompt,
    build_mcq_s4_word_prompt,
)

FIFTH_OPTIONS = {
    "S2_Unknown":    "Unknown",
    "NOTA":          "None of the above",
    "C2_Triangular": "Triangular",
    "C3_Cerulean":   "Cerulean",
}


# =============================================================================
# Parsers
# =============================================================================

_FINAL_ANSWER_RE = re.compile(
    r"Final answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_FINAL_ANSWER_PREFIX = re.compile(r"^final answer\s*:\s*", re.IGNORECASE)


def _extract_final_answer_line(text: str) -> str:
    m = _FINAL_ANSWER_RE.search(text)
    if not m:
        return ""
    return _FINAL_ANSWER_PREFIX.sub("", m.group(0)).strip()


def _parse_letter(text: str, valid: str) -> str:
    """Return the first letter matching `valid` pattern, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    upper = text.strip().upper()
    m = re.match(rf"^\(?\s*([{valid}])\s*[\.\):,\s]", upper)
    if m:
        return m.group(1)
    m = re.match(rf"^\(?\s*([{valid}])\s*\)?$", upper)
    if m:
        return m.group(1)
    m = re.search(rf"ANSWER\s*(?:IS|:|=)?\s*\(?\s*([{valid}])\b", upper)
    if m:
        return m.group(1)
    m = re.search(rf"\b([{valid}])\b", upper)
    if m:
        return m.group(1)
    return None


def parse_s1_output(text: str) -> str:
    """Binary parser. Returns letter A-D or 'UNPARSEABLE'."""
    if not text or not text.strip():
        return "UNPARSEABLE"
    final = _extract_final_answer_line(text)
    letter = _parse_letter(final or text, "ABCD")
    return letter if letter else "UNPARSEABLE"


def parse_control_output(text: str, fifth_option: str) -> Tuple[str, str]:
    """Parser for conditions with a fifth option (S2, NOTA, C2, C3).

    Returns (pred, category):
        pred     ∈ {"A","B","C","D","OPT_X","UNPARSEABLE"}
        category ∈ {"explicit_opt_x",  model said E or the fifth_option word
                    "ghost_abstain",   model said Unknown/Uncertain when fifth≠Unknown
                    "letter",          A/B/C/D
                    "unparseable"}
    """
    if not text or not text.strip():
        return "UNPARSEABLE", "unparseable"

    final = _extract_final_answer_line(text)
    target = final if final else text

    # Letter E → explicit fifth option
    if _parse_letter(target, "E") == "E":
        return "OPT_X", "explicit_opt_x"

    # Fifth-option text match (handles "None of the above", "Triangular", etc.)
    if re.search(re.escape(fifth_option), target, re.IGNORECASE):
        return "OPT_X", "explicit_opt_x"

    # Ghost abstain: model fell back to Unknown/Uncertain vocabulary
    # Only relevant when the offered fifth option is NOT already Unknown
    if fifth_option.lower() not in ("unknown", "uncertain"):
        if re.search(r"\bunknown\b|\buncertain\b", target, re.IGNORECASE):
            return "OPT_X", "ghost_abstain"

    # Standard A–D
    letter = _parse_letter(target, "ABCD")
    if letter:
        return letter, "letter"

    # Fallback: scan whole text
    letter = _parse_letter(text, "ABCDE")
    if letter == "E":
        return "OPT_X", "explicit_opt_x"
    if letter in ("A", "B", "C", "D"):
        return letter, "letter"

    return "UNPARSEABLE", "unparseable"


# =============================================================================
# Metrics
# =============================================================================

def compute_acc(preds: List[str], answer_idxs: List[int]) -> float:
    if not preds:
        return 0.0
    correct = sum(
        p == chr(ord("A") + ai)
        for p, ai in zip(preds, answer_idxs)
    )
    return correct / len(preds)


def condition_metrics(preds: List[str], answer_idxs: List[int],
                      categories: List[str] = None) -> Dict:
    n = len(preds)
    acc = compute_acc(preds, answer_idxs)
    n_opt_x = sum(1 for p in preds if p == "OPT_X")
    result = {
        "n": n,
        "label_acc": round(acc, 4),
        "opt_x_rate": round(n_opt_x / n, 4),
        "n_opt_x": n_opt_x,
        "n_unparseable": sum(1 for p in preds if p == "UNPARSEABLE"),
    }
    if categories:
        result["n_explicit_opt_x"] = categories.count("explicit_opt_x")
        result["n_ghost_abstain"] = categories.count("ghost_abstain")
    return result


# =============================================================================
# Dataset loading
# =============================================================================

def load_mcq_samples(ds_name: str, n: int):
    if ds_name.startswith("ARC-"):
        samples = load_arc(ds_name)
    elif ds_name.startswith("MedQA"):
        samples = load_medqa(ds_name)
    else:
        raise ValueError(f"Unsupported MCQ dataset: {ds_name}")
    # Answerable only (answer_idx >= 0), first n
    answerable = [s for s in samples if s.answer_idx >= 0]
    return answerable[:n]


# =============================================================================
# Per-dataset runner
# =============================================================================

async def run_one_dataset(ds_name: str, n_samples: int,
                          llm_handler: LLMHandler, results_dir: Path,
                          model_name: str) -> Dict:
    samples = load_mcq_samples(ds_name, n_samples)
    print(f"  Loaded {len(samples)} answerable samples.")
    n = len(samples)
    answer_idxs = [s.answer_idx for s in samples]

    # Build prompts for all 5 conditions
    s1_prompts    = [build_mcq_s1_prompt(s.question, s.options) for s in samples]
    s2_prompts    = [build_mcq_s2_prompt(s.question, s.options) for s in samples]
    nota_prompts  = [build_mcq_s4_word_prompt(s.question, s.options, FIFTH_OPTIONS["NOTA"])
                     for s in samples]
    c2_prompts    = [build_mcq_s4_word_prompt(s.question, s.options, FIFTH_OPTIONS["C2_Triangular"])
                     for s in samples]
    c3_prompts    = [build_mcq_s4_word_prompt(s.question, s.options, FIFTH_OPTIONS["C3_Cerulean"])
                     for s in samples]

    print(f"  Querying S1 / S2 / NOTA / C2 / C3 in parallel ({n} samples each) ...")
    raw_s1, raw_s2, raw_nota, raw_c2, raw_c3 = await asyncio.gather(
        llm_handler.batch_query(s1_prompts),
        llm_handler.batch_query(s2_prompts),
        llm_handler.batch_query(nota_prompts),
        llm_handler.batch_query(c2_prompts),
        llm_handler.batch_query(c3_prompts),
    )

    # Parse
    preds_s1 = [parse_s1_output(r) for r in raw_s1]

    s2_parsed   = [parse_control_output(r, FIFTH_OPTIONS["S2_Unknown"])    for r in raw_s2]
    nota_parsed = [parse_control_output(r, FIFTH_OPTIONS["NOTA"])          for r in raw_nota]
    c2_parsed   = [parse_control_output(r, FIFTH_OPTIONS["C2_Triangular"]) for r in raw_c2]
    c3_parsed   = [parse_control_output(r, FIFTH_OPTIONS["C3_Cerulean"])   for r in raw_c3]

    preds_s2,   cats_s2   = zip(*s2_parsed)   if s2_parsed   else ([], [])
    preds_nota, cats_nota = zip(*nota_parsed) if nota_parsed else ([], [])
    preds_c2,   cats_c2   = zip(*c2_parsed)   if c2_parsed   else ([], [])
    preds_c3,   cats_c3   = zip(*c3_parsed)   if c3_parsed   else ([], [])

    s1_acc = compute_acc(preds_s1, answer_idxs)
    m_s2   = condition_metrics(list(preds_s2),   answer_idxs, list(cats_s2))
    m_nota = condition_metrics(list(preds_nota), answer_idxs, list(cats_nota))
    m_c2   = condition_metrics(list(preds_c2),   answer_idxs, list(cats_c2))
    m_c3   = condition_metrics(list(preds_c3),   answer_idxs, list(cats_c3))

    # Print summary table
    print(f"\n  {ds_name} results (S1_Acc={s1_acc:.1%}):")
    print(f"  {'Condition':<30} {'OPT_X Rate':>12} {'Acc':>8} {'ΔAcc':>8}")
    print(f"  {'-'*62}")
    print(f"  {'S1 (no 5th option)':<30} {'—':>12} {s1_acc:>8.1%} {'—':>8}")
    for label, m in [
        ("S2 (E. Unknown)",            m_s2),
        ("NOTA (E. None of the above)", m_nota),
        (f"C2 (E. Triangular)",         m_c2),
        (f"C3 (E. Cerulean)",           m_c3),
    ]:
        delta = m["label_acc"] - s1_acc
        print(f"  {label:<30} {m['opt_x_rate']:>12.1%} {m['label_acc']:>8.1%} {delta:>+8.1%}")

    result = {
        "dataset":    ds_name,
        "model":      model_name,
        "n_samples":  n,
        "S1_acc":     round(s1_acc, 4),
        "S2_Unknown": {**m_s2,   "delta_acc": round(m_s2["label_acc"]   - s1_acc, 4)},
        "NOTA":       {**m_nota, "delta_acc": round(m_nota["label_acc"] - s1_acc, 4)},
        "C2_Triangular": {**m_c2, "delta_acc": round(m_c2["label_acc"] - s1_acc, 4)},
        "C3_Cerulean":   {**m_c3, "delta_acc": round(m_c3["label_acc"] - s1_acc, 4)},
        "per_sample": [
            {
                "id":         samples[i].id,
                "answer_idx": answer_idxs[i],
                "pred_s1":    preds_s1[i],
                "pred_s2":    preds_s2[i],    "cat_s2":   cats_s2[i],
                "pred_nota":  preds_nota[i],  "cat_nota": cats_nota[i],
                "pred_c2":    preds_c2[i],    "cat_c2":   cats_c2[i],
                "pred_c3":    preds_c3[i],    "cat_c3":   cats_c3[i],
                "raw_s1":     raw_s1[i],
                "raw_s2":     raw_s2[i],
                "raw_nota":   raw_nota[i],
                "raw_c2":     raw_c2[i],
                "raw_c3":     raw_c3[i],
            }
            for i in range(n)
        ],
    }

    safe_model = model_name.replace("/", "_")
    safe_ds = ds_name.replace("-", "_").replace("/", "_")
    out_path = results_dir / f"mcq_llm_{safe_ds}_{safe_model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}")
    return result


# =============================================================================
# Main
# =============================================================================

async def run_experiment(config: Dict):
    cfg = config.get("mcq_latent_label_mapping", {})
    datasets  = cfg.get("datasets", ["ARC-Challenge_250", "MedQA"])
    n_samples = cfg.get("n_samples", 200)
    results_dir = Path(cfg.get("results_dir", "results/mcq_latent_label_mapping"))
    results_dir.mkdir(parents=True, exist_ok=True)

    data_handler = DataHandler(config)
    llm_handler  = LLMHandler(config)
    model_name   = config.get("model_name", "unknown")

    print(f"Model: {model_name}")
    print(f"n_samples per dataset: {n_samples}")
    print(f"Fifth options: {FIFTH_OPTIONS}")

    all_results = {}
    for ds_name in datasets:
        print(f"\n===== {ds_name} =====")
        all_results[ds_name] = await run_one_dataset(
            ds_name, n_samples, llm_handler, results_dir, model_name
        )

    # Pooled summary
    print("\n\n===== POOLED SUMMARY =====")
    print(f"  {'Condition':<30} {'OPT_X Rate':>12} {'Acc':>8} {'ΔAcc':>8}")
    print(f"  {'-'*62}")
    for cond_key, label in [
        ("S2_Unknown",    "S2 (Unknown)"),
        ("NOTA",          "NOTA (None of the above)"),
        ("C2_Triangular", "C2 (Triangular)"),
        ("C3_Cerulean",   "C3 (Cerulean)"),
    ]:
        rates, accs, deltas, ns = [], [], [], []
        for r in all_results.values():
            m = r[cond_key]
            rates.append(m["opt_x_rate"] * m["n"])
            accs.append(m["label_acc"]   * m["n"])
            deltas.append(m["delta_acc"] * m["n"])
            ns.append(m["n"])
        total_n = sum(ns)
        print(
            f"  {label:<30} {sum(rates)/total_n:>12.1%} "
            f"{sum(accs)/total_n:>8.1%} {sum(deltas)/total_n:>+8.1%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="MCQ Latent Label Mapping experiment"
    )
    parser.add_argument("--config",
                        default="configs/mcq_latent_label_mapping_deepseek.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
