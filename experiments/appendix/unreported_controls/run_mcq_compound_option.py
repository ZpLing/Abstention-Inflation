"""MCQ Compound Option experiment — supplement to run_mcq_latent_label_mapping.

Adds two compound conditions to the existing MCQ LLM result files:

    COMP    — E. "Both X and Y are correct"
              X = true correct answer,  Y = adjacent wrong option  (partial truth)
    COMP_WW — E. "Both X and Y are correct"
              X and Y are both wrong options                        (pure fabrication)

Comparing COMP vs COMP_WW isolates whether the OPT_X attraction comes from
the correct answer being mentioned (partial-truth trap) or from the compound
phrasing alone (structural compound effect).

Loads the aligned sample order from the existing result files, skips conditions
already present, and merges new results back into the same JSON files.

Usage:
    python -m scripts.run_mcq_compound_option \
        --config configs/mcq_latent_label_mapping_deepseek.yaml
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import load_config
from core.llm_handler import LLMHandler
from core.dataset_loader import load_arc, load_medqa
from core.prompts import build_mcq_compound_prompt, build_mcq_compound_ww_prompt


# =============================================================================
# Parser  (reuse same logic as run_mcq_latent_label_mapping)
# =============================================================================

_FINAL_ANSWER_RE = re.compile(r"Final answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FINAL_ANSWER_PREFIX = re.compile(r"^final answer\s*:\s*", re.IGNORECASE)


def _extract_final_answer_line(text: str) -> str:
    m = _FINAL_ANSWER_RE.search(text)
    if not m:
        return ""
    return _FINAL_ANSWER_PREFIX.sub("", m.group(0)).strip()


def _parse_letter(text: str, valid: str):
    if not isinstance(text, str) or not text.strip():
        return None
    upper = text.strip().upper()
    for pat in [
        rf"^\(?\s*([{valid}])\s*[\.\):,\s]",
        rf"^\(?\s*([{valid}])\s*\)?$",
        rf"ANSWER\s*(?:IS|:|=)?\s*\(?\s*([{valid}])\b",
        rf"\b([{valid}])\b",
    ]:
        m = re.search(pat, upper)
        if m:
            return m.group(1)
    return None


def parse_compound_output(text: str, compound_text: str) -> Tuple[str, str]:
    """Returns (pred, category).

    pred     ∈ {"A","B","C","D","OPT_X","UNPARSEABLE"}
    category ∈ {"explicit_opt_x",   model said E or the compound phrase
                "ghost_abstain",    model said Unknown/Uncertain
                "letter",           A/B/C/D
                "unparseable"}
    """
    if not text or not text.strip():
        return "UNPARSEABLE", "unparseable"

    final = _extract_final_answer_line(text)
    target = final if final else text

    if _parse_letter(target, "E") == "E":
        return "OPT_X", "explicit_opt_x"

    # Partial match: model writes one of the keywords from compound_text
    # e.g. "Both A and B", or just "A and B"
    if re.search(re.escape(compound_text), target, re.IGNORECASE):
        return "OPT_X", "explicit_opt_x"
    if re.search(r"\bboth\b.{1,10}\band\b", target, re.IGNORECASE):
        return "OPT_X", "explicit_opt_x"

    if re.search(r"\bunknown\b|\buncertain\b", target, re.IGNORECASE):
        return "OPT_X", "ghost_abstain"

    letter = _parse_letter(target, "ABCD")
    if letter:
        return letter, "letter"

    letter = _parse_letter(text, "ABCDE")
    if letter == "E":
        return "OPT_X", "explicit_opt_x"
    if letter in ("A", "B", "C", "D"):
        return letter, "letter"

    return "UNPARSEABLE", "unparseable"


def compute_acc(preds: List[str], answer_idxs: List[int]) -> float:
    if not preds:
        return 0.0
    return sum(
        p == chr(ord("A") + ai) for p, ai in zip(preds, answer_idxs)
    ) / len(preds)


# =============================================================================
# Dataset loading (aligned to existing result file order)
# =============================================================================

def load_samples_aligned(ds_name: str, per_sample_ids: List[str]):
    """Load samples in the same order as the existing result file."""
    if ds_name.startswith("ARC-"):
        all_samples = load_arc(ds_name)
    elif ds_name.startswith("MedQA"):
        all_samples = load_medqa(ds_name)
    else:
        raise ValueError(f"Unsupported dataset: {ds_name}")

    id_map = {s.id: s for s in all_samples}
    ordered = []
    missing = []
    for sid in per_sample_ids:
        if sid in id_map:
            ordered.append(id_map[sid])
        else:
            missing.append(sid)
    if missing:
        print(f"  [warn] {len(missing)} sample IDs not found in {ds_name}.")
    return ordered


# =============================================================================
# Per-dataset runner
# =============================================================================

def _build_metrics(preds, answer_idxs, cats, s1_acc):
    n = len(preds)
    acc = compute_acc(preds, answer_idxs)
    n_opt_x = sum(1 for p in preds if p == "OPT_X")
    return {
        "n": n,
        "label_acc": round(acc, 4),
        "opt_x_rate": round(n_opt_x / n, 4),
        "n_opt_x": n_opt_x,
        "n_unparseable": sum(1 for p in preds if p == "UNPARSEABLE"),
        "n_explicit_opt_x": list(cats).count("explicit_opt_x"),
        "n_ghost_abstain":  list(cats).count("ghost_abstain"),
        "delta_acc": round(acc - s1_acc, 4),
    }


async def run_one_dataset(result_path: Path, llm_handler: LLMHandler,
                          model_name: str):
    with open(result_path, encoding="utf-8") as f:
        existing = json.load(f)

    ds_name = existing["dataset"]
    per_sample = existing["per_sample"]
    ids = [row["id"] for row in per_sample]
    answer_idxs = [row["answer_idx"] for row in per_sample]
    s1_acc = existing["S1_acc"]

    samples = load_samples_aligned(ds_name, ids)
    n = len(samples)
    print(f"  {ds_name}: {n} samples aligned from existing result.")

    # --- COMP (correct + wrong) — skip if already present ---
    if "COMP_compound" not in existing:
        comp_prompts, comp_texts = zip(*[
            build_mcq_compound_prompt(s.question, s.options, ai)
            for s, ai in zip(samples, answer_idxs)
        ])
        print(f"  Querying COMP (correct+wrong) ({n} samples) ...")
        raw_comp = await llm_handler.batch_query(list(comp_prompts))
        comp_parsed = [parse_compound_output(r, ct) for r, ct in zip(raw_comp, comp_texts)]
        preds_comp, cats_comp = zip(*comp_parsed)
        existing["COMP_compound"] = _build_metrics(list(preds_comp), answer_idxs,
                                                    list(cats_comp), s1_acc)
        for i, row in enumerate(per_sample):
            row["pred_comp"]      = preds_comp[i]
            row["cat_comp"]       = cats_comp[i]
            row["raw_comp"]       = raw_comp[i]
            row["compound_text"]  = comp_texts[i]
    else:
        print(f"  COMP already present — skipping.")

    # --- COMP_WW (wrong + wrong) ---
    if "COMP_WW" not in existing:
        ww_prompts, ww_texts = zip(*[
            build_mcq_compound_ww_prompt(s.question, s.options, ai)
            for s, ai in zip(samples, answer_idxs)
        ])
        print(f"  Querying COMP_WW (wrong+wrong) ({n} samples) ...")
        raw_ww = await llm_handler.batch_query(list(ww_prompts))
        ww_parsed = [parse_compound_output(r, ct) for r, ct in zip(raw_ww, ww_texts)]
        preds_ww, cats_ww = zip(*ww_parsed)
        existing["COMP_WW"] = _build_metrics(list(preds_ww), answer_idxs,
                                              list(cats_ww), s1_acc)
        for i, row in enumerate(per_sample):
            row["pred_ww"]     = preds_ww[i]
            row["cat_ww"]      = cats_ww[i]
            row["raw_ww"]      = raw_ww[i]
            row["ww_text"]     = ww_texts[i]
    else:
        print(f"  COMP_WW already present — skipping.")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\n  {ds_name} results (S1_Acc={s1_acc:.1%}):")
    print(f"  {'Condition':<34} {'OPT_X':>8} {'Acc':>7} {'ΔAcc':>7}  {'explicit':>9}")
    print(f"  {'-'*74}")
    for k, label in [
        ("S2_Unknown",    "S2 Unknown"),
        ("NOTA",          "NOTA (None of the above)"),
        ("C2_Triangular", "C2 Triangular"),
        ("C3_Cerulean",   "C3 Cerulean"),
        ("COMP_compound", "COMP correct+wrong"),
        ("COMP_WW",       "COMP_WW wrong+wrong"),
    ]:
        if k not in existing:
            continue
        m = existing[k]
        exp = m.get("n_explicit_opt_x", "—")
        print(f"  {label:<34} {m['opt_x_rate']:>8.1%} {m['label_acc']:>7.1%} "
              f"{m['delta_acc']:>+7.1%}  {str(exp):>9}")

    print(f"  Saved → {result_path}")
    return existing.get("COMP_WW", {})


# =============================================================================
# Main
# =============================================================================

async def run_experiment(config: Dict):
    cfg         = config.get("mcq_latent_label_mapping", {})
    results_dir = Path(cfg.get("results_dir", "results/mcq_latent_label_mapping"))
    model_name  = config.get("model_name", "unknown")
    safe_model  = model_name.replace("/", "_")

    llm_handler = LLMHandler(config)

    result_files = sorted(results_dir.glob(f"mcq_llm_*_{safe_model}.json"))
    if not result_files:
        print(f"No result files found in {results_dir} for model {model_name}.")
        return

    print(f"Model: {model_name}")
    print(f"Found {len(result_files)} result file(s): {[f.name for f in result_files]}")

    all_metrics = {}
    for rf in result_files:
        print(f"\n===== {rf.stem} =====")
        all_metrics[rf.stem] = await run_one_dataset(rf, llm_handler, model_name)

    # Reload merged files for pooled summary
    print("\n\n===== POOLED SUMMARY =====")
    print(f"  {'Condition':<34} {'OPT_X':>8} {'Acc':>7} {'ΔAcc':>7}")
    print(f"  {'-'*60}")
    for cond_key, label in [
        ("COMP_compound", "COMP correct+wrong"),
        ("COMP_WW",       "COMP_WW wrong+wrong"),
    ]:
        totals = {"opt_x": 0.0, "acc": 0.0, "delta": 0.0, "n": 0}
        for rf in result_files:
            with open(rf, encoding="utf-8") as f:
                d = json.load(f)
            if cond_key not in d:
                continue
            m = d[cond_key]
            totals["opt_x"]  += m["opt_x_rate"] * m["n"]
            totals["acc"]    += m["label_acc"]   * m["n"]
            totals["delta"]  += m["delta_acc"]   * m["n"]
            totals["n"]      += m["n"]
        if totals["n"] == 0:
            continue
        tn = totals["n"]
        print(f"  {label:<34} {totals['opt_x']/tn:>8.1%} {totals['acc']/tn:>7.1%} "
              f"{totals['delta']/tn:>+7.1%}")


def main():
    parser = argparse.ArgumentParser(description="MCQ Compound Option experiment")
    parser.add_argument("--config",
                        default="configs/mcq_latent_label_mapping_deepseek.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
