"""S4 Word Content Ablation — random-word variants (paper §C1, Figure 2 right).

Tests whether Abstention Inflation is driven by the *semantic content* of the
``Unknown`` option, or by the mere structural presence of an extra slot.

Three conditions on identical paired samples:
    S1         — binary (True | False)                — baseline
    S2         — ternary (True | False | Unknown)     — main experiment
    Rand1      — ternary (True | False | Triangular)  — random control word 1
    Rand2      — ternary (True | False | Cerulean)    — random control word 2

The two random words are topic-irrelevant: any non-zero abstention into the
third slot is attributable to the slot itself (structural), not to the word
(semantic). Everything else (model, temperature T=0, CoT format, the same
samples) is held constant.

Datasets: FLD + FOLIO, 200 answerable samples each (100 True-labeled + 100
False-labeled).

Usage::

    python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_random_words.py \\
        --config configs/C1_structural_trigger/S4_random_words.yaml
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
from core.data_handler import DataHandler
from core.llm_handler import LLMHandler
from core.label_scheme import get_scheme
from core.prompts import (
    build_judge_s1_prompt,
    build_judge_s2_prompt,
    build_judge_s4_word_prompt,
)

RANDOM_WORD_1 = "Triangular"
RANDOM_WORD_2 = "Cerulean"


# =============================================================================
# Parser
# =============================================================================

def _extract_final_answer_line(text: str) -> str:
    m = re.search(r"Final answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_s1_output(text: str, scheme) -> str:
    """Binary parser for S1. Returns 'A', 'B', or 'UNPARSEABLE'."""
    if not text or not text.strip():
        return "UNPARSEABLE"
    norm = text.strip().upper()
    if norm == scheme.pos_verb.upper():
        return "A"
    if norm == scheme.neg_verb.upper():
        return "B"
    final = _extract_final_answer_line(text)
    target = final if final else text
    for pat in scheme.neg_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "B"
    for pat in scheme.pos_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "A"
    return "UNPARSEABLE"


def parse_c1_output(text: str, scheme) -> str:
    """Ternary parser for C1 (Unknown). Returns 'A', 'B', 'OPT_X', or 'UNPARSEABLE'.
    Mirrors parse_judge_tiered: ABSTAIN → OPT_X."""
    if not text or not text.strip():
        return "UNPARSEABLE"
    norm = text.strip().upper()
    if norm == scheme.pos_verb.upper():
        return "A"
    if norm == scheme.neg_verb.upper():
        return "B"
    if norm == scheme.abstain_verb.upper():
        return "OPT_X"
    final = _extract_final_answer_line(text)
    target = final if final else text
    for pat in scheme.abstain_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "OPT_X"
    for pat in scheme.neg_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "B"
    for pat in scheme.pos_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "A"
    return "UNPARSEABLE"


def parse_control_output(text: str, scheme, third_option: str) -> Tuple[str, str]:
    """Parser for C2 / C3.

    Returns (pred, category) where:
        pred     ∈ {"A", "B", "OPT_X", "UNPARSEABLE"}
        category ∈ {"explicit_opt_x",   # model said exactly third_option
                    "ghost_abstain",    # model said Unknown/Uncertain despite third_option offered
                    "pos", "neg",
                    "unparseable"}

    Both "explicit_opt_x" and "ghost_abstain" → pred = "OPT_X" (any non-binary response).
    Tracked separately so we can split them in analysis.
    """
    if not text or not text.strip():
        return "UNPARSEABLE", "unparseable"

    norm = text.strip().upper()
    if norm == scheme.pos_verb.upper():
        return "A", "pos"
    if norm == scheme.neg_verb.upper():
        return "B", "neg"
    # Exact match: third option word
    if norm == third_option.upper():
        return "OPT_X", "explicit_opt_x"

    final = _extract_final_answer_line(text)
    target = final if final else text

    # Explicit: model wrote the third_option word in its final answer
    if re.search(re.escape(third_option), target, re.IGNORECASE):
        return "OPT_X", "explicit_opt_x"

    # Ghost: model fell back to scheme's own abstain vocabulary
    for pat in scheme.abstain_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "OPT_X", "ghost_abstain"

    # Binary fallback
    for pat in scheme.neg_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "B", "neg"
    for pat in scheme.pos_patterns:
        if re.search(pat, target, re.IGNORECASE):
            return "A", "pos"

    return "UNPARSEABLE", "unparseable"


# =============================================================================
# Metrics
# =============================================================================

def compute_acc(preds: List[str], answer_idxs: List[int]) -> float:
    """Accuracy: 'A' correct iff answer_idx==0, 'B' correct iff answer_idx==1."""
    n = len(preds)
    if n == 0:
        return 0.0
    correct = sum(
        (p == "A" and ai == 0) or (p == "B" and ai == 1)
        for p, ai in zip(preds, answer_idxs)
    )
    return correct / n


def condition_metrics(preds: List[str], answer_idxs: List[int],
                      categories: List[str] = None) -> Dict:
    n = len(preds)
    acc = compute_acc(preds, answer_idxs)
    opt_x_rate = sum(1 for p in preds if p == "OPT_X") / n
    result = {
        "n": n,
        "label_acc": round(acc, 4),
        "opt_x_rate": round(opt_x_rate, 4),
        "n_opt_x": sum(1 for p in preds if p == "OPT_X"),
        "n_unparseable": sum(1 for p in preds if p == "UNPARSEABLE"),
    }
    if categories:
        result["n_explicit_opt_x"] = categories.count("explicit_opt_x")
        result["n_ghost_abstain"] = categories.count("ghost_abstain")
    return result


# =============================================================================
# Sample loading
# =============================================================================

def load_balanced_samples(data_handler: DataHandler, ds_name: str, n_per_class: int):
    """Load n_per_class proved + n_per_class disproved answerable samples."""
    all_samples = data_handler.load_dataset(ds_name)
    answerable = [s for s in all_samples if s.answer_idx >= 0]
    proved = [s for s in answerable
              if (s.extra or {}).get("native_label") == "__PROVED__"][:n_per_class]
    disproved = [s for s in answerable
                 if (s.extra or {}).get("native_label") == "__DISPROVED__"][:n_per_class]
    return proved + disproved


# =============================================================================
# Main experiment
# =============================================================================

def load_baseline_from_file(baseline_path: str, data_handler, ds_name: str):
    """Load S1/C1 results from a previous RPC output file.

    Returns (samples_ordered, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1)
    where samples_ordered matches the per_sample order in the file.
    """
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)

    per_sample = baseline["per_sample"]
    id_to_row = {row["id"]: row for row in per_sample}

    # Reload data file to get full sample objects (needed for C2/C3 prompt building)
    all_samples = data_handler.load_dataset(ds_name)
    id_to_sample = {s.id: s for s in all_samples}

    samples_ordered, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1 = [], [], [], [], [], []
    missing = []
    for row in per_sample:
        sid = row["id"]
        if sid not in id_to_sample:
            missing.append(sid)
            continue
        samples_ordered.append(id_to_sample[sid])
        answer_idxs.append(row["answer_idx"])
        preds_s1.append(row["pred_s1"])
        preds_c1.append(row["pred_c1"])
        raw_s1.append(row.get("raw_s1", ""))
        raw_c1.append(row.get("raw_c1", ""))

    if missing:
        print(f"  [warn] {len(missing)} sample IDs from baseline not found in data file.")
    print(f"  Loaded {len(samples_ordered)} samples from baseline: {baseline_path}")
    return samples_ordered, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1


async def run_one_dataset(ds_name: str, samples, llm_handler: LLMHandler,
                          results_dir: Path, model_name: str,
                          baseline: tuple = None) -> Dict:
    """Run one dataset.

    baseline: if provided, tuple of (answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1)
              loaded from a previous run — S1/C1 queries are skipped.
    """
    scheme = get_scheme(ds_name)
    n = len(samples)

    if baseline is not None:
        answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1 = baseline
        c2_prompts = [build_judge_s4_word_prompt(scheme, s.question, s.context, RANDOM_WORD_1)
                      for s in samples]
        c3_prompts = [build_judge_s4_word_prompt(scheme, s.question, s.context, RANDOM_WORD_2)
                      for s in samples]
        print(f"  Querying C2 / C3 only ({n} samples each, S1/C1 loaded from baseline) ...")
        raw_c2, raw_c3 = await asyncio.gather(
            llm_handler.batch_query(c2_prompts),
            llm_handler.batch_query(c3_prompts),
        )
    else:
        answer_idxs = [s.answer_idx for s in samples]
        # Build prompts — same infrastructure, only third-option label differs
        s1_prompts = [build_judge_s1_prompt(scheme, s.question, s.context) for s in samples]
        c1_prompts = [build_judge_s2_prompt(scheme, s.question, s.context) for s in samples]
        c2_prompts = [build_judge_s4_word_prompt(scheme, s.question, s.context, RANDOM_WORD_1)
                      for s in samples]
        c3_prompts = [build_judge_s4_word_prompt(scheme, s.question, s.context, RANDOM_WORD_2)
                      for s in samples]
        print(f"  Querying S1 / C1 / C2 / C3 in parallel ({n} samples each) ...")
        raw_s1, raw_c1, raw_c2, raw_c3 = await asyncio.gather(
            llm_handler.batch_query(s1_prompts),
            llm_handler.batch_query(c1_prompts),
            llm_handler.batch_query(c2_prompts),
            llm_handler.batch_query(c3_prompts),
        )
        preds_s1 = [parse_s1_output(r, scheme) for r in raw_s1]
        preds_c1 = [parse_c1_output(r, scheme) for r in raw_c1]

    c2_parsed = [parse_control_output(r, scheme, RANDOM_WORD_1) for r in raw_c2]
    c3_parsed = [parse_control_output(r, scheme, RANDOM_WORD_2) for r in raw_c3]
    preds_c2, cats_c2 = zip(*c2_parsed) if c2_parsed else ([], [])
    preds_c3, cats_c3 = zip(*c3_parsed) if c3_parsed else ([], [])

    s1_acc = compute_acc(preds_s1, answer_idxs)
    m_c1 = condition_metrics(list(preds_c1), answer_idxs)
    m_c2 = condition_metrics(list(preds_c2), answer_idxs, list(cats_c2))
    m_c3 = condition_metrics(list(preds_c3), answer_idxs, list(cats_c3))

    # Print table
    print(f"\n  {ds_name} results (S1_Acc={s1_acc:.1%}):")
    print(f"  {'Condition':<26} {'OPT_X Rate':>12} {'Acc':>8} {'ΔAcc':>8}")
    print(f"  {'-'*58}")
    print(f"  {'S1 (binary)':<26} {'—':>12} {s1_acc:>8.1%} {'—':>8}")
    for label, m in [
        (f"C1 Unknown",          m_c1),
        (f"C2 Random ({RANDOM_WORD_1})", m_c2),
        (f"C3 Random ({RANDOM_WORD_2})",  m_c3),
    ]:
        delta = m["label_acc"] - s1_acc
        print(f"  {label:<26} {m['opt_x_rate']:>12.1%} {m['label_acc']:>8.1%} {delta:>+8.1%}")

    result = {
        "dataset": ds_name,
        "model": model_name,
        "n_samples": n,
        "third_options": {"C2": RANDOM_WORD_1, "C3": RANDOM_WORD_2},
        "S1_acc": round(s1_acc, 4),
        "C1_Unknown": {**m_c1, "delta_acc": round(m_c1["label_acc"] - s1_acc, 4)},
        "C2_Rand1":   {**m_c2, "delta_acc": round(m_c2["label_acc"] - s1_acc, 4)},
        "C3_Rand2":   {**m_c3, "delta_acc": round(m_c3["label_acc"] - s1_acc, 4)},
        "per_sample": [
            {
                "id":         samples[i].id,
                "answer_idx": answer_idxs[i],
                "pred_s1":    preds_s1[i],
                "pred_c1":    preds_c1[i],
                "pred_c2":    preds_c2[i],
                "cat_c2":     cats_c2[i],
                "pred_c3":    preds_c3[i],
                "cat_c3":     cats_c3[i],
                "raw_s1":     raw_s1[i],
                "raw_c1":     raw_c1[i],
                "raw_c2":     raw_c2[i],
                "raw_c3":     raw_c3[i],
            }
            for i in range(n)
        ],
    }

    safe_model = model_name.replace("/", "_")
    out_path = results_dir / f"rpc_{ds_name}_{safe_model}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}")
    return result


async def run_experiment(config: Dict):
    rpc_cfg = config.get("random_perturbation_control", {})
    datasets = rpc_cfg.get("datasets", ["FLD", "FOLIO"])
    n_per_class = rpc_cfg.get("n_per_class", 100)
    results_dir = Path(rpc_cfg.get("results_dir", "results/random_perturbation_control"))
    results_dir.mkdir(parents=True, exist_ok=True)
    # Optional: {FLD: "path/to/rpc_FLD_*.json", FOLIO: "..."} — skips S1/C1 queries
    baseline_files: Dict[str, str] = rpc_cfg.get("baseline_result_files", {})

    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    model_name = config.get("model_name", "unknown")

    print(f"Model: {model_name}")
    print(f"Third options: C2={RANDOM_WORD_1!r}  C3={RANDOM_WORD_2!r}")
    if baseline_files:
        print("Baseline mode: S1/C1 loaded from existing files, only C2/C3 queried.")
    else:
        print(f"Samples per class per dataset: {n_per_class} (total: {n_per_class*2})")

    all_results = {}
    for ds_name in datasets:
        print(f"\n===== {ds_name} =====")
        if ds_name in baseline_files:
            samples, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1 = load_baseline_from_file(
                baseline_files[ds_name], data_handler, ds_name
            )
            baseline = (answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1)
        else:
            samples = load_balanced_samples(data_handler, ds_name, n_per_class)
            print(f"  Loaded {len(samples)} samples")
            baseline = None
        all_results[ds_name] = await run_one_dataset(
            ds_name, samples, llm_handler, results_dir, model_name, baseline=baseline
        )

    # Pooled summary across datasets
    print("\n\n===== POOLED SUMMARY =====")
    print(f"{'Condition':<22} {'OPT_X Rate':>12} {'Acc':>8} {'ΔAcc':>8}")
    print("-" * 54)
    for cond_key, label in [
        ("C1_Unknown", "C1 Unknown"),
        ("C2_Rand1",   f"C2 Random ({RANDOM_WORD_1})"),
        ("C3_Rand2",   f"C3 Random ({RANDOM_WORD_2})"),
    ]:
        rates, accs, deltas, ns = [], [], [], []
        for r in all_results.values():
            m = r[cond_key]
            rates.append(m["opt_x_rate"] * m["n"])
            accs.append(m["label_acc"] * m["n"])
            deltas.append(m["delta_acc"] * m["n"])
            ns.append(m["n"])
        total_n = sum(ns)
        pooled_rate  = sum(rates) / total_n
        pooled_acc   = sum(accs) / total_n
        pooled_delta = sum(deltas) / total_n
        print(f"  {label:<20} {pooled_rate:>12.1%} {pooled_acc:>8.1%} {pooled_delta:>+8.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="§4.3 Appendix: Random Perturbation Control experiment"
    )
    parser.add_argument("--config", default="configs/random_perturbation_control.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
