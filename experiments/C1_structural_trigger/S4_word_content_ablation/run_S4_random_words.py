"""S4 Word Content Ablation — random-word variants.

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

Datasets: FLD + FOLIO, 500 answerable samples each (250 True-labeled + 100
False-labeled).

Usage::

    python experiments/C1_structural_trigger/S4_word_content_ablation/run_S4_random_words.py \\
        --config configs/C1_structural_trigger/S4_random_words_GPT_5_4_nano.yaml
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from loader.config_loader import load_config
from loader.data_handler import DataHandler
from infra.llm_handler import LLMHandler
from infra.label_scheme import get_scheme
# imported by name: a parameter in this module is also called third_option
from infra.third_option import RANDOM_WORDS, result_path, results_dir as results_path_dir
from infra.result_schema import stamp, load_cell
from loader.config_loader import get_block
from infra.prompts import (
    build_judge_s1_prompt,
    build_judge_s2_prompt,
    build_judge_s4_word_prompt,
)

#: Both words, and the fact that they route to S4 rather than S2, come
#: from third_option so the two halves of S4 cannot drift apart again.
RANDOM_WORD_1, RANDOM_WORD_2 = RANDOM_WORDS


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
    """Load n_per_class True + n_per_class False answerable samples.

    The class is read from `answer_idx` (0 = True, 1 = False), which is what
    the unified dataset carries; the `native_label` field this used to filter
    on belonged to the pre-rename schema and is not in the shipped files, so
    the filter matched nothing and every cell came back empty.
    """
    all_samples = data_handler.load_dataset(ds_name)
    answerable = [s for s in all_samples if s.answer_idx >= 0]
    pos = [s for s in answerable if s.answer_idx == 0][:n_per_class]
    neg = [s for s in answerable if s.answer_idx == 1][:n_per_class]
    return pos + neg


# =============================================================================
# Main experiment
# =============================================================================

def load_baseline_from_main(ds_name: str, model_name: str, model_slug: str, data_handler):
    """S1 and the Unknown condition for this cell, read from the main table.

    The Unknown condition of this control *is* S2 -- c1_prompts is
    build_judge_s2_prompt -- so the main experiment already holds both baselines
    for the same 500 items. Reusing them keeps the control on the exact S2 cell
    the paper reports and spares two of the four query passes.

    Returns (samples_ordered, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1)
    in the main table's per_sample order.
    """
    cell = load_cell(ds_name, model_name.replace("/", "_"), model_slug, "tf")
    all_samples = data_handler.load_dataset(ds_name)
    id_to_sample = {s.id: s for s in all_samples}

    samples_ordered, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1 = [], [], [], [], [], []
    missing = []
    for row in cell["per_sample"]:
        sid = row["id"]
        if sid not in id_to_sample:
            missing.append(sid)
            continue
        samples_ordered.append(id_to_sample[sid])
        answer_idxs.append(row["answer_idx"])
        preds_s1.append(row.get("pred_s1"))
        preds_c1.append(row.get("pred_s2"))
        raw_s1.append(row.get("raw_s1") or "")
        raw_c1.append(row.get("raw_s2") or "")
    if missing:
        print(f"  [warn] {len(missing)} ids from the main table not found in the data file.")
    print(f"  Loaded {len(samples_ordered)} samples from the main table ({model_slug}/{ds_name}).")
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

    # One file per substituted word, the same shape the synonym sweep writes,
    # so the two halves of S4 are read by one code path.
    # Only the substituted words are written. The Unknown condition this run
    # also collects is the S2 prompt (c1_prompts is build_judge_s2_prompt), and
    # the S2 cell of the main table is the baseline both halves of S4 compare
    # against; storing a second copy here gave the two halves two baselines.
    conditions = [
        (RANDOM_WORD_1.lower(), RANDOM_WORD_1, m_c2, preds_c2, raw_c2, cats_c2),
        (RANDOM_WORD_2.lower(), RANDOM_WORD_2, m_c3, preds_c3, raw_c3, cats_c3),
    ]
    safe_model = model_name.replace("/", "_")
    written = {}
    for word, text, m, preds, raws, cats in conditions:
        rows = []
        for i in range(n):
            row = {
                "id":         samples[i].id,
                "answer_idx": answer_idxs[i],
                "pred":       preds[i],
                "raw":        raws[i],
                "pred_s1":    preds_s1[i],
                "raw_s1":     raw_s1[i],
            }
            if cats is not None:
                row["cat"] = cats[i]
            rows.append(row)
        out = {
            **stamp("S4/random_words"),
            "wording_id":   word,
            "abstain_text": text,
            "dataset":      ds_name,
            "model":        model_name,
            "n":            m["n"],
            "abs_rate":     m["opt_x_rate"],
            "label_acc":    m["label_acc"],
            "delta_acc":    round(m["label_acc"] - s1_acc, 4),
            "s1_acc":       round(s1_acc, 4),
            **{k: v for k, v in m.items()
               if k not in ("n", "opt_x_rate", "label_acc", "delta_acc")},
            "per_sample":   rows,
        }
        out_path = results_dir / result_path(text, ds_name, model_name).name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  Saved → {out_path}")
        written[word] = out
    result = written
    return result


async def run_experiment(config: Dict):
    rpc_cfg = get_block(config, "s4_random_words")
    datasets = rpc_cfg.get("datasets", ["FLD", "FOLIO"])
    n_per_class = rpc_cfg.get("n_per_class", 100)
    results_dir = Path(rpc_cfg.get("results_dir", results_path_dir(RANDOM_WORD_1)))
    results_dir.mkdir(parents=True, exist_ok=True)
    # With model_slug set, S1 and the Unknown condition are read from the main
    # table for the same items and only the two random words are queried.
    model_slug = rpc_cfg.get("model_slug")

    data_handler = DataHandler(config)
    llm_handler = LLMHandler(config)
    model_name = config.get("model_name", "unknown")

    print(f"Model: {model_name}")
    print(f"Third options: C2={RANDOM_WORD_1!r}  C3={RANDOM_WORD_2!r}")
    if model_slug:
        print("Baseline mode: S1/Unknown taken from the main table, only the random words are queried.")
    else:
        print(f"Samples per class per dataset: {n_per_class} (total: {n_per_class*2})")

    all_results = {}
    for ds_name in datasets:
        print(f"\n===== {ds_name} =====")
        if model_slug:
            samples, answer_idxs, preds_s1, preds_c1, raw_s1, raw_c1 = load_baseline_from_main(
                ds_name, model_name, model_slug, data_handler
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
        ("unknown", "Unknown"),
        (RANDOM_WORD_1.lower(), f"Random ({RANDOM_WORD_1})"),
        (RANDOM_WORD_2.lower(), f"Random ({RANDOM_WORD_2})"),
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
        description="S4 Word Content Ablation — random-word control"
    )
    parser.add_argument("--config", default="configs/C1_structural_trigger/S4_random_words_GPT_5_4_nano.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
