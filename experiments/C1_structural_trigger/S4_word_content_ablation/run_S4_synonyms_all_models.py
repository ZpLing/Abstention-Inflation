"""S4 Word Content Ablation — the four wording variants of the S2 option.

Runs W2-W5 × {FLD, FOLIO} × the full 500 samples for each model.

Output: results/wording_sweep/summary_{W}_{DS}_{MODEL}.json
"""
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(".")
sys.path.insert(0, str(ROOT))

from core.config_loader import load_config
from core.llm_handler import LLMHandler
from core.label_scheme import get_scheme
from core.dataset_loader import load_judge

WORDINGS = [
    ("W2", "I don't know"),
    ("W3", "Indeterminate"),
    ("W4", "Cannot be determined from the facts"),
    ("W5", "Insufficient information"),
]
DATASETS = ["FLD", "FOLIO"]

MODELS = [
    {
        "name": "gpt-5.4-nano",
        "config": "configs/C1_structural_trigger/TFQ_n500_GPT_5_4_nano.yaml",
        "sources": {
            "FLD":   ["tfq_n500/nano/ab_summary_FLD_gpt-5.4-nano.json"],
            "FOLIO": ["tfq_n500/nano/ab_summary_FOLIO_gpt-5.4-nano.json"],
        },
    },
    {
        "name": "gemini-3.1-flash-lite",
        "config": "configs/C1_structural_trigger/TFQ_n500_Gemini_3_1_Flash_Lite.yaml",
        "sources": {
            "FLD":   ["tfq_n500/gemini31/ab_summary_FLD_gemini-3.1-flash-lite.json"],
            "FOLIO": ["tfq_n500/gemini31/ab_summary_FOLIO_gemini-3.1-flash-lite.json"],
        },
    },
    {
        "name": "deepseek-v4-flash",
        "config": "configs/C1_structural_trigger/TFQ_n500_DeepSeek_V4_Flash.yaml",
        "sources": {
            "FLD":   ["tfq_n500/dsv4flash/ab_summary_FLD_deepseek-v4-flash.json"],
            "FOLIO": ["tfq_n500/dsv4flash/ab_summary_FOLIO_deepseek-v4-flash.json"],
        },
    },
]

OUT_DIR = ROOT / "results/wording_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_s2_wording_prompt(scheme, claim: str, context: str,
                             abstain_text: str) -> List[Dict[str, str]]:
    pos = scheme.pos_verb
    neg = scheme.neg_verb
    verb_opts = f"{pos} | {neg} | {abstain_text}"
    ctx_block = f"\n{scheme.context_label}:\n{context}\n" if context else "\n"
    body = (
        f"{scheme.task_instruction_ternary}\n"
        f"{ctx_block}"
        f"\n{scheme.claim_label}:\n{claim}\n\n"
        f"Output one of: {verb_opts}"
    )
    cot = (
        "\nFormat your response exactly as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        f"Final answer: <one of {verb_opts}>"
    )
    return [{"role": "user", "content": body + cot}]


# ── Parser ────────────────────────────────────────────────────────────────────
_FA_RE = re.compile(r"final\s*answer\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def parse_output(text: str, scheme, abstain_text: str) -> Tuple[str, str]:
    if not text:
        return "UNPARSEABLE", "unparseable"
    options = sorted(
        [(_norm(abstain_text), "UNKNOWN"), (_norm(scheme.pos_verb), "A"),
         (_norm(scheme.neg_verb), "B")],
        key=lambda x: -len(x[0])
    )
    m = _FA_RE.search(text)
    if m:
        line = _norm(m.group(1))
        for needle, label in options:
            if line == needle:
                return label, "strict_em"
        for needle, label in options:
            if needle and needle in line:
                return label, "lenient_em"
    full = _norm(text)
    for needle, label in options:
        if needle and needle in full:
            return label, "lenient_global"
    return "UNPARSEABLE", "unparseable"


# ── Sample loader ─────────────────────────────────────────────────────────────
def load_sample_ids(sources: List[str]) -> List[str]:
    """Ids of the S2 run this ablation re-words, in the order it stored them."""
    seen, ids = set(), []
    for rel in sources:
        path = ROOT / "results" / rel
        if not path.exists():
            print(f"  [warn] missing {path}")
            continue
        for ps in json.loads(path.read_text()).get("per_sample", []):
            sid = ps["id"]
            if sid not in seen:
                seen.add(sid)
                ids.append(sid)
    return ids


# ── Cell runner ───────────────────────────────────────────────────────────────
async def run_one_cell(handler: LLMHandler, scheme, samples,
                        abstain_text: str, wording_id: str,
                        dataset: str, model_name: str) -> Dict:
    prompts = [build_s2_wording_prompt(scheme, s.question, s.context, abstain_text)
               for s in samples]
    print(f"  [{wording_id}] {dataset} — querying {len(prompts)} prompts ...")
    raw = await handler.batch_query(prompts)
    parsed = [parse_output(r, scheme, abstain_text) for r in raw]
    preds = [p[0] for p in parsed]
    tiers = [p[1] for p in parsed]
    n_unk = sum(p == "UNKNOWN" for p in preds)
    abs_rate = n_unk / len(preds) if preds else 0.0
    print(f"  [{wording_id}] {dataset} → Abs Rate={abs_rate:.1%}  "
          f"(A={preds.count('A')}, B={preds.count('B')}, "
          f"UNK={n_unk}, UNP={preds.count('UNPARSEABLE')})")
    return {
        "wording_id": wording_id, "abstain_text": abstain_text,
        "dataset": dataset, "model": model_name,
        "n": len(samples), "abs_rate": abs_rate,
        "counts": {"A": preds.count("A"), "B": preds.count("B"),
                   "UNKNOWN": n_unk, "UNPARSEABLE": preds.count("UNPARSEABLE")},
        "tier_counts": {t: sum(x == t for x in tiers)
                        for t in ("strict_em", "lenient_em", "lenient_global", "unparseable")},
        "per_sample": [{"id": samples[i].id, "pred": preds[i],
                        "tier": tiers[i], "raw": raw[i]}
                       for i in range(len(samples))],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        config = load_config(str(ROOT / model_cfg["config"]))
        config["max_workers"] = 100
        config["model_name"] = model_name
        handler = LLMHandler(config)

        for ds in DATASETS:
            scheme = get_scheme(ds)
            all_samples = load_judge(ds)
            by_id = {s.id: s for s in all_samples}
            ids = load_sample_ids(model_cfg["sources"][ds])
            samples = [by_id[i] for i in ids if i in by_id]
            print(f"\n[{ds}] {len(samples)} samples loaded")

            for wording_id, abstain_text in WORDINGS:
                out_path = OUT_DIR / f"summary_{wording_id}_{ds}_{model_name}.json"
                summary = await run_one_cell(
                    handler, scheme, samples, abstain_text, wording_id, ds, model_name)
                out_path.write_text(json.dumps(summary, indent=2))
                print(f"  saved → {out_path.name} ({len(summary['per_sample'])} items)")

    print("\nAll done.")


if __name__ == "__main__":
    asyncio.run(main())
