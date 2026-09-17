"""Exp 5 / R1 — Option-Presence Logit Calibration (CoT variant).

R1 corrects the structural prior phi_struct(Y^+e) of Theorem 1 by subtracting
the mean abstention logprob (estimated on a calibration split) from the
abstention-token logprob at inference time. A constrained-output prompt
suppresses the abstention effect entirely (smoke run: S2 Abs Rate=0%, paper
reports 30-60%), so this script targets the *final-answer position* of a
chain-of-thought generation: after "Final answer:", we read the logprob of
the abstention verb token and apply the calibration there.

Method (per model x dataset):
    1. Sample N answerable items; split 50/50 calibration / test.
    2. Build paper-style CoT prompts:
         S1: "Reasoning:... / Final answer: <True|False>"
         S2: same with "<True|False|Unknown>"
       Query with logprobs=True, top_logprobs=20, max_tokens=1024,
       return_all_positions=True so we get per-token logprobs across the
       whole generation.
    3. For each generation, find the token position immediately after
       "Final answer:" (first non-whitespace content token). Extract the
       top-20 logprobs at that position. Read off:
           lp[True] = max over tokens t in top-20 whose normalized form
                      starts with "True" / "T" / "tr"...
           lp[False] = ... "False" / "F" / ...
           lp[Unknown] = ... "Unknown" / "U" / ...
    4. Calibration: beta_hat = mean over calibration items of lp[Unknown] (S2).
    5. R1 inference on test split: subtract beta_hat from Unknown-logprob,
       argmax over {True, False, Unknown}.
    6. Report Acc, Abs Rate for S1 / S2 baseline / R1 corrected.

Usage:
    python -m core.appendix_remedy_r1_logit_calibration --config configs/r1_deepseek.yaml
"""
import argparse
import asyncio
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from infra.llm_handler import LLMHandler


ABSTAIN_VERB = "Unknown"
_IDX_TO_VERB = {0: "True", 1: "False"}


def load_dataset(name: str, n: int = 100) -> List[Dict[str, Any]]:
    """Load up to ``n`` answerable items from the bundled paper dataset.

    Reads ``software/dataset/<name>.json`` via the unified loader and shapes
    each item into the local ``{id, facts, conclusion, gold_verb}`` record
    used by the S5 multi-turn rerun.
    """
    from infra.dataset_loader import answerable_subset, load_dataset as _load
    samples = answerable_subset(_load(name))
    out: List[Dict[str, Any]] = []
    for s in samples:
        verb = _IDX_TO_VERB.get(s.answer_idx)
        if verb is None:
            continue
        out.append({
            "id": s.id,
            "facts": s.context or "",
            "conclusion": s.question or "",
            "gold_verb": verb,
        })
        if len(out) >= n:
            break
    return out


def build_cot_prompt(item: Dict[str, str], with_unknown: bool) -> List[Dict[str, str]]:
    if with_unknown:
        verb_opts = "True, False, Unknown"
        task = ("Given the following facts, classify whether the conclusion is "
                "True (proved by the facts), False (disproved by the facts), or "
                "Unknown (the facts do not determine the conclusion).")
    else:
        verb_opts = "True, False"
        task = ("Given the following facts, classify whether the conclusion is "
                "True (proved by the facts) or False (disproved by the facts).")
    user = (
        f"{task}\n\n"
        f"Facts:\n{item['facts']}\n\n"
        f"Conclusion:\n{item['conclusion']}\n\n"
        f"Format your response exactly as:\n"
        f"Reasoning: <your step-by-step reasoning>\n"
        f"Final answer: <one of {verb_opts}>"
    )
    return [{"role": "user", "content": user}]


# Final-answer marker: tolerate "Final answer:", "Final Answer :", etc.
_FINAL_RE = re.compile(r"(?i)final\s*answer\s*[:\-=]\s*")


def find_answer_position(tokens: List[Dict[str, Any]]) -> Optional[int]:
    """Walk the token stream concatenating tokens; return the index of the
    first content token strictly after the regex match of 'Final answer:'.

    The match may straddle multiple tokens. We find where 'final answer:' ends
    in the concatenated string, then map that character offset back to a token
    index by tracking cumulative token lengths.
    """
    if not tokens:
        return None
    # Build text with per-token end offsets.
    text_parts = []
    end_offsets = []
    running = 0
    for t in tokens:
        s = t["token"]
        running += len(s)
        text_parts.append(s)
        end_offsets.append(running)
    text = "".join(text_parts)
    m = _FINAL_RE.search(text)
    if not m:
        return None
    target_char = m.end()
    # First token whose end offset is strictly > target_char and whose
    # content is non-whitespace.
    for i, eo in enumerate(end_offsets):
        if eo > target_char:
            tok = tokens[i]["token"]
            if tok.strip():
                return i
            # If this is whitespace, keep walking.
            for j in range(i + 1, len(tokens)):
                if tokens[j]["token"].strip():
                    return j
            return None
    return None


def extract_verb_logprobs(top_logprobs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return {"True","False","Unknown"} -> max logprob across matching tokens."""
    out = {"True": -math.inf, "False": -math.inf, "Unknown": -math.inf}
    for entry in top_logprobs:
        tok = entry["token"]
        # Normalize: strip whitespace and markdown asterisks; case-insensitive.
        norm = tok.strip().lstrip("*_(\"'`").rstrip("*_)\"'`").lower()
        if not norm:
            continue
        match = None
        # Multi-char prefixes
        if norm.startswith("true") or norm == "t":
            match = "True"
        elif norm.startswith("false") or norm == "f":
            match = "False"
        elif norm.startswith("unknown") or norm == "u" or norm.startswith("unk"):
            match = "Unknown"
        if match is not None and entry["logprob"] > out[match]:
            out[match] = entry["logprob"]
    return out


def argmax_verb(lp: Dict[str, float]) -> str:
    if all(math.isinf(v) for v in lp.values()):
        return "UNPARSEABLE"
    return max(lp.keys(), key=lambda k: lp[k])


def evaluate(rows: List[Dict[str, Any]], pred_key: str) -> Dict[str, float]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "acc": 0.0, "abs_rate": 0.0}
    n_correct = sum(1 for r in rows if r.get(pred_key) == r.get("gold_verb"))
    n_abs = sum(1 for r in rows if r.get(pred_key) == ABSTAIN_VERB)
    return {"n": n, "acc": n_correct / n, "abs_rate": n_abs / n}


async def run_one(llm: LLMHandler, dataset: str, n_samples: int,
                  calib_split: float, max_tokens: int) -> Dict[str, Any]:
    items = load_dataset(dataset, n=n_samples)
    if not items:
        return {"dataset": dataset, "error": "no items"}
    print(f"[{dataset}] loaded {len(items)} items")

    msgs_s1 = [build_cot_prompt(it, with_unknown=False) for it in items]
    msgs_s2 = [build_cot_prompt(it, with_unknown=True) for it in items]

    print(f"[{dataset}] querying S1_CoT x {len(msgs_s1)} ...")
    out_s1 = await llm.batch_query_with_logprobs(
        msgs_s1, top_logprobs=20, max_tokens=max_tokens, return_all_positions=True,
    )
    print(f"[{dataset}] querying S2_CoT x {len(msgs_s2)} ...")
    out_s2 = await llm.batch_query_with_logprobs(
        msgs_s2, top_logprobs=20, max_tokens=max_tokens, return_all_positions=True,
    )

    rows = []
    n_parse_fail = {"s1": 0, "s2": 0}
    for it, r1, r2 in zip(items, out_s1, out_s2):
        # S1 final-answer position + logprobs (no Unknown by construction).
        pos1 = find_answer_position(r1["tokens"])
        lp_s1 = (extract_verb_logprobs(r1["tokens"][pos1]["top_logprobs"])
                 if pos1 is not None else {"True": -math.inf, "False": -math.inf, "Unknown": -math.inf})
        if pos1 is None:
            n_parse_fail["s1"] += 1
        # S2
        pos2 = find_answer_position(r2["tokens"])
        lp_s2 = (extract_verb_logprobs(r2["tokens"][pos2]["top_logprobs"])
                 if pos2 is not None else {"True": -math.inf, "False": -math.inf, "Unknown": -math.inf})
        if pos2 is None:
            n_parse_fail["s2"] += 1
        # S1 argmax is restricted to True/False.
        lp_s1_bin = {k: lp_s1[k] for k in ("True", "False")}
        pred_s1 = argmax_verb({**lp_s1_bin, "Unknown": -math.inf})
        pred_s2_raw = argmax_verb(lp_s2)
        rows.append({
            "id": it["id"],
            "gold_verb": it["gold_verb"],
            "raw_s1": r1["content"][-300:] if r1["content"] else "",
            "raw_s2": r2["content"][-300:] if r2["content"] else "",
            "lp_s1": lp_s1,
            "lp_s2": lp_s2,
            "pos_s1": pos1,
            "pos_s2": pos2,
            "pred_s1": pred_s1,
            "pred_s2_raw": pred_s2_raw,
        })

    # Calibration / test split.
    n_calib = int(len(rows) * calib_split)
    calib = rows[:n_calib]
    test = rows[n_calib:]

    # beta_hat from S2 lp[Unknown] over calibration, dropping -inf.
    c_lps = [r["lp_s2"]["Unknown"] for r in calib if not math.isinf(r["lp_s2"]["Unknown"])]
    if not c_lps:
        beta_hat = 0.0
        beta_n = 0
    else:
        beta_hat = sum(c_lps) / len(c_lps)
        beta_n = len(c_lps)

    # R1 on test: subtract beta_hat from Unknown lp, argmax.
    for r in test:
        lp = dict(r["lp_s2"])
        if not math.isinf(lp["Unknown"]):
            lp["Unknown"] = lp["Unknown"] - beta_hat
        r["lp_r1"] = lp
        r["pred_r1"] = argmax_verb(lp)

    m_s1 = evaluate(test, "pred_s1")
    m_s2 = evaluate(test, "pred_s2_raw")
    m_r1 = evaluate(test, "pred_r1")

    return {
        "dataset": dataset,
        "model": llm.model_name,
        "n_total": len(rows),
        "n_calib": len(calib),
        "n_test": len(test),
        "n_parse_fail": n_parse_fail,
        "beta_hat": beta_hat,
        "beta_hat_n_samples_used": beta_n,
        "metrics": {
            "S1_CoT": m_s1,
            "S2_CoT": m_s2,
            "R1": {
                **m_r1,
                "delta_acc_vs_s2": m_r1["acc"] - m_s2["acc"],
                "delta_abs_vs_s2": m_r1["abs_rate"] - m_s2["abs_rate"],
            },
        },
        "per_sample": rows,
    }


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # runners/<this file> -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    secrets_path = repo_root / cfg.get("secrets_path", "secrets.yaml")
    if secrets_path.exists():
        with open(secrets_path) as f:
            sec = yaml.safe_load(f) or {}
        cfg.setdefault("api_key", sec.get("api_key"))
        cfg.setdefault("base_url", sec.get("base_url"))
    gateway_path = repo_root / "config"
    if gateway_path.exists():
        with open(gateway_path) as f:
            extra = yaml.safe_load(f) or {}
        llm_cfg = extra.get("llm", {}) if isinstance(extra, dict) else {}
        if "api_key" in llm_cfg:
            cfg["api_key"] = llm_cfg["api_key"]
        if "base_url" in llm_cfg:
            cfg["base_url"] = llm_cfg["base_url"]
        if not cfg.get("override_model") and "model" in llm_cfg:
            cfg["model_name"] = llm_cfg["model"]
    if not cfg.get("api_key"):
        cfg["api_key"] = os.environ.get("OPENAI_API_KEY")
    if not cfg.get("base_url"):
        cfg["base_url"] = os.environ.get("OPENAI_BASE_URL")
    return cfg


async def main_async(config_path: str):
    cfg = load_config(config_path)
    r1_cfg = cfg.get("r1_experiment", {})
    datasets = r1_cfg.get("datasets", ["FLD", "FOLIO", "FEVER"])
    n_samples = r1_cfg.get("n_samples", 100)
    calib_split = r1_cfg.get("calib_split", 0.5)
    cot_max_tokens = r1_cfg.get("cot_max_tokens", 1024)
    out_dir = Path(r1_cfg.get("results_dir", "results/remedy_r1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMHandler(cfg)
    overview = []
    for ds in datasets:
        res = await run_one(llm, ds, n_samples, calib_split, cot_max_tokens)
        model_safe = (cfg["model_name"] or "unknown").replace("/", "_")
        out_path = out_dir / f"r1_summary_{ds}_{model_safe}.json"
        out_path.write_text(json.dumps(res, indent=2))
        print(f"  -> wrote {out_path}")
        overview.append({
            "model": cfg["model_name"],
            "dataset": ds,
            **{k: res["metrics"][k] for k in ("S1_CoT", "S2_CoT", "R1")},
            "beta_hat": res["beta_hat"],
            "n_parse_fail": res["n_parse_fail"],
        })

    print("\n=== Overview ===")
    for row in overview:
        m = row
        s1 = m["S1_CoT"]; s2 = m["S2_CoT"]; r1 = m["R1"]
        print(f"  {m['model']} / {m['dataset']}: "
              f"S1={s1['acc']*100:.1f}%  S2={s2['acc']*100:.1f}% (abs {s2['abs_rate']*100:.1f}%)  "
              f"R1={r1['acc']*100:.1f}% (abs {r1['abs_rate']*100:.1f}%)  "
              f"ΔAcc={r1['delta_acc_vs_s2']*100:+.1f}pp  β̂={m['beta_hat']:.2f}  "
              f"parse_fail={m['n_parse_fail']}")

    (out_dir / "r1_overview.json").write_text(json.dumps(overview, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    asyncio.run(main_async(args.config))


if __name__ == "__main__":
    main()
